"""The models.yaml keys as CLI flags, for deploying models and pipelines without a
config file.

`--model REF` opens a model and `--pipeline NAME` a pipeline, ffmpeg style: the
`--model.<key>` or `--pipeline.<key>` flags right after it, up to the next
`--model` or `--pipeline`, set keys of that entry. Every key of the schema has
one, at any depth, named for its config path (`--model.num-gpus`,
`--model.llama-server-config.n-ctx`), a block's own key included. A value is YAML
text: the same thing you'd write after the colon in the file. A block's flag sets
the whole block, and deeper flags then set keys inside it. Nothing is coerced or
checked on the way through: pydantic sees what YAML parsed, from either surface.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from modelship.utils.config_schema import PIPELINE_TYPES, ModelshipModelConfig


@dataclass(frozen=True)
class ModelArg:
    """One generated flag and the config path it writes to."""

    path: tuple[str, ...]
    option: str
    metavar: str
    help: str


def _block_model(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel a field holds, unwrapping `X | None`; None for a leaf."""
    candidates = get_args(annotation) if get_origin(annotation) in (Union, UnionType) else (annotation,)
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _type_label(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "").replace("NoneType", "None")


_METAVARS = {"dict": "MAP", "list": "LIST", "Literal": "VALUE", "Any": "VALUE"}


def _metavar(annotation: Any) -> str:
    """Carries the field's type, so the help text only has to carry its default."""
    if _block_model(annotation) is not None:
        return "MAP"
    base = _type_label(annotation).split(" | ")[0].split("[")[0]
    return _METAVARS.get(base, base.upper())


def _literal_choices(annotation: Any) -> str:
    for candidate in (annotation, *get_args(annotation)):
        if get_origin(candidate) is Literal:
            return " | ".join(str(v) for v in get_args(candidate)) + "; "
    return ""


def _option(prefix: str, path: tuple[str, ...]) -> str:
    return prefix + ".".join(path).replace("_", "-")


def _model_arg(prefix: str, path: tuple[str, ...], info: FieldInfo) -> ModelArg:
    if _block_model(info.annotation) is not None:
        help_text = "the whole block, as a YAML mapping"
    elif info.is_required():
        help_text = _type_label(info.annotation)
    else:
        help_text = _literal_choices(info.annotation) + f"default: {info.get_default(call_default_factory=True)!r}"
    return ModelArg(path=path, option=_option(prefix, path), metavar=_metavar(info.annotation), help=help_text)


def _generate(
    model: type[BaseModel], prefix: str, excluded: tuple[str, ...], parent: tuple[str, ...] = ()
) -> list[ModelArg]:
    args: list[ModelArg] = []
    for name, info in model.model_fields.items():
        path = (*parent, name)
        if path == excluded:
            continue
        args.append(_model_arg(prefix, path, info))
        block = _block_model(info.annotation)
        if block is not None:
            args.extend(_generate(block, prefix, excluded, path))
    return args


@dataclass(frozen=True)
class GroupKind:
    """`--<flag> VALUE` opens an entry of this kind; VALUE is its `key`."""

    flag: str
    noun: str
    key: str
    metavar: str
    location: str  # the entries' list in a models.yaml
    # an entry's schemas: a pipeline's, one per type
    schemas: tuple[type[BaseModel], ...]

    @property
    def prefix(self) -> str:
        return self.flag + "."

    @property
    def opener(self) -> str:
        return f"{self.flag} {self.metavar}"


MODEL = GroupKind("--model", "model", "model", "REF", "models", (ModelshipModelConfig,))
PIPELINE = GroupKind("--pipeline", "pipeline", "name", "NAME", "pipelines", PIPELINE_TYPES)
KINDS = (MODEL, PIPELINE)


def _kind_args(kind: GroupKind) -> tuple[ModelArg, ...]:
    """The flags of every schema of *kind*; a key two schemas share gets one flag, the first's."""
    found: dict[str, ModelArg] = {}
    for schema in kind.schemas:
        for arg in _generate(schema, kind.prefix, (kind.key,)):
            found.setdefault(arg.option, arg)
    return tuple(found.values())


GENERATED_ARGS: dict[GroupKind, tuple[ModelArg, ...]] = {kind: _kind_args(kind) for kind in KINDS}
GENERATED_MODEL_ARGS = GENERATED_ARGS[MODEL]
GENERATED_PIPELINE_ARGS = GENERATED_ARGS[PIPELINE]
_BY_OPTION = {arg.option: (kind, arg) for kind, args in GENERATED_ARGS.items() for arg in args}


@dataclass
class Group:
    """One `--model REF` or `--pipeline NAME` and the flags that follow it."""

    kind: GroupKind
    value: str
    flags: list[tuple[ModelArg, Any]] = field(default_factory=list)


_HELP = {
    MODEL: (
        "Deploy models with no config file; mutually exclusive with --config. Each --model REF opens a model, and "
        "the --model.<key> flags right after it set keys of its entry. Every key of a models.yaml entry has one, at "
        "any depth, named for its config path; a value is YAML, i.e. the text you'd write after the colon in "
        "models.yaml. A block's flag sets the whole block, and deeper flags then set keys inside it.",
        "Model reference: an HF repo id, a repo id with a file selector (repo:*Q4_K_M.gguf), or a local "
        "file/directory path. Repeat it to deploy several models",
    ),
    PIPELINE: (
        "Experimental: /v1/realtime pipelines, with no config file. Each --pipeline NAME opens a pipeline, and the "
        "--pipeline.<key> flags right after it set keys of its entry, the same way --model's do.",
        "Pipeline name clients pass as /v1/realtime?model=NAME. Repeat it for several pipelines",
    ),
}


def add_group_help(parser: argparse.ArgumentParser) -> None:
    """Lists `--model`, `--pipeline` and their flags in `--help`. `split_groups`
    removes them from argv before argparse runs, so these entries only document them."""
    for kind in KINDS:
        description, value_help = _HELP[kind]
        top = parser.add_argument_group(kind.location, description)
        top.add_argument(kind.flag, dest=f"_help_{kind.noun}", metavar=kind.metavar, help=value_help)
        groups: dict[str, argparse._ArgumentGroup] = {"": top}
        for arg in GENERATED_ARGS[kind]:
            block = arg.path[0] if len(arg.path) > 1 else ""
            group = groups.get(block)
            if group is None:
                group = groups[block] = parser.add_argument_group(f"{kind.location} `{block}:` block")
            group.add_argument(arg.option, dest=f"_help_{arg.option}", metavar=arg.metavar, help=arg.help)


def _yaml_value(option: str, text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{option}: not valid YAML: {exc}") from exc


def _take_value(argv: list[str], i: int, option: str, inline: str | None) -> tuple[str, int]:
    if inline is not None:
        return inline, i + 1
    if i + 1 >= len(argv):
        raise ValueError(f"{option} takes a value.")
    return argv[i + 1], i + 2


def _scoped_kind(option: str) -> GroupKind | None:
    return next((kind for kind in KINDS if option.startswith(kind.prefix)), None)


def split_groups(argv: list[str]) -> tuple[list[str], list[Group]]:
    """Splits *argv* into the arguments argparse parses and the `--model` and
    `--pipeline` groups. Raises ValueError on a misplaced, unknown or valueless
    group flag."""
    rest: list[str] = []
    groups: list[Group] = []
    openers = {kind.flag: kind for kind in KINDS}
    i = 0
    while i < len(argv):
        option, eq, inline = argv[i].partition("=")
        value_inline = inline if eq else None
        if option in openers:
            value, i = _take_value(argv, i, option, value_inline)
            groups.append(Group(openers[option], value))
        elif (kind := _scoped_kind(option)) is not None:
            found = _BY_OPTION.get(option)
            if found is None:
                raise ValueError(f"unrecognized {kind.noun} flag {option}.")
            if not groups:
                raise ValueError(f"{option} sets a key of the {kind.noun} a {kind.opener} opens; put it after one.")
            if groups[-1].kind is not kind:
                raise ValueError(
                    f"{option} sets a key of a {kind.noun}, but follows {groups[-1].kind.opener}: "
                    f"put it right after the {kind.opener} it configures."
                )
            text, i = _take_value(argv, i, option, value_inline)
            groups[-1].flags.append((found[1], _yaml_value(option, text)))
        else:
            rest.append(argv[i])
            i += 1
    return rest, groups


def group_entry(group: Group) -> dict:
    """The raw models.yaml entry *group* describes. Unset keys are absent, not
    defaulted: validators branch on `model_fields_set`, and an untouched block
    stays absent rather than empty, which the fingerprint also reads."""
    raw: dict = {group.kind.key: group.value}
    seen: set[str] = set()
    for arg, value in sorted(group.flags, key=lambda flag: len(flag[0].path)):
        if arg.option in seen:
            raise ValueError(f"{arg.option} is given twice for {group.kind.flag} {group.value}.")
        seen.add(arg.option)
        node = raw
        for depth, key in enumerate(arg.path[:-1], start=1):
            if key not in node:
                node[key] = {}
            elif not isinstance(node[key], dict):
                parent = _option(group.kind.prefix, arg.path[:depth])
                raise ValueError(f"{arg.option}: {parent} is set to {node[key]!r}, not a mapping to set a key in.")
            node = node[key]
        node[arg.path[-1]] = value
    return raw


_LOCATION = re.compile(r"\b(models|pipelines)\.(\d+)\.([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)")


def flag_locations(text: str, counts: dict[str, int]) -> str:
    """Rewrites pydantic's `models.<i>.<path>` and `pipelines.<i>.<path>` locations
    in *text* as the flag that sets them, naming which entry when *counts* (entries
    per list) says there are several."""
    kinds = {kind.location: kind for kind in KINDS}

    def flag(match: re.Match) -> str:
        kind = kinds[match[1]]
        option = kind.prefix + match[3].replace("_", "-")
        return option if counts.get(kind.location, 0) < 2 else f"{option} ({kind.flag} #{int(match[2]) + 1})"

    return _LOCATION.sub(flag, text)
