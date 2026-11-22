"""The models.yaml keys as CLI flags, for deploying models without a config file.

`--model REF` opens a model, ffmpeg style: the `--model.<key>` flags after it, up
to the next `--model`, set keys of that model's entry. Every key of the schema has
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

from modelship.utils.config_schema import ModelshipModelConfig

MODEL_FLAG = "--model"
_PREFIX = MODEL_FLAG + "."
# `--model REF` itself sets the entry's `model` key.
_EXCLUDED = {("model",)}


@dataclass(frozen=True)
class ModelArg:
    """One generated flag and the config path it writes to."""

    path: tuple[str, ...]
    option: str
    metavar: str
    help: str


@dataclass
class ModelGroup:
    """One `--model REF` and the flags that follow it."""

    ref: str
    flags: list[tuple[ModelArg, Any]] = field(default_factory=list)


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


def _model_arg(path: tuple[str, ...], info: FieldInfo) -> ModelArg:
    if _block_model(info.annotation) is not None:
        help_text = "the whole block, as a YAML mapping"
    elif info.is_required():
        help_text = _type_label(info.annotation)
    else:
        help_text = _literal_choices(info.annotation) + f"default: {info.get_default(call_default_factory=True)!r}"
    return ModelArg(
        path=path,
        option=_PREFIX + ".".join(path).replace("_", "-"),
        metavar=_metavar(info.annotation),
        help=help_text,
    )


def _generate(model: type[BaseModel], parent: tuple[str, ...] = ()) -> list[ModelArg]:
    args: list[ModelArg] = []
    for name, info in model.model_fields.items():
        path = (*parent, name)
        if path in _EXCLUDED:
            continue
        args.append(_model_arg(path, info))
        block = _block_model(info.annotation)
        if block is not None:
            args.extend(_generate(block, path))
    return args


GENERATED_MODEL_ARGS = tuple(_generate(ModelshipModelConfig))
_BY_OPTION = {arg.option: arg for arg in GENERATED_MODEL_ARGS}


def add_model_help(parser: argparse.ArgumentParser) -> None:
    """Lists `--model` and its flags in `--help`. `split_model_groups` removes them
    from argv before argparse runs, so these entries only document them."""
    models = parser.add_argument_group(
        "models",
        "Deploy models with no config file; mutually exclusive with --config. Each --model REF opens a model, and "
        "the --model.<key> flags after it, up to the next --model, set keys of its entry. Every key of a "
        "models.yaml entry has one, at any depth, named for its config path; a value is YAML, i.e. the text you'd "
        "write after the colon in models.yaml. A block's flag sets the whole block, and deeper flags then set keys "
        "inside it.",
    )
    models.add_argument(
        MODEL_FLAG,
        dest="_help_model",
        metavar="REF",
        help=(
            "Model reference: an HF repo id, a repo id with a file selector (repo:*Q4_K_M.gguf), or a local "
            "file/directory path. Repeat it to deploy several models"
        ),
    )
    groups: dict[str, argparse._ArgumentGroup] = {"": models}
    for arg in GENERATED_MODEL_ARGS:
        block = arg.path[0] if len(arg.path) > 1 else ""
        group = groups.get(block)
        if group is None:
            group = groups[block] = parser.add_argument_group(f"models.yaml `{block}:` block")
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


def split_model_groups(argv: list[str]) -> tuple[list[str], list[ModelGroup]]:
    """Splits *argv* into the arguments argparse parses and the `--model` groups.
    Raises ValueError on a misplaced, unknown or valueless model flag."""
    rest: list[str] = []
    groups: list[ModelGroup] = []
    i = 0
    while i < len(argv):
        option, eq, inline = argv[i].partition("=")
        value_inline = inline if eq else None
        if option == MODEL_FLAG:
            ref, i = _take_value(argv, i, MODEL_FLAG, value_inline)
            groups.append(ModelGroup(ref))
        elif option.startswith(_PREFIX):
            arg = _BY_OPTION.get(option)
            if arg is None:
                raise ValueError(f"unrecognized model flag {option}.")
            if not groups:
                raise ValueError(f"{option} sets a key of the model a --model REF opens; put it after one.")
            text, i = _take_value(argv, i, option, value_inline)
            groups[-1].flags.append((arg, _yaml_value(option, text)))
        else:
            rest.append(argv[i])
            i += 1
    return rest, groups


def model_entry(group: ModelGroup) -> dict:
    """The raw models.yaml entry *group* describes. Unset keys are absent, not
    defaulted: validators branch on `model_fields_set`, and an untouched block
    stays absent rather than empty, which the fingerprint also reads."""
    raw: dict = {"model": group.ref}
    seen: set[str] = set()
    for arg, value in sorted(group.flags, key=lambda flag: len(flag[0].path)):
        if arg.option in seen:
            raise ValueError(f"{arg.option} is given twice for --model {group.ref}.")
        seen.add(arg.option)
        node = raw
        for depth, key in enumerate(arg.path[:-1], start=1):
            if key not in node:
                node[key] = {}
            elif not isinstance(node[key], dict):
                parent = _PREFIX + ".".join(arg.path[:depth]).replace("_", "-")
                raise ValueError(f"{arg.option}: {parent} is set to {node[key]!r}, not a mapping to set a key in.")
            node = node[key]
        node[arg.path[-1]] = value
    return raw


_LOCATION = re.compile(r"\bmodels\.(\d+)\.([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)")


def flag_locations(text: str, count: int) -> str:
    """Rewrites pydantic's `models.<i>.<path>` locations in *text* as the flag
    that sets them, naming which --model when *count* says there are several."""

    def flag(match: re.Match) -> str:
        option = _PREFIX + match[2].replace("_", "-")
        return option if count < 2 else f"{option} (--model #{int(match[1]) + 1})"

    return _LOCATION.sub(flag, text)
