"""The TFLite library the bootstrapper takes out of pymicro-wakeword's macOS wheel: pinned to the wheel uv.lock
pins, provisioned on macOS only."""

import hashlib
import os
import re
import zipfile
from pathlib import Path

import pytest

from mship_bootstrap import tflite
from mship_bootstrap.variants import VARIANTS

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_wheel_is_the_macos_wheel_uv_lock_pins():
    # read as text: tomllib needs Python 3.11
    lock = (_REPO_ROOT / "uv.lock").read_text()
    assert f'name = "pymicro-wakeword"\nversion = "{tflite._VERSION}"\n' in lock
    assert f'{{ url = "{tflite._WHEEL_URL}", hash = "sha256:{tflite._WHEEL_SHA256}"' in lock


def test_every_extra_that_pins_pymicro_wakeword_pins_the_bootstrappers_version():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    assert set(re.findall(r'"pymicro-wakeword==([^"]+)"', pyproject)) == {tflite._VERSION}


@pytest.fixture
def wheel(tmp_path, monkeypatch):
    """A wheel holding a fake library, served from a file URL at its own hash."""
    path = tmp_path / "pymicro_wakeword.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(tflite._MEMBER, b"the library")
    monkeypatch.setattr(tflite, "_WHEEL_URL", path.as_uri())
    monkeypatch.setattr(tflite, "_WHEEL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setenv("MSHIP_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(tflite.platform, "system", lambda: "Darwin")
    return path


def test_on_macos_the_library_is_taken_out_of_the_wheel(wheel):
    library = tflite.provision(VARIANTS["metal"])
    assert library is not None and library.endswith("libtensorflowlite_c.dylib")
    assert Path(library).read_bytes() == b"the library"
    # the wheel is gone, the library stays
    assert os.listdir(os.path.dirname(library)) == ["libtensorflowlite_c.dylib"]
    assert tflite.locate(VARIANTS["metal"]) == library
    # a cpu node on macOS needs it too
    assert tflite.provision(VARIANTS["cpu"]) is not None


def test_a_wheel_that_isnt_the_pinned_one_is_refused(wheel, monkeypatch):
    monkeypatch.setattr(tflite, "_WHEEL_SHA256", "0" * 64)
    with pytest.raises(tflite.TfliteError, match="sha256"):
        tflite.provision(VARIANTS["metal"])
    assert tflite.locate(VARIANTS["metal"]) is None


def test_off_macos_nothing_is_provisioned(wheel, monkeypatch):
    monkeypatch.setattr(tflite.platform, "system", lambda: "Linux")
    assert tflite.provision(VARIANTS["cpu"]) is None
    assert tflite.locate(VARIANTS["cpu"]) is None


def test_thin_needs_no_library(wheel):
    assert tflite.provision(VARIANTS["thin"]) is None
