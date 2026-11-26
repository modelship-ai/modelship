"""The TFLite C library microWakeWord runs on, on macOS.

pymicro-wakeword's only macOS wheel is tagged macosx_15_0, so on an older macOS uv installs its sdist, which ships
only the Linux library. The library inside the wheel loads and runs on macOS 14, so the bootstrapper takes it out of
the wheel, at the hash uv.lock records, and the engine gets its path in MSHIP_TFLITE_C_LIB. On Linux the package's
own library loads.
"""

from __future__ import annotations

import os
import platform
import sys
import uuid
import zipfile

from . import paths
from .fetch import FetchError, download, verify_sha256
from .variants import Variant

# The wheel uv.lock pins for macOS; tests/test_tflite_pin.py checks both against the lock.
_VERSION = "2.5.0"
_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/62/b8/209abb67371d296964899e6ff867f93fc525ef185ac3b800e33d55fc8a3a/"
    "pymicro_wakeword-2.5.0-py3-none-macosx_15_0_universal2.whl"
)
_WHEEL_SHA256 = "950336248aefc9e7d0334f1894f878a3e319e2850b165d592c69b48bc706c823"
_MEMBER = "pymicro_wakeword/lib/libtensorflowlite_c.dylib"
_LIBRARY = "libtensorflowlite_c.dylib"


class TfliteError(Exception):
    pass


def _applies(variant: Variant) -> bool:
    return variant.serves_models and platform.system() == "Darwin"


def _library_path(variant: Variant) -> str:
    return os.path.join(paths.builds_dir(variant.name), "pymicro-wakeword", _VERSION, _LIBRARY)


def provision(variant: Variant) -> str | None:
    """Takes the library out of the wheel if it isn't there yet. Returns its path; None off macOS."""
    if not _applies(variant):
        return None
    library = _library_path(variant)
    if os.path.isfile(library):
        return library
    print(f"mship: fetching the TFLite library of pymicro-wakeword {_VERSION}", flush=True)
    wheel = os.path.join(os.path.dirname(library), f"wheel.{uuid.uuid4().hex}.tmp")
    tmp = f"{library}.{uuid.uuid4().hex}.tmp"
    try:
        download(_WHEEL_URL, wheel)
        verify_sha256(wheel, _WHEEL_SHA256)
        with zipfile.ZipFile(wheel) as archive, open(tmp, "wb") as out:
            out.write(archive.read(_MEMBER))
        os.replace(tmp, library)
    except (FetchError, OSError, KeyError, zipfile.BadZipFile) as e:
        raise TfliteError(
            f"error: the TFLite library of pymicro-wakeword {_VERSION} couldn't be provisioned: {e}"
        ) from e
    finally:
        for leftover in (wheel, tmp):
            if os.path.exists(leftover):
                os.remove(leftover)
    return library


def locate(variant: Variant) -> str | None:
    """Same, but never downloads: for the commands that install nothing."""
    if not _applies(variant):
        return None
    library = _library_path(variant)
    if os.path.isfile(library):
        return library
    print(
        f"warning: no TFLite library under {os.path.dirname(library)}; wake words will not work. "
        f"Run: mship bootstrap --{variant.name}",
        file=sys.stderr,
    )
    return None
