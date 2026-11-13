import os
from unittest.mock import patch

import pytest

from modelship.utils.variant import VariantError, current_variant


class TestCurrentVariant:
    def test_reads_the_variant(self):
        with patch.dict(os.environ, {"MSHIP_VARIANT": " metal "}, clear=True):
            assert current_variant() == "metal"

    def test_accepts_vulkan(self):
        with patch.dict(os.environ, {"MSHIP_VARIANT": "vulkan"}, clear=True):
            assert current_variant() == "vulkan"

    def test_refuses_when_unset(self):
        with patch.dict(os.environ, {}, clear=True), pytest.raises(VariantError, match="MSHIP_VARIANT is not set"):
            current_variant()

    def test_refuses_an_unknown_variant(self):
        with (
            patch.dict(os.environ, {"MSHIP_VARIANT": "rocm"}, clear=True),
            pytest.raises(VariantError, match="MSHIP_VARIANT is 'rocm'"),
        ):
            current_variant()
