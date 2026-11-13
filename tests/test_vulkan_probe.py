from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
from unittest.mock import patch

import pytest

from modelship.utils import vulkan_probe as vp

GiB = 1024**3
_AMD, _INTEL, _NVIDIA = 0x1002, 0x8086, 0x10DE
_RADV, _AMDVLK, _AMD_PRO, _NV_PRO, _ANV, _NVK, _MOLTENVK, _LLVMPIPE = 3, 2, 1, 4, 6, 24, 14, 13


def _dev(index, type_="integrated", *, vendor=_AMD, driver=_RADV, uuid=None, fp16=True, heaps=(), render=None):
    return {
        "index": index,
        "name": f"gpu{index}",
        "type": type_,
        "vendor_id": vendor,
        "driver_id": driver,
        "uuid": uuid or f"uuid-{index}",
        "luid": None,
        "fp16_storage": fp16,
        "render": render,
        "heaps": list(heaps),
    }


def _picked(devices):
    return [d["index"] for d in vp.select_devices(devices)]


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="layouts below are for 64-bit")
class TestStructLayouts:
    @pytest.mark.parametrize(
        ("struct", "size", "offsets"),
        [
            (vp._ApplicationInfo, 48, {"apiVersion": 44}),
            (vp._InstanceCreateInfo, 64, {"pApplicationInfo": 24, "ppEnabledExtensionNames": 56}),
            (vp._ExtensionProperties, 260, {}),
            (
                vp._PhysicalDeviceProperties,
                824,
                {"deviceType": 16, "deviceName": 20, "limits": 296, "sparseProperties": 800},
            ),
            (vp._PhysicalDeviceProperties2, 840, {"properties": 16}),
            (vp._PhysicalDeviceIDProperties, 64, {"deviceUUID": 16, "deviceLUID": 48, "deviceLUIDValid": 60}),
            (vp._PhysicalDeviceDriverProperties, 536, {"driverID": 16, "conformanceVersion": 532}),
            (vp._PhysicalDeviceDrmProperties, 56, {"hasRender": 20, "renderMajor": 40, "renderMinor": 48}),
            (vp._PhysicalDeviceFeatures2, 240, {}),
            (vp._PhysicalDevice16BitStorageFeatures, 32, {"storageBuffer16BitAccess": 16}),
            (vp._MemoryHeap, 16, {}),
            (vp._MemoryType, 8, {}),
            (vp._PhysicalDeviceMemoryProperties, 520, {"memoryHeapCount": 260, "memoryHeaps": 264}),
            (vp._PhysicalDeviceMemoryProperties2, 536, {"memoryProperties": 16}),
            (vp._PhysicalDeviceMemoryBudgetProperties, 272, {"heapBudget": 16, "heapUsage": 144}),
        ],
    )
    def test_matches_the_vulkan_header(self, struct, size, offsets):
        assert ctypes.sizeof(struct) == size
        for name, offset in offsets.items():
            assert getattr(struct, name).offset == offset, name

    def test_every_chained_struct_starts_with_stype_and_pnext(self):
        for struct in (
            vp._PhysicalDeviceIDProperties,
            vp._PhysicalDeviceDriverProperties,
            vp._PhysicalDeviceDrmProperties,
            vp._PhysicalDevice16BitStorageFeatures,
            vp._PhysicalDeviceMemoryBudgetProperties,
        ):
            assert (struct.sType.offset, struct.pNext.offset) == (0, 8)


class TestSelectDevices:
    def test_an_igpu_alone_is_used_and_llvmpipe_never(self):
        assert _picked([_dev(0), _dev(1, "cpu", vendor=0x10005, driver=_LLVMPIPE)]) == [0]

    def test_llvmpipe_alone_is_no_gpu(self):
        assert _picked([_dev(0, "cpu", vendor=0x10005, driver=_LLVMPIPE)]) == []

    def test_a_discrete_gpu_hides_the_igpu(self):
        assert _picked([_dev(0, "integrated"), _dev(1, "discrete", vendor=_NVIDIA, driver=_NV_PRO)]) == [1]

    def test_every_discrete_gpu_counts_in_enumeration_order(self):
        assert _picked([_dev(0, "discrete"), _dev(1, "integrated"), _dev(2, "discrete")]) == [0, 2]

    def test_a_gpu_without_16_bit_storage_is_skipped(self):
        assert _picked([_dev(0, "discrete", fp16=False), _dev(1, "discrete")]) == [1]

    def test_with_no_usable_gpu_the_first_non_cpu_device_is_used_without_the_16_bit_check(self):
        devices = [_dev(0, "cpu"), _dev(1, "virtual", fp16=False), _dev(2, "other")]
        assert _picked(devices) == [1]

    @pytest.mark.parametrize(
        ("first", "second", "kept"),
        [
            (_AMDVLK, _RADV, 1),
            (_RADV, _AMDVLK, 0),
            (_AMD_PRO, _AMDVLK, 1),
            (_RADV, 999, 0),
        ],
    )
    def test_one_gpu_under_two_drivers_is_kept_once_by_ggml_priority(self, first, second, kept):
        devices = [_dev(0, "discrete", driver=first, uuid="same"), _dev(1, "discrete", driver=second, uuid="same")]
        assert _picked(devices) == [kept]

    def test_the_replacing_driver_moves_to_the_end_as_in_ggml(self):
        devices = [
            _dev(0, "discrete", driver=_AMDVLK, uuid="a"),
            _dev(1, "discrete", vendor=_NVIDIA, driver=_NV_PRO, uuid="b"),
            _dev(2, "discrete", driver=_RADV, uuid="a"),
        ]
        assert _picked(devices) == [1, 2]

    def test_nvidia_prefers_its_proprietary_driver_over_nvk(self):
        devices = [
            _dev(0, "discrete", vendor=_NVIDIA, driver=_NVK, uuid="n"),
            _dev(1, "discrete", vendor=_NVIDIA, driver=_NV_PRO, uuid="n"),
        ]
        assert _picked(devices) == [1]

    def test_two_moltenvk_devices_sharing_a_uuid_are_both_kept(self):
        devices = [
            _dev(0, "discrete", driver=_MOLTENVK, uuid="m"),
            _dev(1, "discrete", driver=_MOLTENVK, uuid="m"),
        ]
        assert _picked(devices) == [0, 1]


def _heap(size, *, local=False, budget=None, usage=None):
    return vp.Heap(size=size, device_local=local, budget=budget, usage=usage)


def _device(type_="integrated", *, vendor=_AMD, heaps=(), render=(226, 128)):
    return vp.VulkanDevice(
        index=0,
        name="gpu",
        type=type_,
        vendor_id=vendor,
        driver_id=_RADV,
        uuid="u",
        render=render,
        heaps=tuple(heaps),
    )


def _write_mem_info(root, render=(226, 128), **values):
    base = root / "dev" / "char" / f"{render[0]}:{render[1]}" / "device"
    base.mkdir(parents=True)
    for name, value in values.items():
        (base / f"mem_info_{name}").write_text(f"{value}\n")


_STRIX_HEAPS = (
    _heap(int(56.5 * GiB / 3), budget=int(56.5 * GiB / 3), usage=0),
    _heap(int(56.5 * GiB * 2 / 3), local=True, budget=int(56.5 * GiB * 2 / 3), usage=0),
)


class TestEngineFree:
    def test_an_igpu_counts_every_heap(self):
        heaps = [_heap(10 * GiB, budget=8 * GiB, usage=1 * GiB), _heap(20 * GiB, local=True, budget=30 * GiB, usage=0)]
        assert vp.engine_free_bytes(_device(heaps=heaps)) == 7 * GiB + 20 * GiB

    def test_a_dgpu_counts_its_device_local_heaps(self):
        heaps = [_heap(16 * GiB, local=True, budget=15 * GiB, usage=2 * GiB), _heap(64 * GiB, budget=60 * GiB, usage=0)]
        assert vp.engine_free_bytes(_device("discrete", heaps=heaps)) == 13 * GiB

    def test_usage_above_the_budget_counts_as_nothing_free(self):
        heaps = [_heap(16 * GiB, local=True, budget=1 * GiB, usage=2 * GiB)]
        assert vp.engine_free_bytes(_device("discrete", heaps=heaps)) == 0

    def test_without_the_budget_extension_the_heap_size(self):
        assert vp.engine_free_bytes(_device("discrete", heaps=[_heap(16 * GiB, local=True)])) == 16 * GiB


class TestDeviceMemory:
    def test_a_dgpu_is_its_device_local_heaps(self):
        heaps = [_heap(16 * GiB, local=True, budget=15 * GiB, usage=1 * GiB), _heap(64 * GiB, budget=60 * GiB, usage=0)]
        memory = vp.device_memory(_device("discrete", heaps=heaps), available_ram_bytes=1 * GiB)
        assert memory == vp.DeviceMemory(total_bytes=16 * GiB, free_bytes=14 * GiB, engine_free_bytes=14 * GiB)

    def test_a_dgpu_without_the_budget_extension_is_all_free(self):
        memory = vp.device_memory(_device("discrete", heaps=[_heap(8 * GiB, local=True)]), available_ram_bytes=0)
        assert memory.free_bytes == memory.total_bytes == 8 * GiB

    def test_an_amd_igpu_reads_the_kernel_files_and_caps_only_gtt(self, tmp_path):
        _write_mem_info(tmp_path, vram_total=512 * 2**20, vram_used=148 * 2**20, gtt_total=56 * GiB, gtt_used=1 * GiB)
        memory = vp.device_memory(_device(heaps=_STRIX_HEAPS), available_ram_bytes=30 * GiB, sys_root=str(tmp_path))
        assert memory.total_bytes == 512 * 2**20 + 56 * GiB
        assert memory.free_bytes == (512 - 148) * 2**20 + 30 * GiB
        assert memory.engine_free_bytes == vp.engine_free_bytes(_device(heaps=_STRIX_HEAPS))

    def test_a_large_carve_out_is_not_capped_by_the_free_ram(self, tmp_path):
        _write_mem_info(tmp_path, vram_total=96 * GiB, vram_used=0, gtt_total=16 * GiB, gtt_used=0)
        memory = vp.device_memory(_device(heaps=_STRIX_HEAPS), available_ram_bytes=28 * GiB, sys_root=str(tmp_path))
        assert memory.free_bytes == 96 * GiB + 16 * GiB

    def test_unknown_free_ram_caps_nothing(self, tmp_path):
        _write_mem_info(tmp_path, vram_total=0, vram_used=0, gtt_total=56 * GiB, gtt_used=0)
        memory = vp.device_memory(_device(heaps=_STRIX_HEAPS), available_ram_bytes=0, sys_root=str(tmp_path))
        assert memory.free_bytes == 56 * GiB

    def test_an_amd_igpu_without_a_render_node_has_no_memory_figures(self, tmp_path):
        with pytest.raises(vp.MemoryUnreadableError, match="VK_EXT_physical_device_drm"):
            vp.device_memory(_device(heaps=_STRIX_HEAPS, render=None), 0, sys_root=str(tmp_path))

    def test_an_unreadable_kernel_file_names_it_and_has_no_fallback(self, tmp_path):
        _write_mem_info(tmp_path, vram_total=0, vram_used=0, gtt_total=56 * GiB)
        with pytest.raises(vp.MemoryUnreadableError, match="mem_info_gtt_used"):
            vp.device_memory(_device(heaps=_STRIX_HEAPS), 0, sys_root=str(tmp_path))

    def test_another_igpu_is_its_heaps_capped_by_the_free_ram(self):
        heaps = [_heap(48 * GiB, local=True, budget=40 * GiB, usage=2 * GiB)]
        memory = vp.device_memory(_device(vendor=_INTEL, heaps=heaps), available_ram_bytes=20 * GiB)
        assert memory == vp.DeviceMemory(total_bytes=48 * GiB, free_bytes=20 * GiB, engine_free_bytes=38 * GiB)


def _probe_with(devices, selected):
    return vp.parse(json.dumps({"devices": devices, "selected": selected, "error": None}))


class TestParseAndVisible:
    def test_parse_orders_the_selected_devices_as_ggml_does(self):
        result = _probe_with([_dev(0), _dev(1, "discrete"), _dev(2, "discrete")], [2, 1])
        assert [d.index for d in result.devices] == [2, 1]
        assert [d.index for d in result.all_devices] == [0, 1, 2]

    def test_parse_keeps_the_render_node_and_heaps(self):
        raw = _dev(0, render=[226, 128], heaps=[{"size": 8, "device_local": True, "budget": 6, "usage": 1}])
        device = _probe_with([raw], [0]).devices[0]
        assert device.render == (226, 128)
        assert device.heaps == (vp.Heap(8, True, 6, 1),)

    def test_parse_reads_an_error(self):
        result = vp.parse(json.dumps({"devices": [], "selected": [], "error": {"stage": "loader", "message": "x"}}))
        assert result.error == vp.ProbeError("loader", "x")
        assert result.devices == []

    @pytest.mark.parametrize(
        ("mask", "expected"),
        [
            (None, [2, 0]),
            ("", []),
            ("0,2", [0, 2]),
            ("1", [1]),
        ],
    )
    def test_visible_follows_ggml_vk_visible_devices(self, mask, expected):
        result = _probe_with([_dev(0, "discrete"), _dev(1, "cpu"), _dev(2, "discrete")], [2, 0])
        assert [d.index for d in vp.visible(result, mask)] == expected


class TestGgmlVisibleDevices:
    @pytest.fixture
    def probe(self):
        devices = [vp.VulkanDevice(i, f"gpu{i}", "discrete", _AMD, _RADV, f"u{i}", None) for i in (2, 0)]
        with patch.object(vp, "probe", return_value=vp.VulkanProbe(devices=devices, all_devices=devices)) as probe:
            yield probe

    @pytest.mark.parametrize(("slots", "expected"), [(["0"], "2"), (["1"], "0"), (["1", "0"], "2,0")])
    def test_slots_become_raw_vulkan_indices(self, probe, slots, expected):
        assert vp.ggml_visible_devices(slots) == expected

    def test_no_slot_hides_every_device_without_probing(self, probe):
        assert vp.ggml_visible_devices([]) == ""
        probe.assert_not_called()


class TestChild:
    def test_prints_every_device_and_the_selection(self, capsys):
        devices = [_dev(0), _dev(1, "cpu")]
        with patch.object(vp, "_read_devices", return_value=devices):
            vp._child_main()
        out = json.loads(capsys.readouterr().out)
        assert out == {"devices": devices, "selected": [0], "error": None}

    def test_no_usable_device_is_the_no_device_stage_naming_what_was_found(self, capsys):
        with patch.object(vp, "_read_devices", return_value=[_dev(0, "cpu")]):
            vp._child_main()
        error = json.loads(capsys.readouterr().out)["error"]
        assert error["stage"] == "no_device"
        assert "gpu0 (cpu)" in error["message"]

    def test_a_failed_stage_is_reported(self, capsys):
        with patch.object(vp, "_read_devices", side_effect=vp.ProbeStageError("loader", "no libvulkan")):
            vp._child_main()
        assert json.loads(capsys.readouterr().out)["error"] == {"stage": "loader", "message": "no libvulkan"}

    def test_a_missing_loader_is_the_loader_stage(self):
        with (
            patch.object(vp.ctypes, "CDLL", side_effect=OSError("libvulkan.so.1: cannot open shared object file")),
            pytest.raises(vp.ProbeStageError) as failure,
        ):
            vp._read_devices()
        assert failure.value.stage == "loader"

    def test_runs_as_a_module_without_importing_anything_heavy(self):
        code = (
            "import sys; from unittest.mock import patch; import modelship.utils.vulkan_probe as vp;"
            "patch.object(vp, '_read_devices', return_value=[]).start(); vp._child_main();"
            "heavy = [m for m in ('ray', 'torch', 'pydantic') if m in sys.modules]; print(heavy, file=sys.stderr)"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert json.loads(out.stdout)["error"]["stage"] == "no_device"
        assert out.stderr.strip() == "[]"


class TestParent:
    @pytest.fixture(autouse=True)
    def _no_cache(self, monkeypatch):
        monkeypatch.setattr(vp, "_cached", None)

    def _completed(self, stdout="", returncode=0, stderr=""):
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)

    def test_reads_the_childs_last_line(self):
        answer = json.dumps({"devices": [_dev(0)], "selected": [0], "error": None})
        with patch.object(vp.subprocess, "run", return_value=self._completed(f"noise\n{answer}\n")) as run:
            result = vp.probe(python="/venv/bin/python")
        assert [d.index for d in result.devices] == [0]
        assert run.call_args[0][0] == ["/venv/bin/python", "-m", "modelship.utils.vulkan_probe"]

    def test_is_cached_unless_fresh(self):
        answer = json.dumps({"devices": [_dev(0)], "selected": [0], "error": None})
        with patch.object(vp.subprocess, "run", return_value=self._completed(answer)) as run:
            vp.probe()
            vp.probe()
            assert run.call_count == 1
            vp.probe(fresh=True)
            assert run.call_count == 2

    @pytest.mark.parametrize(
        ("outcome", "message"),
        [
            (subprocess.TimeoutExpired(cmd="probe", timeout=30), "did not finish in 30 s"),
            (OSError("no such file"), "could not be run"),
        ],
    )
    def test_a_child_that_cannot_answer_is_the_probe_stage(self, outcome, message):
        with patch.object(vp.subprocess, "run", side_effect=outcome):
            result = vp.probe()
        assert result.error is not None
        assert result.error.stage == "probe"
        assert message in result.error.message

    @pytest.mark.parametrize(
        "completed",
        [
            subprocess.CompletedProcess([], 1, "", "Segmentation fault"),
            subprocess.CompletedProcess([], 0, "not json", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ],
    )
    def test_a_crash_or_garbage_is_the_probe_stage(self, completed):
        with patch.object(vp.subprocess, "run", return_value=completed):
            result = vp.probe()
        assert result.error is not None
        assert result.error.stage == "probe"


_LIST_DEVICES_ROW = re.compile(r"^\s*(Vulkan\d+): (.+) \((\d+) MiB, (\d+) MiB free\)$", re.MULTILINE)


@pytest.mark.integration
@pytest.mark.llama_server
def test_the_probe_agrees_with_llama_cpp():
    from tests.conftest import MSHIP_BIN

    info = subprocess.run([MSHIP_BIN, "info"], capture_output=True, text=True, check=True).stdout
    if "variant: vulkan" not in info:
        pytest.skip("not a vulkan bootstrap")
    binary = re.search(r"^llama-server: (.+)$", info, re.MULTILINE)
    assert binary is not None
    env = {k: v for k, v in os.environ.items() if k != "GGML_VK_VISIBLE_DEVICES"}
    listed = subprocess.run(
        [binary[1].strip(), "serve", "--list-devices"], capture_output=True, text=True, env=env, check=True
    ).stdout
    rows = _LIST_DEVICES_ROW.findall(listed)

    result = vp.probe(fresh=True)
    assert result.error is None
    assert [name for _, name, _, _ in rows] == [d.name for d in result.devices]
    for (_, _, total_mib, free_mib), device in zip(rows, result.devices, strict=True):
        assert sum(h.size for h in vp._counted_heaps(device)) // 2**20 == int(total_mib)
        assert abs(vp.engine_free_bytes(device) // 2**20 - int(free_mib)) < 256


def test_a_device_dict_has_every_field_the_parent_reads():
    device = vp._device_from_json(_dev(3, "discrete", render=[226, 129], heaps=[]))
    assert device == vp.VulkanDevice(
        index=3,
        name="gpu3",
        type="discrete",
        vendor_id=_AMD,
        driver_id=_RADV,
        uuid="uuid-3",
        render=(226, 129),
        heaps=(),
    )
