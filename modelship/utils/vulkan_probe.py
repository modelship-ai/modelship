from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

_VK_API_VERSION_1_2 = (1 << 22) | (2 << 12)

_VK_SUCCESS = 0
_VK_INCOMPLETE = 5

_VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1

_VENDOR_AMD = 0x1002


class _ApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _InstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_ApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class _ExtensionProperties(ctypes.Structure):
    _fields_ = [("extensionName", ctypes.c_char * 256), ("specVersion", ctypes.c_uint32)]


class _PhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_int),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("limits", ctypes.c_uint64 * 63),
        ("sparseProperties", ctypes.c_uint32 * 5),
    ]


class _PhysicalDeviceProperties2(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("properties", _PhysicalDeviceProperties)]


class _PhysicalDeviceIDProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("deviceUUID", ctypes.c_uint8 * 16),
        ("driverUUID", ctypes.c_uint8 * 16),
        ("deviceLUID", ctypes.c_uint8 * 8),
        ("deviceNodeMask", ctypes.c_uint32),
        ("deviceLUIDValid", ctypes.c_uint32),
    ]


class _PhysicalDeviceDriverProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("driverID", ctypes.c_int),
        ("driverName", ctypes.c_char * 256),
        ("driverInfo", ctypes.c_char * 256),
        ("conformanceVersion", ctypes.c_uint8 * 4),
    ]


class _PhysicalDeviceDrmProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("hasPrimary", ctypes.c_uint32),
        ("hasRender", ctypes.c_uint32),
        ("primaryMajor", ctypes.c_int64),
        ("primaryMinor", ctypes.c_int64),
        ("renderMajor", ctypes.c_int64),
        ("renderMinor", ctypes.c_int64),
    ]


class _PhysicalDeviceFeatures2(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("features", ctypes.c_uint32 * 55)]


class _PhysicalDevice16BitStorageFeatures(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("storageBuffer16BitAccess", ctypes.c_uint32),
        ("uniformAndStorageBuffer16BitAccess", ctypes.c_uint32),
        ("storagePushConstant16", ctypes.c_uint32),
        ("storageInputOutput16", ctypes.c_uint32),
    ]


class _MemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class _MemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _PhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", _MemoryType * 32),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", _MemoryHeap * 16),
    ]


class _PhysicalDeviceMemoryProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("memoryProperties", _PhysicalDeviceMemoryProperties),
    ]


class _PhysicalDeviceMemoryBudgetProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("heapBudget", ctypes.c_uint64 * 16),
        ("heapUsage", ctypes.c_uint64 * 16),
    ]


class ProbeStageError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def _format_uuid(raw: bytes) -> str:
    h = raw.hex()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _read_devices() -> list[dict[str, Any]]:
    try:
        lib = ctypes.CDLL("libvulkan.so.1")
    except OSError as e:
        raise ProbeStageError("loader", f"the Vulkan loader libvulkan.so.1 did not load: {e}") from None

    get_proc = lib.vkGetInstanceProcAddr
    get_proc.restype = ctypes.c_void_p
    get_proc.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    def function(instance: ctypes.c_void_p | None, name: str, restype: Any, *argtypes: Any) -> Any:
        address = get_proc(instance, name.encode())
        if not address:
            raise ProbeStageError("instance", f"the Vulkan loader has no {name}")
        return ctypes.CFUNCTYPE(restype, *argtypes)(address)

    create_instance = function(
        None, "vkCreateInstance", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    )
    app = _ApplicationInfo(sType=0, pApplicationName=b"modelship", apiVersion=_VK_API_VERSION_1_2)
    info = _InstanceCreateInfo(sType=1, pApplicationInfo=ctypes.pointer(app))
    instance = ctypes.c_void_p()
    result = create_instance(ctypes.byref(info), None, ctypes.byref(instance))
    if result != _VK_SUCCESS:
        raise ProbeStageError("instance", f"vkCreateInstance failed with VkResult {result}")
    try:
        return _read_instance_devices(instance, function)
    finally:
        function(instance, "vkDestroyInstance", None, ctypes.c_void_p, ctypes.c_void_p)(instance, None)


def _read_instance_devices(instance: ctypes.c_void_p, function: Any) -> list[dict[str, Any]]:
    void_p, u32_p = ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
    enumerate_devices = function(instance, "vkEnumeratePhysicalDevices", ctypes.c_int, void_p, u32_p, void_p)
    enumerate_extensions = function(
        instance, "vkEnumerateDeviceExtensionProperties", ctypes.c_int, void_p, ctypes.c_char_p, u32_p, void_p
    )
    get_properties = function(instance, "vkGetPhysicalDeviceProperties", None, void_p, void_p)
    get_properties2 = function(instance, "vkGetPhysicalDeviceProperties2", None, void_p, void_p)
    get_features2 = function(instance, "vkGetPhysicalDeviceFeatures2", None, void_p, void_p)
    get_memory = function(instance, "vkGetPhysicalDeviceMemoryProperties", None, void_p, void_p)
    get_memory2 = function(instance, "vkGetPhysicalDeviceMemoryProperties2", None, void_p, void_p)

    count = ctypes.c_uint32()
    result = enumerate_devices(instance, ctypes.byref(count), None)
    if result != _VK_SUCCESS:
        raise ProbeStageError("instance", f"vkEnumeratePhysicalDevices failed with VkResult {result}")
    handles = (ctypes.c_void_p * count.value)()
    result = enumerate_devices(instance, ctypes.byref(count), handles)
    if result not in (_VK_SUCCESS, _VK_INCOMPLETE):
        raise ProbeStageError("instance", f"vkEnumeratePhysicalDevices failed with VkResult {result}")

    devices = []
    for index in range(count.value):
        handle = handles[index]
        props = _PhysicalDeviceProperties()
        get_properties(handle, ctypes.byref(props))
        extensions = _device_extensions(handle, enumerate_extensions)
        device: dict[str, Any] = {
            "index": index,
            "name": props.deviceName.decode(errors="replace"),
            "type": {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}.get(props.deviceType, "other"),
            "vendor_id": props.vendorID,
            "driver_id": None,
            "uuid": None,
            "luid": None,
            "fp16_storage": False,
            "render": None,
            "heaps": [],
        }
        if props.apiVersion >= ((1 << 22) | (1 << 12)):
            _read_identity(device, handle, props.apiVersion, extensions, get_properties2)
            _read_features(device, handle, get_features2)
            device["heaps"] = _read_heaps2(handle, "VK_EXT_memory_budget" in extensions, get_memory2)
        else:
            device["heaps"] = _read_heaps(handle, get_memory)
        devices.append(device)
    return devices


def _device_extensions(handle: Any, enumerate_extensions: Any) -> set[str]:
    count = ctypes.c_uint32()
    if enumerate_extensions(handle, None, ctypes.byref(count), None) != _VK_SUCCESS:
        return set()
    entries = (_ExtensionProperties * count.value)()
    if enumerate_extensions(handle, None, ctypes.byref(count), entries) not in (_VK_SUCCESS, _VK_INCOMPLETE):
        return set()
    return {entries[i].extensionName.decode(errors="replace") for i in range(count.value)}


def _read_identity(device: dict[str, Any], handle: Any, api: int, extensions: set[str], get_properties2: Any) -> None:
    ident = _PhysicalDeviceIDProperties(sType=1000071004)
    driver = _PhysicalDeviceDriverProperties(sType=1000196000)
    drm = _PhysicalDeviceDrmProperties(sType=1000353000)
    chain: list[ctypes.Structure] = [ident]
    has_driver = api >= _VK_API_VERSION_1_2 or "VK_KHR_driver_properties" in extensions
    if has_driver:
        chain.append(driver)
    has_drm = "VK_EXT_physical_device_drm" in extensions
    if has_drm:
        chain.append(drm)
    props2 = _PhysicalDeviceProperties2(sType=1000059001)
    _link(props2, chain)
    get_properties2(handle, ctypes.byref(props2))

    device["uuid"] = _format_uuid(bytes(ident.deviceUUID))
    if ident.deviceLUIDValid:
        device["luid"] = bytes(ident.deviceLUID).hex()
    if has_driver:
        device["driver_id"] = driver.driverID
    if has_drm and drm.hasRender:
        device["render"] = [drm.renderMajor, drm.renderMinor]


def _read_features(device: dict[str, Any], handle: Any, get_features2: Any) -> None:
    storage16 = _PhysicalDevice16BitStorageFeatures(sType=1000083000)
    features2 = _PhysicalDeviceFeatures2(sType=1000059000)
    _link(features2, [storage16])
    get_features2(handle, ctypes.byref(features2))
    device["fp16_storage"] = bool(storage16.storageBuffer16BitAccess)


def _read_heaps2(handle: Any, has_budget: bool, get_memory2: Any) -> list[dict[str, Any]]:
    budget = _PhysicalDeviceMemoryBudgetProperties(sType=1000237000)
    memory2 = _PhysicalDeviceMemoryProperties2(sType=1000059006)
    if has_budget:
        _link(memory2, [budget])
    get_memory2(handle, ctypes.byref(memory2))
    memory = memory2.memoryProperties
    return [
        {
            "size": memory.memoryHeaps[i].size,
            "device_local": bool(memory.memoryHeaps[i].flags & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT),
            "budget": budget.heapBudget[i] if has_budget else None,
            "usage": budget.heapUsage[i] if has_budget else None,
        }
        for i in range(min(memory.memoryHeapCount, 16))
    ]


def _read_heaps(handle: Any, get_memory: Any) -> list[dict[str, Any]]:
    memory = _PhysicalDeviceMemoryProperties()
    get_memory(handle, ctypes.byref(memory))
    return [
        {
            "size": memory.memoryHeaps[i].size,
            "device_local": bool(memory.memoryHeaps[i].flags & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT),
            "budget": None,
            "usage": None,
        }
        for i in range(min(memory.memoryHeapCount, 16))
    ]


def _link(head: ctypes.Structure, chain: list[ctypes.Structure]) -> None:
    previous = head
    for struct in chain:
        previous.pNext = ctypes.cast(ctypes.pointer(struct), ctypes.c_void_p)  # type: ignore[attr-defined]
        previous = struct


def _priority(vendor_id: int, driver_id: int | None) -> int:
    table = {
        _VENDOR_AMD: {3: 1, 2: 2, 1: 3},
        0x8086: {6: 1, 5: 2},
        0x10DE: {4: 1, 24: 2},
        0x5143: {8: 1, 18: 2},
    }.get(vendor_id, {})
    return {**table, 23: 100}.get(driver_id, sys.maxsize) if driver_id is not None else sys.maxsize


def _same_gpu(a: dict[str, Any], b: dict[str, Any]) -> bool:
    same = (a["uuid"] is not None and a["uuid"] == b["uuid"]) or (a["luid"] is not None and a["luid"] == b["luid"])
    return same and not (a["driver_id"] == 14 and b["driver_id"] == 14)


def select_devices(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for device in devices:
        if device["type"] not in ("discrete", "integrated") or not device["fp16_storage"]:
            continue
        kept = next((c for c in chosen if _same_gpu(c, device)), None)
        if kept is None:
            chosen.append(device)
        elif _priority(kept["vendor_id"], device["driver_id"]) < _priority(kept["vendor_id"], kept["driver_id"]):
            chosen.remove(kept)
            chosen.append(device)
    if not chosen:
        chosen = [d for d in devices if d["type"] != "cpu"][:1]
    if any(d["type"] == "discrete" for d in chosen):
        chosen = [d for d in chosen if d["type"] == "discrete"]
    return chosen


def _describe(devices: list[dict[str, Any]]) -> str:
    return ", ".join(f"{d['name']} ({d['type']})" for d in devices) or "nothing"


def _child_main() -> None:
    try:
        devices = _read_devices()
        selected = select_devices(devices)
        if not selected:
            raise ProbeStageError(
                "no_device",
                f"Vulkan lists no GPU llama.cpp can use; it found {_describe(devices)}",
            )
        out: dict[str, Any] = {"devices": devices, "selected": [d["index"] for d in selected], "error": None}
    except ProbeStageError as e:
        out = {"devices": [], "selected": [], "error": {"stage": e.stage, "message": e.message}}
    print(json.dumps(out))


@dataclass(frozen=True)
class Heap:
    size: int
    device_local: bool
    budget: int | None
    usage: int | None


@dataclass(frozen=True)
class VulkanDevice:
    index: int
    name: str
    type: str
    vendor_id: int
    driver_id: int | None
    uuid: str | None
    render: tuple[int, int] | None
    heaps: tuple[Heap, ...] = ()


@dataclass(frozen=True)
class ProbeError:
    stage: str
    message: str


@dataclass(frozen=True)
class VulkanProbe:
    devices: list[VulkanDevice] = field(default_factory=list)
    all_devices: list[VulkanDevice] = field(default_factory=list)
    error: ProbeError | None = None


def parse(raw: str) -> VulkanProbe:
    data = json.loads(raw)
    if data.get("error"):
        return VulkanProbe(error=ProbeError(stage=data["error"]["stage"], message=data["error"]["message"]))
    every = [_device_from_json(d) for d in data["devices"]]
    by_index = {d.index: d for d in every}
    return VulkanProbe(devices=[by_index[i] for i in data["selected"]], all_devices=every)


def _device_from_json(d: dict[str, Any]) -> VulkanDevice:
    return VulkanDevice(
        index=int(d["index"]),
        name=str(d["name"]),
        type=str(d["type"]),
        vendor_id=int(d["vendor_id"]),
        driver_id=d["driver_id"],
        uuid=d["uuid"],
        render=tuple(d["render"]) if d["render"] else None,  # type: ignore[arg-type]
        heaps=tuple(Heap(int(h["size"]), bool(h["device_local"]), h["budget"], h["usage"]) for h in d["heaps"]),
    )


_cached: VulkanProbe | None = None


def probe(*, fresh: bool = False, python: str | None = None) -> VulkanProbe:
    global _cached
    if fresh or _cached is None:
        _cached = _run_child(python or sys.executable)
    return _cached


def _run_child(python: str) -> VulkanProbe:
    try:
        result = subprocess.run(
            [python, "-m", "modelship.utils.vulkan_probe"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return VulkanProbe(error=ProbeError("probe", f"the Vulkan probe did not finish in {e.timeout:.0f} s"))
    except OSError as e:
        return VulkanProbe(error=ProbeError("probe", f"the Vulkan probe could not be run: {e}"))
    lines = result.stdout.strip().splitlines()
    try:
        if result.returncode != 0 or not lines:
            raise ValueError
        return parse(lines[-1])
    except (ValueError, KeyError, TypeError):
        detail = (result.stderr.strip().splitlines() or ["no output"])[-1]
        return VulkanProbe(
            error=ProbeError("probe", f"the Vulkan probe exited {result.returncode} without an answer: {detail}")
        )


def visible(result: VulkanProbe, mask: str | None) -> list[VulkanDevice]:
    if mask is None:
        return list(result.devices)
    by_index = {d.index: d for d in result.all_devices}
    return [by_index[int(i)] for i in mask.split(",") if i]


def ggml_visible_devices(gpu_slots: list[str]) -> str:
    slots = sorted(int(s) for s in gpu_slots)
    devices = probe().devices if slots else []
    return ",".join(str(devices[slot].index) for slot in slots)


@dataclass(frozen=True)
class DeviceMemory:
    total_bytes: int
    free_bytes: int
    engine_free_bytes: int


def _heap_free(heap: Heap) -> int:
    if heap.budget is None or heap.usage is None:
        return heap.size
    return max(0, min(heap.budget, heap.size) - heap.usage)


def _counted_heaps(device: VulkanDevice) -> list[Heap]:
    if device.type == "integrated":
        return list(device.heaps)
    return [h for h in device.heaps if h.device_local]


def engine_free_bytes(device: VulkanDevice) -> int:
    return sum(_heap_free(h) for h in _counted_heaps(device))


class MemoryUnreadableError(Exception):
    pass


def device_memory(device: VulkanDevice, available_ram_bytes: int, sys_root: str = "/sys") -> DeviceMemory:
    engine_free = engine_free_bytes(device)
    heaps = _counted_heaps(device)
    if device.type != "integrated":
        return DeviceMemory(sum(h.size for h in heaps), engine_free, engine_free)
    if device.vendor_id == _VENDOR_AMD:
        vram_total, vram_used, gtt_total, gtt_used = _amdgpu_mem_info(device, sys_root)
        gtt_free = max(0, gtt_total - gtt_used)
        if available_ram_bytes:
            gtt_free = min(gtt_free, available_ram_bytes)
        return DeviceMemory(vram_total + gtt_total, max(0, vram_total - vram_used) + gtt_free, engine_free)
    free = min(engine_free, available_ram_bytes) if available_ram_bytes else engine_free
    return DeviceMemory(sum(h.size for h in heaps), free, engine_free)


def _amdgpu_mem_info(device: VulkanDevice, sys_root: str) -> tuple[int, int, int, int]:
    if device.render is None:
        raise MemoryUnreadableError(f"{device.name} reports no DRM render node (VK_EXT_physical_device_drm)")
    major, minor = device.render
    base = os.path.join(sys_root, "dev", "char", f"{major}:{minor}", "device")
    values = []
    for name in ("mem_info_vram_total", "mem_info_vram_used", "mem_info_gtt_total", "mem_info_gtt_used"):
        path = os.path.join(base, name)
        try:
            with open(path) as f:
                values.append(int(f.read().strip()))
        except (OSError, ValueError) as e:
            raise MemoryUnreadableError(f"can't read {path}: {e}") from None
    return values[0], values[1], values[2], values[3]


if __name__ == "__main__":
    _child_main()
