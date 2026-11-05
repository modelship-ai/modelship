import os
import stat

import ray


def socket_root() -> str:
    return f"/tmp/mship-{os.getuid()}"


def is_private_dir(path: str) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700


def session_socket_dir() -> str:
    root = socket_root()
    os.makedirs(root, mode=0o700, exist_ok=True)
    if not is_private_dir(root):
        raise RuntimeError(f"{root} must be a directory owned by uid {os.getuid()} with mode 0700")
    path = os.path.join(root, ray.get_runtime_context().get_session_name())
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path
