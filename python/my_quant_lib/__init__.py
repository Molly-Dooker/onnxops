# python/my_quant_lib/__init__.py

import platform
from pathlib import Path

def get_library_path():
    """컴파일된 커스텀 연산자 라이브러리의 경로를 반환합니다."""
    pkg_dir = Path(__file__).resolve().parent
    if platform.system() == "Windows":
        lib_name = "my_quant_ops.dll"
    elif platform.system() == "Darwin":
        lib_name = "libmy_quant_ops.dylib"
    else:
        lib_name = "libmy_quant_ops.so"

    lib_path = pkg_dir / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(f"Could not find custom op library at {lib_path}")
    return str(lib_path)

from .my_quant_lib import register_observer, get_observer_state, ObserverState
