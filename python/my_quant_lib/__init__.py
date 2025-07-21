import os
import platform
from pathlib import Path

def get_library_path():
    """컴파일된 커스텀 연산자 라이브러리의 경로를 반환합니다."""
    build_dir = Path(__file__).resolve().parent.parent.parent
    
    if platform.system() == "Windows":
        lib_name = "my_quant_ops.dll"
        lib_path = build_dir / "bin" / lib_name
        if not lib_path.exists():
             lib_path = build_dir / "lib" / lib_name
    elif platform.system() == "Darwin":
        lib_name = "libmy_quant_ops.dylib"
        lib_path = build_dir / "lib" / lib_name
    else:
        lib_name = "libmy_quant_ops.so"
        lib_path = build_dir / "lib" / lib_name

    if not lib_path.exists():
        raise FileNotFoundError(f"Could not find custom op library at {lib_path}")
    return str(lib_path)

from.my_quant_lib import register_observer, get_observer_state, ObserverState