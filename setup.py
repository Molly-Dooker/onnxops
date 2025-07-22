# setup.py

import setuptools
import shutil
import glob
import os
from pathlib import Path
from setuptools.command.build_py import build_py as _build_py

class build_py(_build_py):
    """
    빌드 전에 CMake로 생성된 .so 파일을
    python/my_quant_lib/ 폴더로 복사합니다.
    """
    def run(self):
        here = Path(__file__).parent.resolve()

        # CMake가 만든 .so 가 있는 디렉터리
        build_so_dir = here / "build" / "python" / "my_quant_lib"
        # 패키지 소스 디렉터리
        pkg_dir = here / "python" / "my_quant_lib"

        if build_so_dir.exists():
            for so in glob.glob(str(build_so_dir / "*.so")):
                shutil.copy2(so, pkg_dir)
                print(f"Copied {so} → {pkg_dir}")
        else:
            print(f"[Warning] {build_so_dir} not found, skipping .so copy")

        # 원래 build_py 로직 수행
        super().run()

setuptools.setup(
    name="my_quant_lib",
    version="0.1.0",
    description="ONNX custom-op 기반 moving-average observer 파이썬 바인딩",
    author="Your Name",
    packages=["my_quant_lib"],
    package_dir={"my_quant_lib": "python/my_quant_lib"},
    package_data={"my_quant_lib": ["*.so"]},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
    cmdclass={
        "build_py": build_py,      # 여기에 커스텀 빌드 커맨드 등록
    },
)
