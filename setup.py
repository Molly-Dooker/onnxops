#!/usr/bin/env python3
import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

class CMakeBuild(_build_py):
    """build_py 단계에서 CMake configure/build/install을 실행합니다."""
    def run(self):
        # 1) 빌드 디렉토리 생성
        build_dir = os.path.abspath("build")
        os.makedirs(build_dir, exist_ok=True)

        # 2) CMake configure
        subprocess.check_call(
            ["cmake", os.path.abspath("." )],
            cwd=build_dir
        )

        # 3) CMake build
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "--", f"-j{os.cpu_count()}"],
            cwd=build_dir
        )

        # 4) CMake install into python/my_quant_lib
        subprocess.check_call(
            ["cmake", "--install", ".", "--prefix", os.path.abspath(".")],
            cwd=build_dir
        )

        # 5) 그 뒤에 setuptools 기본 build_py 실행 (패키지 복사 등)
        super().run()

setup(
    name="my_quant_lib",
    version="0.1.0",
    description="Python bindings for MyQuantLib",
    author="Your Name",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"my_quant_lib": ["*.so"]},
    include_package_data=True,
    install_requires=[
        "onnxruntime>=1.15",
        "numpy",
    ],
    zip_safe=False,
    cmdclass={"build_py": CMakeBuild},
)
