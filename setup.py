# setup.py

import setuptools
from pathlib import Path

here = Path(__file__).parent.resolve()

setuptools.setup(
    name="my_quant_lib",
    version="0.1.0",
    description="ONNX custom-op 기반 moving-average observer 파이썬 바인딩",
    author="Your Name",
    packages=["my_quant_lib"],
    package_dir={"my_quant_lib": "python/my_quant_lib"},
    package_data={"my_quant_lib": ["*.so"]},
    include_package_data=True,
    zip_safe=False,     # C 확장 모듈 포함 시 False
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7",
)
