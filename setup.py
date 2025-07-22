#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="my_quant_lib",
    version="0.1.0",
    description="Python bindings for MyQuantLib",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"my_quant_lib": ["*.so"]},
    include_package_data=True,
)
