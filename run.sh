rm -rf build
rm -rf dist
rm -rf my_quant_lib.egg-info
mkdir build
cd build
cmake ..
make -j32
cd ..
python3 setup.py bdist_wheel
pip uninstall my_quant_lib
pip install dist/my_quant_lib-0.1.0-py3-none-any.whl