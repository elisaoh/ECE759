# test bash script for granger(arima) example

rm -rf build

mkdir build && cd build

cmake .. -DCUML_LIBRARY_DIR=$CONDA_PREFIX/lib -DCUML_INCLUDE_DIR=$CONDA_PREFIX/include

make

./granger_example


