this is modified version of cuml. 

Granger added to arima related files. 

No extra files so as to avoid other problems.

--------------------------------------------

How to run the example (standalone ver.)

0. **I have already done this**

 In cuml/cpp/examples/arima, rename "CMakeList_standalone.txt" as "CMakeList.txt"
 
In "CMakeList.txt" replace **dbscan_example** with **arima_example**

1. activate conda enviroment(if built with conda)
   `conda activate cuml_dev`
2. cd into the *arima* directory
`cd cuml/cpp/examples/arima`
3. build in *build* dir
`mkdir build && cd build`
4. run cmake
`cmake .. -DCUML_LIBRARY_DIR=$CONDA_PREFIX/lib -DCUML_INCLUDE_DIR=$CONDA_PREFIX/include`
5. run make
`make`
6. there will be a binary file in build dir, run the example
`./arima_example`
