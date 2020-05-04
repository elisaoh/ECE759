# How to run ormqr
example [ref](https://docs.nvidia.com/cuda/archive/10.1/pdf/CUSOLVER_Library.pdf) P224

run with 

`nvcc test.cpp ols.cpp -lcublas -lcusolver -o test `
` test`


function Ols

A and B is stored in col major, i.e. index = row + col*n_of_rows(leading dimension)