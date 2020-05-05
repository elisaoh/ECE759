# How to run ormqr
example [ref](https://docs.nvidia.com/cuda/archive/10.1/pdf/CUSOLVER_Library.pdf) P224

run with 


1. generate data of n_len
`g++ generate_data.cpp -o generate_data`

`./generate_data n_len`

2. granger causality test

`nvcc test_v2.cpp ols.cpp gen_lag_matrix.cu -lcublas -lcusolver -o test_v2 `


` ./test_v2 n_len`


function Ols

A and B is stored in col major, i.e. index = row + col*n_of_rows(leading dimension)
