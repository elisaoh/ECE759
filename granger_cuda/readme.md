# How to run test

1. generate data of n_len
`g++ generate_data.cpp -o generate_data`

`./generate_data n_len`

2. granger causality test

`nvcc test_v2.cpp ols.cpp gen_lag_matrix.cu -lcublas -lcusolver -o test_v2 `


` ./test_v2 n_len`



