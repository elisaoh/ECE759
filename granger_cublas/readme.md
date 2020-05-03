# How to run ormqr
example [ref](https://docs.nvidia.com/cuda/archive/10.1/pdf/CUSOLVER_Library.pdf) P224
I don't know why
`nvcc -c -I/usr/local/cuda/include ormqr_example.cpp`

`nvcc -o a.out ormqr_example.o -lcudart -lcublas -lcusolver`

`./a.out`