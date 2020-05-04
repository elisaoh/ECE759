/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 * nvcc -c -I/usr/local/cuda/include ormqr_example.cpp
 * nvcc -o -fopenmp a.out ormqr_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 * 

 nvcc -c -I/usr/local/cuda/include ormqr_example.cpp
 *nvcc -o a.out ormqr_example.o -lcudart -lcublas -lcusolver
 ./a.out


 nvcc test.cpp ols.cpp -lcublas -lcusolver -o test
 */
#include "ols.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
 for(int row = 0 ; row < m ; row++){
 for(int col = 0 ; col < n ; col++){
 double Areg = A[row + col*lda];
 printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
 }
 }
}


int main(int argc, char*argv[])
{
 cusolverDnHandle_t cusolverH = NULL;
 cublasHandle_t cublasH = NULL;
 cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
 cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
 cudaError_t cudaStat1 = cudaSuccess;
 cudaError_t cudaStat2 = cudaSuccess;
 cudaError_t cudaStat3 = cudaSuccess;
 cudaError_t cudaStat4 = cudaSuccess;
 const int m = 3;
 const int p = 4;
 const int lda = p; 
 const int ldb = p;
 const int nrhs = 1; // number of right hand side vectors
/* | 1 2 3 |
 * A = | 4 5 6 |
 * | 2 1 1 |
 *
 * x = (1 1 1)'
 * b = (6 15 4)'
 */
 double A[lda*m] = { 1.0, 4.0, 2.0, 1.0, 2.0, 5.0, 1.0, 2.0, 3.0, 6.0, 1.0, 3.0};
// double X[ldb*nrhs] = { 1.0, 1.0, 1.0}; // exact solution
 double B[ldb*nrhs] = { 6.0, 15.0, 4.0, 6.0};
 double XC[m*nrhs]; // solution matrix from GPU
 double *d_A = NULL; // linear memory of GPU
 double *d_tau = NULL; // linear memory of GPU
 double *d_B = NULL;
 int *devInfo = NULL; // info in gpu (device copy)
 double *d_work = NULL;
 // int lwork = 0;
 // int info_gpu = 0;
 // const double one = 1;
 printf("A = (matlab base-1)\n");
 printMatrix(lda, m, A, lda, "A");
 printf("=====\n");
 printf("B = (matlab base-1)\n");
 printMatrix(ldb, nrhs, B, ldb, "B");
 printf("=====\n");
// step 1: create cusolver/cublas handle
 cusolver_status = cusolverDnCreate(&cusolverH);
 assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
 cublas_status = cublasCreate(&cublasH);
 assert(CUBLAS_STATUS_SUCCESS == cublas_status);




// step 2: copy A and B to device
 cudaStat1 = cudaMalloc ((void**)&d_A , sizeof(double) * lda * m);
 cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(double) * m);
 cudaStat3 = cudaMalloc ((void**)&d_B , sizeof(double) * ldb * nrhs);
 cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
 assert(cudaSuccess == cudaStat1);
 assert(cudaSuccess == cudaStat2);
 assert(cudaSuccess == cudaStat3);
 assert(cudaSuccess == cudaStat4);
 cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m ,
 cudaMemcpyHostToDevice);
 cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs,
 cudaMemcpyHostToDevice);
 assert(cudaSuccess == cudaStat1);
 assert(cudaSuccess == cudaStat2);


Ols(cudaStat1, cusolver_status, cublas_status, cusolverH, cublasH, 
	m, lda, ldb, nrhs, d_A, d_B, d_tau, d_work, 
	devInfo);


 cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*m*nrhs,
 cudaMemcpyDeviceToHost);
 assert(cudaSuccess == cudaStat1);
 printf("X = (matlab base-1)\n");
 printMatrix(m, nrhs, XC, ldb, "X");
// free resources
 if (d_A ) cudaFree(d_A);
 if (d_tau ) cudaFree(d_tau);
 if (d_B ) cudaFree(d_B);
 if (devInfo) cudaFree(devInfo);
 if (d_work ) cudaFree(d_work);
 if (cublasH ) cublasDestroy(cublasH);
 if (cusolverH) cusolverDnDestroy(cusolverH);
 cudaDeviceReset();
 return 0;
}

