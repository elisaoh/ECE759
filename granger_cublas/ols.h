#ifndef OLS_H
#define OLS_H


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void Ols(cudaError_t cudaStat1, cusolverStatus_t cusolver_status, cublasStatus_t cublas_status, cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, const int m, const int lda, const int ldb,
	const int nrhs, double *d_A, double *d_B, double *d_tau, double *d_work, int *devInfo);

#endif