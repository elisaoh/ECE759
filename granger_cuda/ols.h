#ifndef OLS_H
#define OLS_H


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

double Ols(const int m, const int lda, const int ldb,
	const int nrhs, double *A, double *B, double *Bhat);

#endif