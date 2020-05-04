#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void Ols(cudaError_t cudaStat1, cusolverStatus_t cusolver_status, cublasStatus_t cublas_status,cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, const int m, const int lda, const int ldb,
	const int nrhs, double *d_A, double *d_B, double *d_tau, double *d_work, int *devInfo){

 int lwork = 0;
 int info_gpu = 0;
 const double one = 1;

	// step 3: query working space of geqrf and ormqr
 cusolver_status = cusolverDnDgeqrf_bufferSize(
 cusolverH,
 m,
 m,
 d_A,
 lda,
 &lwork);
 assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
 cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
 assert(cudaSuccess == cudaStat1);
// step 4: compute QR factorization
 cusolver_status = cusolverDnDgeqrf(
 cusolverH,
 m,
 m,
 d_A,
 lda,
 d_tau,
 d_work,
 lwork,
 devInfo);
 cudaStat1 = cudaDeviceSynchronize();
 assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
 assert(cudaSuccess == cudaStat1);
 // check if QR is good or not
 cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
 cudaMemcpyDeviceToHost);
 assert(cudaSuccess == cudaStat1);
 printf("after geqrf: info_gpu = %d\n", info_gpu);
 assert(0 == info_gpu);
// step 5: compute Q^T*B
 cusolver_status= cusolverDnDormqr(
 cusolverH,
 CUBLAS_SIDE_LEFT,
 CUBLAS_OP_T,
 m,
 nrhs,
 m,
 d_A,
 lda,
 d_tau,
 d_B,
 ldb,
 d_work,
 lwork,
 devInfo);
 cudaStat1 = cudaDeviceSynchronize();
 assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
 assert(cudaSuccess == cudaStat1);
 
 // check if QR is good or not
 cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
 cudaMemcpyDeviceToHost);
 assert(cudaSuccess == cudaStat1);
 printf("after ormqr: info_gpu = %d\n", info_gpu);
 assert(0 == info_gpu);
// step 6: compute x = R \ Q^T*B
 cublas_status = cublasDtrsm(
 cublasH,
 CUBLAS_SIDE_LEFT,
 CUBLAS_FILL_MODE_UPPER,
 CUBLAS_OP_N,
 CUBLAS_DIAG_NON_UNIT,
 m,
 nrhs,
 &one,
 d_A,
 lda,
 d_B,
 ldb);
 cudaStat1 = cudaDeviceSynchronize();
 assert(CUBLAS_STATUS_SUCCESS == cublas_status);
 assert(cudaSuccess == cudaStat1);

}