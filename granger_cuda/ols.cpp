#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


double Ols(const int m, const int lda, const int ldb,
	const int nrhs, double *A, double *B, double *Bhat){

 cusolverDnHandle_t cusolverH = NULL;
 cublasHandle_t cublasH = NULL;
 cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
 cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
 cudaError_t cudaStat1 = cudaSuccess;
 cudaError_t cudaStat2 = cudaSuccess;
 cudaError_t cudaStat3 = cudaSuccess;
 cudaError_t cudaStat4 = cudaSuccess;


// step 1: create cusolver/cublas handle
 cusolver_status = cusolverDnCreate(&cusolverH);
 assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
 cublas_status = cublasCreate(&cublasH);
 assert(CUBLAS_STATUS_SUCCESS == cublas_status);


 double *d_A = NULL; // linear memory of GPU
 double *d_tau = NULL; // linear memory of GPU
 double *d_B = NULL;
 int *devInfo = NULL; // info in gpu (device copy)
 double *d_work = NULL;
 
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
 cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs ,
 cudaMemcpyHostToDevice);
 assert(cudaSuccess == cudaStat1);
 assert(cudaSuccess == cudaStat2);



 double *dd_A = NULL;
 cudaStat1 = cudaMalloc ((void**)&dd_A , sizeof(double) * lda * m); 
 assert(cudaSuccess == cudaStat1);
 cudaStat1 = cudaMemcpy(dd_A, A, sizeof(double) * lda * m ,
 cudaMemcpyHostToDevice);

  double *dd_B = NULL;
 cudaStat1 = cudaMalloc ((void**)&dd_B , sizeof(double)  * ldb * nrhs);
 assert(cudaSuccess == cudaStat1);
 cudaStat1 = cudaMemcpy(dd_B, B, sizeof(double) * ldb * nrhs ,
 cudaMemcpyHostToDevice);


 int lwork = 0;
 int info_gpu = 0;
 const double one = 1;
 const double minus = -1;



cudaEvent_t start;
cudaEvent_t stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

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
double ssr = 0.0;

//computer ssr
 cublas_status = cublasDgemm(
 	cublasH, CUBLAS_OP_N, CUBLAS_OP_N, lda, nrhs, m, &one, dd_A, lda, d_B, m, &minus, dd_B, ldb);
 cublas_status = cublasDnrm2(cublasH, ldb,
                            dd_B, 1, &ssr);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);


 printf("OLS time: %f ms \n", ms);



 cudaStat1 = cudaDeviceSynchronize();
 assert(CUBLAS_STATUS_SUCCESS == cublas_status);
 assert(cudaSuccess == cudaStat1);

 cudaStat1 = cudaMemcpy(Bhat, dd_B, sizeof(double)*ldb,
 cudaMemcpyDeviceToHost);
 assert(cudaSuccess == cudaStat1);

 if (d_A ) cudaFree(d_A);
 if (dd_A ) cudaFree(d_A);
 if (d_tau ) cudaFree(d_tau);
 if (d_B ) cudaFree(d_B);
 if (dd_B ) cudaFree(d_B);
 if (devInfo) cudaFree(devInfo);
 if (d_work ) cudaFree(d_work);
 if (cublasH ) cublasDestroy(cublasH);
 if (cusolverH) cusolverDnDestroy(cusolverH);
 cudaDeviceReset();

 return pow(ssr,2.0);

}