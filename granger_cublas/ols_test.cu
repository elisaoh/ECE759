#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>


#define I(i, j, ld) j * ld + i

#define CUDA_CALL(x) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}} 


 #define CUBLAS_CALL(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)


int main(){

cublasHandle_t handle;
int n = 4;
int p = 3;
int matrixSize = n*p;
float X[n * p] = { 1.0, 1.0, 3.0, -2.0,
					1.0, 2.0, -1.0, 3.0,
					2.0, 1.0, 3.0, -1.0};

float Y[n] = {1.0, 2.0, -2.0, 1.0};

  float *a, *b;
  a = (float*) malloc(sizeof(*X));
  b = (float*) malloc(sizeof(*X));
  *a = 1.0;
  *b = 0.0;

float *XtX, *XtY, *beta, *dX, *dXtX, *dXtY, *dbeta, *dY, *dXtXi;


cublasCreate(&handle);

  XtX = (float*) malloc(p * p * sizeof(*X));
  XtY = (float*) malloc(p * sizeof(*X));
  beta = (float*) malloc(p * sizeof(*X));

  CUDA_CALL(cudaMalloc((void**) &dX, n * p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dXtX, p * p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dXtXi, p * p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dXtY, p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dbeta, p * sizeof(*X)));
  CUDA_CALL(cudaMalloc((void**) &dY, n * sizeof(*X)));

  CUDA_CALL(cudaMemcpy(dX, X, n * p * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dY, Y, n * sizeof(float), cudaMemcpyHostToDevice));

  CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, p, n, 
    a, dX, n, dX, n, b, dXtX, p));


 CUDA_CALL(cudaMemcpy(XtX, dXtX, p * p * sizeof(float), cudaMemcpyDeviceToHost));  


 //LU decomposition

int *d_pivot_array;
int *d_info_array;

int batchSize = 1;


//allocate pivoting vector and the info array

CUDA_CALL(cudaMalloc((void **)&d_pivot_array, p * sizeof(int)));
CUDA_CALL(cudaMalloc((void **)&d_info_array, sizeof(int)));



float **devPtrA = 0;
float **devPtrA_dev = NULL;


devPtrA =(float **)malloc(1 * sizeof(*devPtrA));
	if (devPtrA == NULL)
	{ perror("malloc"); exit(EXIT_FAILURE); }

CUDA_CALL(cudaMalloc((void **) devPtrA, matrixSize * sizeof(*X)));
CUDA_CALL(cudaMalloc((void **) &devPtrA_dev, 1 * sizeof(*devPtrA)));

CUDA_CALL(cudaMemcpy(devPtrA_dev, devPtrA, 1 * sizeof(*devPtrA), cudaMemcpyHostToDevice));
CUBLAS_CALL(cublasSetMatrix(n, p, sizeof(a[0]), dXtX, n, devPtrA[0], n));


// // this works
// cublascall(cublasDgetrfBatched(handle, m, devPtrA_dev,m,d_pivot_array,d_info_array,1));

// cublascall(cublasGetMatrix(m, n, sizeof(double), devPtrA[0], m, a, m));



  CUBLAS_CALL(cublasSgetrfBatched(handle, p, devPtrA_dev, p, d_pivot_array, d_info_array, batchSize));


//reversion

 CUBLAS_CALL(cublasSgetriBatched(handle, p, dXtX, p, d_pivot_array, dXtXi, p, d_info_array, batchSize));
  
  CUDA_CALL(cudaMemcpy(XtX, dXtXi, p * p * sizeof(float), cudaMemcpyDeviceToHost));

  
  
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, 
    a, dX, n, dY, n, b, dXtY, p);

  cublasSgemv(handle, CUBLAS_OP_N, p, p, 
    a, dXtXi, p, dXtY, 1, b, dbeta, 1);

  CUDA_CALL(cudaMemcpy(beta, dbeta, p * sizeof(float), cudaMemcpyDeviceToHost));

  printf("CUBLAS matrix algebra parameter estimates:\n");
  for(i = 0; i < p; i++){
    printf("beta_%i = %0.2f\n", i, beta[i]);
  }
  printf("\n");

  cublasDestroy(handle);

  free(X);
  free(XtX);
  free(XtY);
  free(beta);
  free(Y);
  
  CUDA_CALL(cudaFree(dX));
  CUDA_CALL(cudaFree(dXtX));
  CUDA_CALL(cudaFree(dXtXi));
  CUDA_CALL(cudaFree(dXtY));
  CUDA_CALL(cudaFree(dbeta));
  CUDA_CALL(cudaFree(dY));
}

