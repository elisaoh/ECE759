#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
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



void cublas_lu(int m, int n, double* a)
{
    cublasHandle_t handle;
    double **devPtrA = 0;
    double **devPtrA_dev = NULL;
    int *d_pivot_array;
    int *d_info_array;
    int rowsA = m;
    int colsA = n;
    int matrixSizeA;
    cudaError_t error;

	// allocate the pivoting vector and the info array
	cudacall(cudaMalloc((void **)&d_pivot_array, n * sizeof(int)));
    cudacall(cudaMalloc((void **)&d_info_array, sizeof(int)));

    cublascall(cublasCreate(&handle));
    matrixSizeA = rowsA * colsA;

    devPtrA =(double **)malloc(1 * sizeof(*devPtrA));
 	if (devPtrA == NULL)
 	{ perror("malloc"); exit(EXIT_FAILURE); }
	
    cudacall(cudaMalloc((void **) devPtrA, matrixSizeA * sizeof(double)));
    cudacall(cudaMalloc((void **) &devPtrA_dev, 1 * sizeof(*devPtrA)));

    cudacall(cudaMemcpy(devPtrA_dev, devPtrA, 1 * sizeof(*devPtrA), cudaMemcpyHostToDevice));
    
    cublascall(cublasSetMatrix(rowsA, colsA, sizeof(a[0]), a, rowsA, devPtrA[0], rowsA));
    // this works
    cublascall(cublasDgetrfBatched(handle, m, devPtrA_dev,m,d_pivot_array,d_info_array,1));

    cublascall(cublasGetMatrix(m, n, sizeof(double), devPtrA[0], m, a, m));
}


int main()
{
	const int n = 4;
	double A[n * n] = { 1.0, 1.0, 3.0, -2.0,
						1.0, 2.0, -1.0, 3.0,
						2.0, 1.0, 3.0, -1.0,
						1.0, 2.0, -2.0, 1.0 };
						
	cublas_lu(n, n, A);
		
	fprintf(stdout, "Output:\n\n");
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
            fprintf(stdout,"%f\t",A[i*n+j]);
        fprintf(stdout,"\n");
    }					
}