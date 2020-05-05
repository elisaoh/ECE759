#include<cuda.h>
#include<math.h>
#include<iostream>

__global__ void lag_matrix_kernel(const double* X1, const double* X2, double* lag_matrix, double* lag_x1, 
                                double* y_label, double bias, int rows, int cols, int p){
    // now the i-th elem: i = threadsIdx + blockIdx*blockDim
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // idx of row
    if(idx<rows){ 
    	// fill in the lag_matrix
        for(int i = 0; i<p; i++){
    	    lag_matrix[idx + i*rows] = X1[i+idx];
    	}
        for(int j = 0; j<p; j++){
            lag_matrix[idx+(j+p)*rows] = X2[j+idx];
        }
        lag_matrix[(cols-1)*rows + idx] = bias; // fill in the bias
        // fill in the lag_x1
        for(int i = 0; i<p; i++){
            lag_x1[idx + i*rows] = X1[i+idx];
        }
        lag_x1[p*rows + idx] = bias;
        // fill in the label
        y_label[idx] = X1[idx+p];
    }
    return;
}

// 
void gen_lag_matrix(const double* x1, const double* x2, double* lag_matrix, double* lag_x1, double* y_label, 
                    double bias, int n, int p, int rows, int cols, int threads_per_block){
    // we want to let each thread fill out each row
    int num_blocks = (rows+threads_per_block - 1)/threads_per_block;

    double * d_x1;
    double * d_x2;
    double * d_matrix_lag = new double[rows*cols];
    double * d_x1_lag = new double[rows*(p+1)];
    double * d_label = new double[rows];

    //std::cout << A[nsq-] << std::endl;
    //std::cout << B[nsq-] << std::endl; 

    cudaMallocManaged((void**)&d_x1, sizeof(double)*n);
    cudaMallocManaged((void**)&d_x2, sizeof(double)*n);
    cudaMallocManaged((void**)&d_matrix_lag, sizeof(double)*(rows*cols));
    cudaMallocManaged((void**)&d_x1_lag, sizeof(double)*(rows*(p+1)));
    cudaMallocManaged((void**)&d_label, sizeof(double)*rows);

    cudaMemcpy(d_x1, x1, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, x2, sizeof(double)*n, cudaMemcpyHostToDevice);

    
    // timing 
    
    cudaEvent_t start;
    cudaEvent_t stop;   
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    lag_matrix_kernel<<<num_blocks,threads_per_block>>>(d_x1, d_x2, d_matrix_lag, d_x1_lag, d_label, bias, rows, cols, p);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // bring the result back  -- dont need for task1
    cudaMemcpy(lag_matrix, d_matrix_lag, (rows*cols)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(lag_x1, d_x1_lag, (rows*(p+1))*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_label, d_label, rows*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_matrix_lag);
    cudaFree(d_x1_lag);
    cudaFree(d_label);

    std::cout << "Lagged matrix time: " << ms << " ms" << std::endl;

}
