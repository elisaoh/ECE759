#ifndef LAGMAT_CUH
#define LAGMAT_CUH


__global__ void lag_matrix_kernel(const double* X1, const double* X2, double* lag_matrix,  double* lag_x1, 
								double* y_label, double bias, int rows, int cols, int p);

void gen_lag_matrix(const double* x1, const double* x2, double* lag_matrix,  double* lag_x1, double* y_label, 
					double bias, int n, int p, int rows, int cols, int threads_per_block);

#endif
