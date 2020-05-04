#ifndef LAGMAT_CUH
#define LAGMAT_CUH


__global__ void lag_matrix_kernel(const double* X1, const double* X2, double* lag_matrix, double* y_label, double bias;
                                    int rows, int cols, int p);

void lag_matrix(const double* x1, const double* x2, double* lag_matrix, double* y_label, double bias;
                int n, int p, int rows, int cols, unsigned int threads_per_block);

#endif