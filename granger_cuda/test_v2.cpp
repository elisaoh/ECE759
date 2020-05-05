/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)

nvcc test_v2.cpp ols.cpp gen_lag_matrix.cu -lcublas -lcusolver -o test_v2
 */


// The std::chrono namespace provides timer functions in C++
#include <chrono>

// std::ratio provides easy conversions between metric units
#include <ratio>
#include "ols.h"
#include "Chi2PLookup.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cusolverDn.h>
#include <fstream>
#include "gen_lag_matrix.cuh"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

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

const int x_len = atoi(argv[1]); // length of the inputs
 double x1[x_len];
 std::ifstream x1_file("x1.txt");
  if(x1_file.is_open())
    { for(int i = 0; i < x_len; ++i)
        {
            x1_file >> x1[i];
        }
    }

double x2[x_len];
 std::ifstream x2_file("x2.txt");
  if(x2_file.is_open())
    { for(int i = 0; i < x_len; ++i)
        {
            x2_file >> x2[i];
        }
    }


 // double x1[x_len] = { 1.0, 2.0, 3.0, 4.0, 7.0, 6.0, 7.0, 8.0};
 // double x2[x_len] = { 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};

 double bias = 1.0; // bias = 1 or 0
 int p_lag = 1; // lag length
 int rows = x_len - p_lag; // rows of the lag matrix
 int cols = p_lag*2 + bias;
 
 double lag_x1[rows * p_lag+1];
 double y_label[rows];
  double lag_matrix[rows * cols];
 double y_joint[rows];
double y_own[rows];

 int threads_per_block = 512;


high_resolution_clock::time_point start;
high_resolution_clock::time_point end;
duration<double, std::milli> duration_sec;


 gen_lag_matrix(x1, x2, lag_matrix, lag_x1, y_label, bias, x_len, p_lag, rows, cols, threads_per_block);


 const int m = cols;
 const int p = rows;
 const int lda = p; 
 const int ldb = p;
 const int nrhs = 1; // number of right hand side vectors

 double XC[m*nrhs]; // solution matrix from GPU

 // printf("lag_matrix = (matlab base-1)\n");
 // printMatrix(lda, m, lag_matrix, lda, "lag_matrix");
 // printf("=====\n");

 // printf("lag_x1 = (matlab base-1)\n");
 // printMatrix(lda, p_lag+1, lag_x1, lda, "lag_matrix");
 // printf("=====\n");

 // printf("y_label = (matlab base-1)\n");
 // printMatrix(ldb, nrhs, y_label, ldb, "y_label");
 // printf("=====\n");



double ssr_joint = 1;
double ssr_own = 1;
ssr_joint = Ols(m, lda, ldb, nrhs, lag_matrix, y_label, y_joint);
 // printf("y_label = (matlab base-1)\n");
 // printMatrix(ldb, nrhs, y_joint, ldb, "y_label");
 // printf("=====\n");


ssr_own = Ols(p_lag+1, lda, ldb, nrhs, lag_x1, y_label, y_own);
 // printf("y_label = (matlab base-1)\n");
 // printMatrix(ldb, nrhs, y_own, ldb, "y_label");
 // printf("=====\n");

start = high_resolution_clock::now();

 double x_chi2 = rows*(ssr_own - ssr_joint) / ssr_joint;

 double p_value = 1.0;

 if (x_chi2 > 0)
 {
 	 Chi2PLookup Chi2PLookupTable;
 	 p_value = Chi2PLookupTable.getPValue(x_chi2, p_lag);
 }

    end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

 printf("ssr joint %f \n", ssr_joint);
 printf("ssr own %f \n", ssr_own);

 printf("chi2x  %f  P value %f \n ", x_chi2, p_value);

 printf("Chi2 test time: %f ms \n", duration_sec.count());



// free resources

 return 0;
}

