/*
	ECE759 final project
	author: Xucheng Wan, Elisa OU
	granger causality
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuml/cuml.hpp>
#include <cuml/tsa/granger.hpp>

#include "common/cumlHandle.hpp"
#include "common/nvtx.hpp"
#include "cuda_utils.h"
#include "linalg/batched/matrix.h"
#include "linalg/matrix_vector_op.h"
#include "metrics/batched/information_criterion.h"
#include "timeSeries/granger_helpers.h"
#include "utils.h"

namespace ML {


/**
 * Auxiliary function of _start_params: least square 
 * approximation of a granger causality test
 */
void _granger_least_squares(cumlHandle& handle, 
                         const MLCommon::LinAlg::Batched::Matrix<double>& X1,
                         const MLCommon::LinAlg::Batched::Matrix<double>& X2,
                         int p,  //p = max value of lag of x1's 
                         int q,  //q = max value of lag of x2's (q=p in our situation)
                         bool estimate, 
                         int bias_len = 1) {

  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();
  auto counting = thrust::make_counting_iterator(0);

  int batch_size = X1.batches();
  int n = X1.shape().first;

  int r = p*2;
  int k = 1; 
  if (estimate) { 
    thrust::device_ptr<double> variance_thrust =
      thrust::device_pointer_cast(variance);
    thrust::fill(thrust::cuda::par.on(stream), variance_thrust,
                 variance_thrust + batch_size, 1.0);
  }

                         
  // Create lagged matrix X1
  int len = X1.shape().first * X1.shape().second;
  int ls_height = len - p;
  MLCommon::LinAlg::Batched::Matrix<double> X1_mt =
    MLCommon::LinAlg::Batched::b_lagged_mat(X1, p);

  // Create lagged matrix X2
  MLCommon::LinAlg::Batched::Matrix<double> X2_mt =
    MLCommon::LinAlg::Batched::b_lagged_mat(X2, p);

  //use only the last (p-1) columns for both X1 and X2 as A
  // X1_fit = X1[:,1:]; X2_fit = X2[:,1:];
  MLCommon::LinAlg::Batched::Matrix<double> X1_fit =
    MLCommon::LinAlg::Batched::b_2dcopy(X1_mt, 0, 1, ls_height, p);

  MLCommon::LinAlg::Batched::Matrix<double> X2_fit =
    MLCommon::LinAlg::Batched::b_2dcopy(X2_mt, 0, 1, ls_height, p);

  // now size of X_fit is (n-p)x(p-1)
  // put X1 and X2 together to be X = [X1:X2] (do we need to add bias?)
  // this "concatinate" function isn't finished yet
  MLCommon::LinAlg::Batched::Matrix<double> X_fit =
    MLCommon::LinAlg::Batched::concatinate(X1_fit,X2_fit,bias_len);

  // Generate true outputs Y for the model fit
  MLCommon::LinAlg::Batched::Matrix<double> Y =
    MLCommon::LinAlg::Batched::b_2dcopy(X1_mt, 0, 0, ls_height, 1);


  // initialize a residual
  MLCommon::LinAlg::Batched::Matrix<double> Y_residual(
    n - p, 1, batch_size, cublas_handle, allocator, stream, false);
  if (estimate) {
    MLCommon::copy(Y_residual.raw_data(), Y.raw_data(),
                   (n - p) * batch_size, stream);
  }

  // use cublas to solve the least square
  // the results are stored back to Y, while X_fit stores the QR factorized elems
  MLCommon::LinAlg::Batched::b_gels(X_fit, Y);

  // Cut results into the weight vectors
  MLCommon::LinAlg::Batched::Matrix<double> weight =
    MLCommon::LinAlg::Batched::b_2dcopy(Y, 0, 0, ls_height, 1);


  if (estimate) {
    // Compute final residual (technically a gemv)
    MLCommon::LinAlg::Batched::b_gemm(false, false, n - r, 1, p + q + k,
                                      -1.0, X_fit, weight, 1.0, Y_residual);

    // Compute variance
    double* d_residual = Y_residual.raw_data();
    double* variance[p*2];
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       double acc = 0.0;
                       const double* b_residual =
                         d_residual + (n - r) * bid;
                       for (int i = q; i < n - r; i++) {
                         double res = b_residual[i];
                         acc += res * res;
                       }
                       variance[bid] = acc / static_cast<double>(n - r - q);
                     });
  }
}	
/**
 * Auxiliary function of estimate_x0: compute the starting parameters for
 * the series pre-processed by estimate_x0
 */
void _start_params(cumlHandle& handle, GrangerParams<double>& params,
                   const MLCommon::LinAlg::Batched::Matrix<double>& X1,
                   const MLCommon::LinAlg::Batched::Matrix<double>& X2,) {

  // Estimate a granger fit
    _granger_least_squares(handle, X1, X2,
                        order.p, order.q, true, order.bias_len);

}

// main function
void granger_causality_test(cumlHandle& handle, GrangerOrder order,
                 const double* d_x1, const double* d_x2, int batch_size = 1, int n) {

  ML::PUSH_RANGE(__func__);
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // prepare X1 and X2
  MLCommon::LinAlg::Batched::Matrix<double> X1(
    n, 1, 1, cublas_handle,allocator, stream, false);
  MLCommon::LinAlg::Batched::Matrix<double> X2(
    n, 1, 1, cublas_handle,allocator, stream, false);
  MLCommon::TimeSeries::prepare_data(X1.raw_data(), d_x1, batch_size, n, stream);
  MLCommon::TimeSeries::prepare_data(X2.raw_data(), d_x2, batch_size, n, stream);

  // Note: mu is not known yet! We just want to difference the data

  // Do the computation of the initial parameters
  _start_params(handle, order, X1, X2);
  ML::POP_RANGE();
}

void printsth(GrangerOrder order){

  std::cout << "order.p = " << order.p << std::endl;

}

}  // namespace MLCommon