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
#include <cuml/tsa/batched_arima.hpp>
#include <cuml/tsa/batched_kalman.hpp>

#include "common/cumlHandle.hpp"
#include "common/nvtx.hpp"
#include "cuda_utils.h"
#include "linalg/batched/matrix.h"
#include "linalg/matrix_vector_op.h"
#include "metrics/batched/information_criterion.h"
#include "timeSeries/arima_helpers.h"
#include "utils.h"

namespace ML {



/**
 * Auxiliary function of _start_params: least square 
 * approximation of a granger causality test
 */
void _granger_least_squares(cumlHandle& handle, 
                         double* d_ar, //
                         double* d_ma, //
                         double* variance, //
                         const MLCommon::LinAlg::Batched::Matrix<double>& X1,
                         const MLCommon::LinAlg::Batched::Matrix<double>& X2,
                         int p,  //p = max value of lag of x1's 
                         int q,  //q = max value of lag of x2's (q=p in our situation)
                         bool estimate, 
                         int bias_len = 1, // here 
                         double* weight = nullptr) {

  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();
  auto counting = thrust::make_counting_iterator(0);

  int batch_size = X1.batches();
  int n_obs = X1.shape().first;

  int r = p+q;
  int k = 1; 
  if (estimate_sigma2) {
    thrust::device_ptr<double> variance_thrust =
      thrust::device_pointer_cast(variance);
    thrust::fill(thrust::cuda::par.on(stream), variance_thrust,
                 variance_thrust + batch_size, 1.0);
  }


  /* Matrix formed by lag matrices of y and the residuals respectively,
   * side by side. The left side will be used to estimate AR, the right
   * side to estimate MA */
  MLCommon::LinAlg::Batched::Matrix<double> bm_ls_ar_res(
    n_obs - r, p + q + k, batch_size, cublas_handle, allocator, stream, false);
  int ar_offset = r - p;
  int res_offset = r - p - q;

  // Create lagged matrix X1 
  int ls_height = n_obs - p;
  MLCommon::LinAlg::Batched::Matrix<double> X1_mt =
    MLCommon::LinAlg::Batched::b_lagged_mat(X1, p);

  // Create lagged matrix X2
  MLCommon::LinAlg::Batched::Matrix<double> X2_mt =
    MLCommon::LinAlg::Batched::b_lagged_mat(X2, p);

  //use only the last (p-1) columns for both X1 and X2 as A
  // X1_fit = X1[:,1:]; X2_fit = X2[:,1:];
  MLCommon::LinAlg::Batched::Matrix<double> X1_fit =
    MLCommon::LinAlg::Batched::b_2dcopy(X1_mt, p*2, 0, ls_height, 1);

  MLCommon::LinAlg::Batched::Matrix<double> X2_fit =
    MLCommon::LinAlg::Batched::b_2dcopy(X2_mt, p*2, 0, ls_height, 1);


  // put X1 and X2 together to be X = [X1:X2] (do we need to add bias?)
  // this "concatinate" function isn't finished yet
  MLCommon::LinAlg::Batched::Matrix<double> X_fit =
    MLCommon::LinAlg::Batched::concatinate(X1_fit,X2_fit,bias_len);

  // Generate true outputs Y for the model fit
  MLCommon::LinAlg::Batched::Matrix<double> Y =
    MLCommon::LinAlg::Batched::b_2dcopy(X1_mt, p*2, 0, ls_height, 1);


  // initialize a residual
  MLCommon::LinAlg::Batched::Matrix<double> Y_residual(
    n_obs - r, 1, batch_size, cublas_handle, allocator, stream, false);
  if (estimate_sigma2) {
    MLCommon::copy(Y_residual.raw_data(), Y.raw_data(),
                   (n_obs - r) * batch_size, stream);
  }

  // Initial AR fit
  MLCommon::LinAlg::Batched::b_gels(X_fit, Y);

  // Copy the results to the weight vectors
  const double* weight_ori = Y.raw_data();
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int bid) {
                     for (int i = 0; i < p+q + bias_len; i++) {
                       weight[i] = weight_ori[i];
                     }
                   });

  if (estimate) {
    // Compute final residual (technically a gemv)
    MLCommon::LinAlg::Batched::b_gemm(false, false, n_obs - r, 1, p + q + k,
                                      -1.0, X_fit, weight, 1.0, Y_residual);

    // Compute variance
    double* d_residual = Y_residual.raw_data();
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int bid) {
                       double acc = 0.0;
                       const double* b_residual =
                         d_residual + (n_obs - r) * bid;
                       for (int i = q; i < n_obs - r; i++) {
                         double res = b_residual[i];
                         acc += res * res;
                       }
                       variance[bid] = acc / static_cast<double>(n_obs - r - q);
                     });
  }
}	
/**
 * Auxiliary function of estimate_x0: compute the starting parameters for
 * the series pre-processed by estimate_x0
 */
void _start_params(cumlHandle& handle, ARIMAParams<double>& params,
                   const MLCommon::LinAlg::Batched::Matrix<double>& X1,
                   const MLCommon::LinAlg::Batched::Matrix<double>& X2,
                   const ARIMAOrder& order) {
  // Estimate a granger fit
    _granger_least_squares(handle, params.ar, params.ma, params.sigma2, X1, X2,
                        order.p, order.q, 1, true, order.k, params.mu);

}

void estimate_x0(cumlHandle& handle, ARIMAParams<double>& params,
                 const double* d_y, int batch_size, int n_obs,
                 const ARIMAOrder& order) {
  ML::PUSH_RANGE(__func__);
  const auto& handle_impl = handle.getImpl();
  auto stream = handle_impl.getStream();
  auto cublas_handle = handle_impl.getCublasHandle();
  auto allocator = handle_impl.getDeviceAllocator();

  // Difference if necessary, copy otherwise
  MLCommon::LinAlg::Batched::Matrix<double> X1d(
    n_obs - order.d - order.s * order.D, 1, batch_size, cublas_handle,
    allocator, stream, false);
  MLCommon::TimeSeries::prepare_data(X1d.raw_data(), d_y, batch_size, n_obs,
                                     order.d, order.D, order.s, stream);
  // Note: mu is not known yet! We just want to difference the data

  // Do the computation of the initial parameters
  _start_params(handle, params, X1d, order);
  ML::POP_RANGE();
}

}  // namespace MLCommon