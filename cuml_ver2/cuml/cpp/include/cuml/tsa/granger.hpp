/*
author xucheng wan
 */

#pragma once

#include <cuml/tsa/arima_common.h>
#include <cuml/cuml.hpp>

namespace ML {

/**
 * Provide initial estimates to ARIMA parameters mu, AR, and MA
 *
 * @param[in]  handle      cuML handle
 * @param[in]  params      ARIMA parameters (device)
 * @param[in]  d_y         Series to fit: shape = (nobs, batch_size) and
 *                         expects column major data layout. (device)
 * @param[in]  batch_size  Total number of batched time series
 * @param[in]  nobs        Number of samples per time series
 *                         (all series must be identical)
 * @param[in]  order       ARIMA hyper-parameters
 */

void granger_causality_test(cumlHandle& handle, GrangerOrder order,
                 const double* d_x1, const double* d_x2, int batch_size = 1, int n=10);

void printsth(GrangerOrder order);

}  // namespace ML
