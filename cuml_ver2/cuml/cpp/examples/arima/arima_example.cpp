/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifdef HAVE_CUB
#include <cuml/common/cubAllocatorAdapter.hpp>
#endif  //HAVE_CUB

#ifdef HAVE_RMM
#include <rmm/rmm.h>
#include <cuml/common/rmmAllocatorAdapter.hpp>
#endif  //HAVE_RMM

#include <cuml/tsa/arima_common.h>
#include <cuml/tsa/batched_arima.hpp>
#include "time_series_datasets.h"
#include <cuml/cuml.hpp>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }
#endif  //CUDA_RT_CALL

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg,
             const T default_val) {
  T argval = default_val;
  char** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
  char** itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

void printUsage() {
  std::cout
    << "To run default example use:" << std::endl
    << "    dbscan_example [-dev_id <GPU id>]" << std::endl
    << "For other cases:" << std::endl
    << "    dbscan_example [-dev_id <GPU id>] -input <samples-file> "
    << "-num_samples <number of samples> -num_features <number of features> "
    << "[-min_pts <minimum number of samples in a cluster>] "
    << "[-eps <maximum distance between any two samples of a cluster>] "
    << "[-max_bytes_per_batch <maximum memory to use (in bytes) for batch size "
       "calculation>] "
    << std::endl;
  return;
}

void loadDefaultDataset(std::vector<double>& h_x1, std::vector<double>& h_x2, size_t& nobs) {
  constexpr size_t NUM_OBS = 10;

  constexpr double x1[NUM_OBS] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0};
  constexpr double x2[NUM_OBS] = {0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0};


  nobs = NUM_OBS;
  h_x1.insert(h_x1.begin(), x1, x1 + nobs);
  h_x2.insert(h_x2.begin(), x2, x2 + nobs);
}

int main(int argc, char* argv[]) {
  int devId = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  size_t nobs = get_argval<size_t>(argv, argv + argc, "-num_samples", 0);
  // size_t nRows = get_argval<size_t>(argv, argv + argc, "-num_samples", 0);
  // size_t nCols = get_argval<size_t>(argv, argv + argc, "-num_features", 0);
  // std::string input =
  //   get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
  // int minPts = get_argval<int>(argv, argv + argc, "-min_pts", 3);
  // float eps = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
  // size_t max_bytes_per_batch =
  //   get_argval<size_t>(argv, argv + argc, "-max_bytes_per_batch", (size_t)13e9);

  {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaSetDevice(devId);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not select CUDA device with the id: " << devId
                << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
    cudaStatus = cudaFree(0);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not initialize CUDA on device: " << devId
                << "(" << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
  }

  // define a handle here

  ML::cumlHandle cumlHandle;


#ifdef HAVE_RMM
  rmmOptions_t rmmOptions;
  rmmOptions.allocation_mode = PoolAllocation;
  rmmOptions.initial_pool_size = 0;
  rmmOptions.enable_logging = false;
  rmmError_t rmmStatus = rmmInitialize(&rmmOptions);
  if (RMM_SUCCESS != rmmStatus) {
    std::cerr << "WARN: Could not initialize RMM: "
              << rmmGetErrorString(rmmStatus) << std::endl;
  }
#endif  //HAVE_RMM
#ifdef HAVE_RMM
  std::shared_ptr<ML::deviceAllocator> allocator(new ML::rmmAllocatorAdapter());
#elif defined(HAVE_CUB)
  std::shared_ptr<ML::deviceAllocator> allocator(
    new ML::cachingDeviceAllocator());
#else
  std::shared_ptr<ML::deviceAllocator> allocator(
    new ML::defaultDeviceAllocator());
#endif  // HAVE_RMM
  cumlHandle.setDeviceAllocator(allocator);











  std::vector<double> h_x1;
  std::vector<double> h_x2;


    // Samples file not specified, run with defaults
    std::cout << "Samples file not specified. (-input option)" << std::endl;
    std::cout << "Running with default dataset:" << std::endl;
    loadDefaultDataset(h_x1, h_x2, nobs);
 
    //else if (nRows == 0 || nCols == 0) {
  //   // Samples file specified but nRows and nCols is not specified
  //   // Print usage and quit
  //   std::cerr << "Samples file: " << input << std::endl;
  //   std::cerr << "Incorrect value for (num_samples x num_features): (" << nRows
  //             << " x " << nCols << ")" << std::endl;
  //   printUsage();
  //   return 1;
  // } else {
  //   // All options are correctly specified
  //   // Try to read input file now
  //   std::ifstream input_stream(input, std::ios::in);
  //   if (!input_stream.is_open()) {
  //     std::cerr << "ERROR: Could not open input file " << input << std::endl;
  //     return 1;
  //   }
  //   std::cout << "Trying to read samples from " << input << std::endl;
  //   h_inputData.reserve(nRows * nCols);
  //   float val = 0.0;
  //   while (input_stream >> val) {
  //     h_inputData.push_back(val);
  //   }
  //   if (h_inputData.size() != nRows * nCols) {
  //     std::cerr << "ERROR: Read " << h_inputData.size() << " from " << input
  //               << ", while expecting to read: " << nRows * nCols
  //               << " (num_samples*num_features)" << std::endl;
  //     return 1;
  //   }
  // }







  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  cumlHandle.setStream(stream);

	double* d_x1 = nullptr;
	double* d_x2 = nullptr;


  CUDA_RT_CALL(cudaMalloc((void**)&d_x1, nobs * sizeof(double)));
  CUDA_RT_CALL(cudaMalloc((void**)&d_x2, nobs * sizeof(double)));
  CUDA_RT_CALL(cudaMemcpyAsync(d_x1, h_x1.data(),
                               nobs * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
  CUDA_RT_CALL(cudaMemcpyAsync(d_x2, h_x2.data(),
                               nobs * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
	
	double * h_back = new double[nobs];

    CUDA_RT_CALL(cudaMemcpyAsync(h_back, d_x1,
                               nobs * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));

  std::cout <<"print h_x1 "<< *h_x1.begin() << std::endl;
  std::cout <<"print h_back "<< *h_back << std::endl;


  ML::ARIMAParams<double> params;
  ML::ARIMAOrder order = {1,1,1,1,1,1,1,1};
  ML::GrangerOrder gorder = {4,4,1};

  std::cout << gorder.p << std::endl;
  int batch_size = 1;


  ML::printsth(gorder);
  // ML::granger_causality_test(cumlHandle, gorder, d_x1, d_x2, batch_size, nobs);
  

  CUDA_RT_CALL(cudaStreamSynchronize(stream));


  CUDA_RT_CALL(cudaStreamDestroy(stream));
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return 0;
}
