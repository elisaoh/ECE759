#define CUB_STDERR // print CUDA runtime errors to console
#include <cub/device/device_scan.cuh>
#include <cub/util_allocator.cuh>
#include <iostream>
#include <stdio.h>
#include <time.h>
using namespace cub;
CachingDeviceAllocator g_allocator(true); // Caching allocator for device memory

void randomArray(float *arr, unsigned int arraySize) {
  for (unsigned i = 0; i < arraySize; i++)
    arr[i] = rand();
}



int main(int argc, char *argv[]) {
  const unsigned int num_items = atoi(argv[1]);

  // Set up host arrays
  float *h_in = new float[num_items];
  srand(time(NULL));
  randomArray(h_in, num_items);

  // Set up device arrays
  float *d_in = NULL;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_in, sizeof(float) * num_items));

  // Initialize device input
  CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items,
                          cudaMemcpyHostToDevice));

  // Setup device output array
  float *d_sum = NULL;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&d_sum, sizeof(float) * num_items));

  // Request and allocate temporary storage
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                        d_in, d_sum, num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Do the actual reduce operation
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                        d_in, d_sum, num_items));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  float *gpu_sum = new float[num_items];
  CubDebugExit(cudaMemcpy(gpu_sum, d_sum, sizeof(float) * num_items,
                          cudaMemcpyDeviceToHost));

  std::cout << *(gpu_sum + num_items - 1) << std::endl;

  std::cout << ms << std::endl;

  // Cleanup
  if (d_in)
    CubDebugExit(g_allocator.DeviceFree(d_in));
  if (d_sum)
    CubDebugExit(g_allocator.DeviceFree(d_sum));
  if (d_temp_storage)
    CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
  delete[] h_in;
  delete[] gpu_sum;

  return 0;
}