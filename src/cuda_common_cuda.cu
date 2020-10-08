#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.hpp"

//cudaError_t err;

//extern "c"
void _cuda_safe_mem (cudaError_t err, const char *file, unsigned int line){

  if(cudaSuccess != err) {
    fprintf(stderr, "Cuda Memory error at %s:%u.\n", file, line);
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    if (err == cudaErrorInvalidValue)
      fprintf(stderr, "You may have tried to allocate zero memory at %s:%u.\n", file, line);
    //errexit();
    exit(1);
  } else {
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Error found during memory operation. Possibly however from an failed operation before. %s:%u.\n", file, line);
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorInvalidValue)
        fprintf(stderr, "You may have tried to allocate zero memory before %s:%u.\n", file, line);
      //errexit();
      exit(1);
    }
  }
}

//extern "c"
void _cuda_check_errors (const dim3 &block, const dim3 &grid, const char *function, const char *file, unsigned int line) {

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf (stderr, "error \"%s\" calling %s with dim %d %d %d, grid %d %d %d in %s:%u\n", 
             cudaGetErrorString(err), function, block.x, block.y, block.z, grid.x, grid.y, grid.z, file, line);
    //errexit();
    exit(1);
  }
}

