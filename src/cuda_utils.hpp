#ifndef _CUDA_UTILS_HPP
#define _CUDA_UTILS_HPP

//#include <cuda.h>
//#include <cuda_runtime.h>

//extern "c"
void _cuda_safe_mem (cudaError_t err, const char *file, unsigned int line);

//extern "c"
void _cuda_check_errors (const dim3 &block, const dim3 &grid, const char *function, const char *file, unsigned int line);

#define cuda_safe_mem(a) _cuda_safe_mem((a), __FILE__, __LINE__)

#define KERNELCALL(_f, _g, _b, _params) \
  _f<<<_g, _b>>>_params; \
  _cuda_check_errors(_b, _g, #_f, __FILE__, __LINE__); 


//#define KERNELCALL_shared(_f, _a, _b, _s, _params) \
//  _f<<<_a, _b, _s, stream[0]>>>_params;  \
//  _cuda_check_errors(_a, _b, #_f, __FILE__, __LINE__);
//
//#define KERNELCALL_stream(_function, _grid, _block, _stream, _params) \
//  _function<<<_grid, _block, 0, _stream>>>_params; \
//  _cuda_check_errors(_grid, _block, #_function, __FILE__, __LINE__);
//
//#define KERNELCALL(_f, _a, _b, _params) KERNELCALL_shared(_f, _a, _b, 0, _params)






#endif  // _CUDA_UTILS_HPP
