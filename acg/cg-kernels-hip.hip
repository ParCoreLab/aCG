/* This file is part of acg.
 *
 * Copyright 2025 Koç University and Simula Research Laboratory
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *  James D. Trotter <james@simula.no>
 *  Sinan Ekmekçibaşı <sekmekcibasi23@ku.edu.tr>
 *
 * Last modified: 2025-04-26
 *
 * HIP kernels for CG solvers
 */

#include "acg/config.h"
#include "acg/cg-kernels-hip.h"
#include "acg/cghip.h"
#include "acg/comm.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/error.h"
#include "acg/time.h"
#include "acg/vector.h"

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

__constant__ double minus_one;
__constant__ double one;
__constant__ double zero;

int acgsolverhip_init_constants(
    double ** d_minus_one,
    double ** d_one,
    double ** d_zero)
{
    hipError_t err;
    double h_minus_one = -1.0, h_one = 1.0, h_zero = 0.0;
    err = hipMemcpyToSymbol(HIP_SYMBOL(minus_one), &h_minus_one, sizeof(double), 0, hipMemcpyHostToDevice);
    if (err) return ACG_ERR_HIP;
    err = hipGetSymbolAddress((void **) d_minus_one, minus_one);
    if (err) return ACG_ERR_HIP;
    err = hipMemcpyToSymbol(HIP_SYMBOL(one), &h_one, sizeof(double), 0, hipMemcpyHostToDevice);
    if (err) return ACG_ERR_HIP;
    err = hipGetSymbolAddress((void **) d_one, one);
    if (err) return ACG_ERR_HIP;
    err = hipMemcpyToSymbol(HIP_SYMBOL(zero), &h_zero, sizeof(double), 0, hipMemcpyHostToDevice);
    if (err) return ACG_ERR_HIP;
    err = hipGetSymbolAddress((void **) d_zero, zero);
    if (err) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

__global__ void acgsolverhip_alpha_kernel(
    double * alpha,
    double * minus_alpha,
    const double * rnrm2sqr,
    const double * pdott)
{
    if (threadIdx.x == 0) {
        *alpha = *rnrm2sqr / *pdott;
        *minus_alpha = -(*rnrm2sqr / *pdott);
    }
}

int acgsolverhip_alpha(
    double * d_alpha,
    double * d_minus_alpha,
    const double * d_rnrm2sqr,
    const double * d_pdott)
{
    acgsolverhip_alpha_kernel<<<1,1>>>(d_alpha, d_minus_alpha, d_rnrm2sqr, d_pdott);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

__global__ void acgsolverhip_beta_kernel(
    double * beta,
    const double * rnrm2sqr,
    const double * rnrm2sqr_prev)
{
    if (threadIdx.x == 0) *beta = *rnrm2sqr / *rnrm2sqr_prev;
}

int acgsolverhip_beta(
    double * d_beta,
    const double * d_rnrm2sqr,
    const double * d_rnrm2sqr_prev)
{
    acgsolverhip_beta_kernel<<<1,1,0>>>(d_beta, d_rnrm2sqr, d_rnrm2sqr_prev);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

__launch_bounds__(256)
__global__ void acgsolverhip_daxpy_alpha_kernel(
    int n,
    const double * __restrict__ rnrm2sqr,
    const double * __restrict__ pdott,
    const double * __restrict__ x,
    double * __restrict__ y)
{
    const double a = (*rnrm2sqr) / (*pdott);
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < n;
    //      i += blockDim.x*gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
    {
        y[i] = a*x[i]+y[i];
        // __builtin_nontemporal_store(a*x[i]+y[i], &y[i]);
    }
}

int acgsolverhip_daxpy_alpha(
    int n,
    const double * __restrict__ d_rnrm2sqr,
    const double * __restrict__ d_pdott,
    const double * __restrict__ d_x,
    double * __restrict__ d_y)
{
    // static int mingridsize = 0, blocksize = 0;
    // if (mingridsize == 0 && blocksize == 0) {
    //     hipOccupancyMaxPotentialBlockSize(
    //         &mingridsize, &blocksize, acgsolverhip_daxpy_alpha_kernel, 0, 0);
    // }
    int mingridsize = (n + 255) / 256, blocksize = 256;
    acgsolverhip_daxpy_alpha_kernel<<<mingridsize,blocksize>>>(
        n, d_rnrm2sqr, d_pdott, d_x, d_y);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

__launch_bounds__(256)
__global__ void acgsolverhip_daxpy_minus_alpha_kernel(
    int n,
    const double * __restrict__ rnrm2sqr,
    const double * __restrict__ pdott,
    const double * __restrict__ x,
    double * __restrict__ y)
{
    const double a = -(*rnrm2sqr) / (*pdott);
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < n;
    //      i += blockDim.x*gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
    {
        y[i] = a*x[i]+y[i];
        // __builtin_nontemporal_store(a*x[i]+y[i], &y[i]);
    }
}

int acgsolverhip_daxpy_minus_alpha(
    int n,
    const double * __restrict__ d_rnrm2sqr,
    const double * __restrict__ d_pdott,
    const double * __restrict__ d_x,
    double * __restrict__ d_y)
{
    // static int mingridsize = 0, blocksize = 0;
    // if (mingridsize == 0 && blocksize == 0) {
    //     hipOccupancyMaxPotentialBlockSize(
    //         &mingridsize, &blocksize, acgsolverhip_daxpy_minus_alpha_kernel, 0, 0);
    // }
    int mingridsize = (n + 255) / 256, blocksize = 256;
    acgsolverhip_daxpy_minus_alpha_kernel<<<mingridsize,blocksize>>>(
        n, d_rnrm2sqr, d_pdott, d_x, d_y);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

__launch_bounds__(256)
__global__ void acgsolverhip_pipelined_daxpy_fused_kernel(
    int n,
    const double * gamma,
    double * gamma_prev,
    const double * delta,
    const double * q,
    double * p,
    double * r,
    double * t,
    double * x,
    double * z,
    double * w,
    double * alpha_prev)
{
    double beta = (*gamma) / (*gamma_prev);
    double alpha = (*gamma) / ((*delta) - beta*(*gamma)/(*alpha_prev));
    /* for (int i = blockIdx.x*blockDim.x+threadIdx.x; */
    /*      i < n; */
    /*      i += blockDim.x*gridDim.x) */
    /* { */
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
    {
        z[i] = q[i]+beta*z[i];
        t[i] = w[i]+beta*t[i];
        p[i] = r[i]+beta*p[i];
        x[i] += alpha*p[i];
        r[i] -= alpha*t[i];
        w[i] -= alpha*z[i];
    }

    /* cg::grid_group grid = cg::this_grid(); */
    /* cg::sync(grid); */
    /* if (grid.thread_rank() == 0) { */
    /*     *gamma_prev = *gamma; */
    /*     *alpha_prev = alpha; */
    /* } */
}

__launch_bounds__(1)
__global__ void acgsolverhip_pipelined_reset_scalars(
    const double * gamma,
    double * gamma_prev,
    const double * delta,
    double * alpha_prev)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double beta = (*gamma) / (*gamma_prev);
        double alpha = (*gamma) / ((*delta) - beta*(*gamma)/(*alpha_prev));
        *gamma_prev = *gamma;
        *alpha_prev = alpha;
    }
}

int acgsolverhip_pipelined_daxpy_fused(
    int n,
    const double * d_gamma,
    double * d_gamma_prev,
    const double * d_delta,
    const double * d_q,
    double * d_p,
    double * d_r,
    double * d_t,
    double * d_x,
    double * d_z,
    double * d_w,
    double * d_alpha_prev,
    hipStream_t stream)
{
    /* static int mingridsize = 0, blocksize = 0, sharedmemsize = 0; */
    /* if (mingridsize == 0 && blocksize == 0) { */
    /*     hipOccupancyMaxPotentialBlockSize( */
    /*         &mingridsize, &blocksize, acgsolverhip_pipelined_daxpy_fused_kernel, 0, 0); */
    /* } */
    int mingridsize = (n + 255) / 256, blocksize = 256;
    acgsolverhip_pipelined_daxpy_fused_kernel<<<mingridsize,blocksize,0,stream>>>(
        n, d_gamma, d_gamma_prev, d_delta,
        d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha_prev);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;

    acgsolverhip_pipelined_reset_scalars<<<1,1,0,stream>>>(
        d_gamma, d_gamma_prev, d_delta, d_alpha_prev);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;

    /* launch device-side CG kernel */
    /* void * kernelargs[] = { */
    /*     (void *) &n, */
    /*     (void *) &d_gamma, */
    /*     (void *) &d_gamma_prev, */
    /*     (void *) &d_delta, */
    /*     (void *) &d_q, */
    /*     (void *) &d_p, */
    /*     (void *) &d_r, */
    /*     (void *) &d_t, */
    /*     (void *) &d_x, */
    /*     (void *) &d_z, */
    /*     (void *) &d_w, */
    /*     (void *) &d_alpha_prev }; */
    /* dim3 blockDim(blocksize, 1, 1); */
    /* dim3 gridDim(mingridsize, 1, 1); */
    /* hipLaunchCooperativeKernel( */
    /*     (void *) acgsolverhip_pipelined_daxpy_fused_kernel, gridDim, blockDim, */
    /*     kernelargs, sharedmemsize, stream); */

    return ACG_SUCCESS;
}

__launch_bounds__(256)
__global__ void acgsolverhip_daypx_beta_kernel(
    int n,
    const double * __restrict__ rnrm2sqr,
    const double * __restrict__ rnrm2sqr_prev,
    double * __restrict__ y,
    const double * __restrict__ x)
{
    double a = (*rnrm2sqr) / (*rnrm2sqr_prev);
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < n;
    //      i += blockDim.x*gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
    {
        y[i] = a*y[i]+x[i];
        // __builtin_nontemporal_store(a*y[i]+x[i], &y[i]);
    }
}

int acgsolverhip_daypx_beta(
    int n,
    const double * __restrict__ d_rnrm2sqr,
    const double * __restrict__ d_rnrm2sqr_prev,
    double * __restrict__ d_y,
    const double * __restrict__ d_x)
{
    // static int mingridsize = 0, blocksize = 0;
    // if (mingridsize == 0 && blocksize == 0) {
    //     hipOccupancyMaxPotentialBlockSize(
    //         &mingridsize, &blocksize, acgsolverhip_daypx_beta_kernel, 0, 0);
    // }
    int mingridsize = (n + 255) / 256, blocksize = 256;
    acgsolverhip_daypx_beta_kernel<<<mingridsize,blocksize>>>(
        n, d_rnrm2sqr, d_rnrm2sqr_prev, d_y, d_x);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

/*
 * Monolithic CG (HIP)
 */

#define THREADS_PER_BLOCK 512
#define TASKS_PER_THREAD 10

__global__ void csrgemv_merge_startrows(
    acgidx_t n,
    const acgidx_t * __restrict rowptr,
    acgidx_t nstartrows,
    acgidx_t * startrows)
{
    acgidx_t nnzs = rowptr[n];
    acgidx_t ntasks = n+nnzs;
    acgidx_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    for (acgidx_t l = tid;
         l < ntasks/TASKS_PER_THREAD;
         l += blockDim.x*gridDim.x)
    {
        /* binary search to find starting row for each thread */
        acgidx_t i = 0;
        acgidx_t count = n;
        while (count > 0) {
            acgidx_t row = i;
            acgidx_t step = count >> 1;
            row += step;
            if (rowptr[row+1] <= l*TASKS_PER_THREAD-row-1) {
                i = ++row;
                count -= step + 1;
            } else { count = step; }
        }
        startrows[l] = i;
    }
}

/*
 * These functions are from:
 *
 * https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h#L171
 */

// Since threads in a wave do not make independent progress, __activemask()
// always returns the exact active mask, i.e, all active threads in the wave.
__device__
inline
unsigned long long __activemask() {
  return __ballot(true);
}

template <typename T>
__device__ inline
T __hip_readfirstlane(T val) {
  // In theory, behaviour is undefined when reading from a union member other
  // than the member that was last assigned to, but it works in practice because
  // we rely on the compiler to do the reasonable thing.
  union {
    unsigned long long l;
    T d;
  } u;
  u.d = val;
  // NOTE: The builtin returns int, so we first cast it to unsigned int and only
  // then extend it to 64 bits.
  unsigned long long lower = (unsigned)__builtin_amdgcn_readfirstlane(u.l);
  unsigned long long upper =
      (unsigned)__builtin_amdgcn_readfirstlane(u.l >> 32);
  u.l = (upper << 32) | lower;
  return u.d;
}

// When compiling for wave32 mode, ignore the upper half of the 64-bit mask.
#define __hip_adjust_mask_for_wave32(MASK)            \
  do {                                          \
    if (warpSize == 32) MASK &= 0xFFFFFFFF;     \
  } while (0)

// We use a macro to expand each builtin into a waterfall that implements the
// mask semantics:
//
// 1. The mask argument may be divergent.
// 2. Each active thread must have its own bit set in its own mask value.
// 3. For a given mask value, all threads that are mentioned in the mask must
//    execute the same static instance of the builtin with the same mask.
// 4. The union of all mask values supplied at a static instance must be equal
//    to the activemask at the program point.
//
// Thus, the mask argument partitions the set of currently active threads in the
// wave into disjoint subsets that cover all active threads.
//
// Implementation notes:
// ---------------------
//
// We implement this as a waterfall loop that executes the builtin for each
// subset separately. The return value is a divergent value across the active
// threads. The value for inactive threads is defined by each builtin
// separately.
//
// As long as every mask value is non-zero, we don't need to check if a lane
// specifies itself in the mask; that is done by the later assertion where all
// chosen lanes must be in the chosen mask.

#define __hip_check_mask(MASK)                                                 \
  do {                                                                         \
    __hip_assert(MASK && "mask must be non-zero");                             \
    bool done = false;                                                         \
    while (__any(!done)) {                                                     \
      if (!done) {                                                             \
        auto chosen_mask = __hip_readfirstlane(MASK);                          \
        if (MASK == chosen_mask) {                                             \
          __hip_assert(MASK == __ballot(true) &&                               \
                       "all threads specified in the mask"                     \
                       " must execute the same operation with the same mask"); \
          done = true;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while(0)

// __match_any, __match_all and sync variants

template <typename T>
__device__ inline
unsigned long long __match_any(T value) {
  static_assert(
      (__hip_internal::is_integral<T>::value || __hip_internal::is_floating_point<T>::value) &&
          (sizeof(T) == 4 || sizeof(T) == 8),
      "T can be int, unsigned int, long, unsigned long, long long, unsigned "
      "long long, float or double.");
  bool done = false;
  unsigned long long retval = 0;

  while (__any(!done)) {
    if (!done) {
      T chosen = __hip_readfirstlane(value);
      if (chosen == value) {
        retval = __activemask();
        done = true;
      }
    }
  }

  return retval;
}

template <typename MaskT, typename T>
__device__ inline
unsigned long long __match_any_sync(MaskT mask, T value) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __match_any(value) & mask;
}

template <typename MaskT, typename T>
__device__ inline
T __shfl_sync(MaskT mask, T var, int srcLane,
              int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl(var, srcLane, width);
}

template <typename MaskT, typename T>
__device__ inline
T __shfl_down_sync(MaskT mask, T var, unsigned int delta,
                                     int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_down(var, delta, width);
}

/*
 * Merge-based CSR SpMV
 */

#define ACG_MERGE_BASED_SPMV_SHARED

__device__ void csrgemv_merge(
    acgidx_t n,
    double * __restrict y,
    const double * __restrict x,
    const acgidx_t * __restrict rowptr,
    const acgidx_t * __restrict colidx,
    const double * __restrict a,
    double alpha,
    acgidx_t nstartrows,
    const acgidx_t * __restrict startrows)
{
    acgidx_t nnzs = rowptr[n];
    acgidx_t ntasks = n+nnzs;
    acgidx_t tid = blockIdx.x*blockDim.x+threadIdx.x;

#ifdef ACG_MERGE_BASED_SPMV_SHARED
    for (acgidx_t l = tid;
         l < ((ntasks/TASKS_PER_THREAD)/blockDim.x)*blockDim.x;
         l += blockDim.x*gridDim.x)
    {
        /* prefetch data to shared memory to improve coalescing */
        extern __shared__ double smem[];
        double * __restrict sa = smem;
        acgidx_t * __restrict srowptr = (acgidx_t *) (&sa[THREADS_PER_BLOCK*TASKS_PER_THREAD]);
        acgidx_t lmin = l-threadIdx.x;
        acgidx_t imin = startrows[lmin];
        acgidx_t kmin = lmin*TASKS_PER_THREAD-imin;
        __syncthreads();
        /* for (int j = 0; j < TASKS_PER_THREAD; j++) { */
        /*     if (kmin+j*THREADS_PER_BLOCK+threadIdx.x < nnzs) { */
        /*         sa[j*THREADS_PER_BLOCK+threadIdx.x] = alpha*a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     } */
        /* } */
        /* for (int j = 0; j < TASKS_PER_THREAD; j++) { */
        /*     if (imin+j*THREADS_PER_BLOCK+threadIdx.x < n) { */
        /*         srowptr[j*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+j*THREADS_PER_BLOCK+threadIdx.x]; */
        /*     } */
        /* } */

        /* if (kmin+15*THREADS_PER_BLOCK+threadIdx.x < nnzs) { */
        /*     sa[ 0*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 1*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 2*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 3*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 4*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 5*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 6*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 7*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 8*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 9*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[10*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+10*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+10*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[11*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+11*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+11*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[12*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+12*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+12*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[13*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+13*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+13*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[14*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+14*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+14*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[15*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+15*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+15*THREADS_PER_BLOCK+threadIdx.x]]; */
        /* } else { */
        /*     for (int j = 0; j < TASKS_PER_THREAD; j++) { */
        /*         if (kmin+j*THREADS_PER_BLOCK+threadIdx.x < nnzs) { */
        /*             sa[j*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*         } */
        /*     } */
        /* } */

        if (kmin+9*THREADS_PER_BLOCK+threadIdx.x < nnzs) {
            sa[ 0*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 1*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 2*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 3*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 4*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 5*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 6*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 7*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 8*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 9*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]];
        } else {
            for (int j = 0; j < TASKS_PER_THREAD; j++) {
                if (kmin+j*THREADS_PER_BLOCK+threadIdx.x < nnzs) {
                    sa[j*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]];
                }
            }
        }

        if (imin+9*THREADS_PER_BLOCK+threadIdx.x < n) {
            srowptr[0*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+0*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[1*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+1*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[2*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+2*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[3*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+3*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[4*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+4*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[5*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+5*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[6*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+6*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[7*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+7*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[8*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+8*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[9*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+9*THREADS_PER_BLOCK+threadIdx.x];
        } else {
            for (int j = 0; j < TASKS_PER_THREAD; j++) {
                if (imin+j*THREADS_PER_BLOCK+threadIdx.x < n) {
                    srowptr[j*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+j*THREADS_PER_BLOCK+threadIdx.x];
                }
            }
        }
        __syncthreads();

        acgidx_t i = startrows[l];
        acgidx_t k = l*TASKS_PER_THREAD-i; /* starting nonzero */
        double sum = 0.0;
        /* #pragma unroll */
        if (k+TASKS_PER_THREAD <= srowptr[i-imin+1]) {
        /* if (k+TASKS_PER_THREAD <= rowptr[i+1]) { */
            for (int l = 0; l < TASKS_PER_THREAD; l++)
                sum += sa[k-kmin+l];
            k+=TASKS_PER_THREAD;
        } else {
            int firstrowtasks = srowptr[i-imin+1] - k;
            /* int firstrowtasks = rowptr[i+1] - k; */
            short j;
            for (j = 0; j < firstrowtasks; j++) {
                sum += sa[k-kmin]; k++;
            }
            unsafeAtomicAdd(&y[i], sum);
            i++, j++, sum = 0.0;
            for (; j < TASKS_PER_THREAD; j++) {
                if (k == srowptr[i-imin+1]) {
                /* if (k == rowptr[i+1]) { */
                    y[i] += sum;
                    i++, sum = 0.0;
                } else {
                    sum += sa[k-kmin];
                    k++;
                }
            }
        }

        if (i < n) unsafeAtomicAdd(&y[i], sum);

#if 0
        /* warp reduction of partial row results */
        long long int rowmask = __match_any_sync(0xFFFFFFFFFFFFFFFF, i);
        short lane = threadIdx.x & (warpSize-1);
        short npeers = __popcll(rowmask);
        short minlane = __ffsll(rowmask)-1;
        short maxlane = minlane + npeers;
        /* #pragma unroll */
        {
            short nextlane = lane+32 < maxlane ? lane+32 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+16 < maxlane ? lane+16 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+8 < maxlane ? lane+8 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+4 < maxlane ? lane+4 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+2 < maxlane ? lane+2 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+1 < maxlane ? lane+1 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        /* for (short offset = warpSize/2; offset > 0; offset /= 2) { */
        /*     short nextlane = lane+offset < maxlane ? lane+offset : warpSize; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        if (lane == minlane) unsafeAtomicAdd(&y[i], sum);
#endif
    }

    /* handle remainder */
    for (acgidx_t l = ((ntasks/TASKS_PER_THREAD)/blockDim.x)*blockDim.x*TASKS_PER_THREAD+tid;
         l < ntasks;
         l += blockDim.x*gridDim.x)
    {
        /* binary search to find starting row for each thread */
        acgidx_t i = 0;
        acgidx_t count = n;
        while (count > 0) {
            acgidx_t row = i;
            acgidx_t step = count >> 1;
            row += step;
            if (rowptr[row+1] <= l-row-1) {
                i = ++row;
                count -= step + 1;
            } else { count = step; }
        }

        /* starting nonzero */
        acgidx_t k = l-i;

        /* a) per-thread atomics */
        if (k < rowptr[i+1]) unsafeAtomicAdd(&y[i], a[k]*x[colidx[k]]);

        /* b) warp reduction of partial row results */
        /* double sum = (k < rowptr[i+1]) ? a[k]*x[colidx[k]] : 0.0; */
        /* int64_t rowmask = __match_any_sync(__activemask(), i); */
        /* int lane = threadIdx.x & (warpSize-1); */
        /* int npeers = __popcll(rowmask); */
        /* int minlane = __ffsll(rowmask)-1; */
        /* int maxlane = minlane + npeers; */
        /* #pragma unroll */
        /* for (int offset = warpSize/2; offset > 0; offset /= 2) { */
        /*     int nextlane = lane+offset < maxlane ? lane+offset : warpSize; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        /* if (lane == minlane) unsafeAtomicAdd(&y[i], sum); */
    }
#else
    for (acgidx_t l = tid;
         l < ntasks/TASKS_PER_THREAD;
         l += blockDim.x*gridDim.x)
    {
        /* /\* binary search to find starting row for each thread *\/ */
        /* acgidx_t i = 0; */
        /* acgidx_t count = n; */
        /* while (count > 0) { */
        /*     acgidx_t row = i; */
        /*     acgidx_t step = count >> 1; */
        /*     row += step; */
        /*     if (rowptr[row+1] <= l*TASKS_PER_THREAD-row-1) { */
        /*         i = ++row; */
        /*         count -= step + 1; */
        /*     } else { count = step; } */
        /* } */

        acgidx_t i = startrows[l];
        acgidx_t k = l*TASKS_PER_THREAD-i; /* starting nonzero */
        double sum = 0.0;
        #pragma unroll
        for (short j = 0; j < TASKS_PER_THREAD; j++) {
            if (k == rowptr[i+1]) {
                unsafeAtomicAdd(&y[i], sum);
                i++, sum = 0.0;
            } else {
                sum += a[k]*x[colidx[k]];
                k++;
            }
        }

        /* warp reduction of partial row results */
        long long int rowmask = __match_any_sync(0xFFFFFFFFFFFFFFFF, i);
        int lane = threadIdx.x & (warpSize-1);
        int npeers = __popcll(rowmask);
        int minlane = __ffsll(rowmask)-1;
        int maxlane = minlane + npeers;
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            int nextlane = lane+offset < maxlane ? lane+offset : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        if (lane == minlane) unsafeAtomicAdd(&y[i], sum);
    }

    /* handle remainder */
    for (acgidx_t l = (ntasks/TASKS_PER_THREAD)*TASKS_PER_THREAD+tid;
         l < ntasks;
         l += blockDim.x*gridDim.x)
    {
        /* binary search to find starting row for each thread */
        acgidx_t i = 0;
        acgidx_t count = n;
        while (count > 0) {
            acgidx_t row = i;
            acgidx_t step = count >> 1;
            row += step;
            if (rowptr[row+1] <= l-row-1) {
                i = ++row;
                count -= step + 1;
            } else { count = step; }
        }

        /* starting nonzero */
        acgidx_t k = l-i;

        /* a) per-thread atomics */
        if (k < rowptr[i+1]) unsafeAtomicAdd(&y[i], a[k]*x[colidx[k]]);

        /* b) warp reduction of partial row results */
        /* double sum = (k < rowptr[i+1]) ? a[k]*x[colidx[k]] : 0.0; */
        /* int64_t rowmask = __match_any_sync(__activemask(), i); */
        /* int lane = threadIdx.x & (warpSize-1); */
        /* int npeers = __popc(rowmask); */
        /* int minlane = __ffsll(rowmask)-1; */
        /* int maxlane = minlane + npeers; */
        /* #pragma unroll */
        /* for (int offset = warpSize/2; offset > 0; offset /= 2) { */
        /*     int nextlane = lane+offset < maxlane ? lane+offset : warpSize; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        /* if (lane == minlane) unsafeAtomicAdd(&y[i], sum); */
    }
#endif
}

__global__ void csrgemv_merge_global(
    acgidx_t n,
    double * __restrict y,
    const double * __restrict x,
    const acgidx_t * __restrict rowptr,
    const acgidx_t * __restrict colidx,
    const double * __restrict a,
    double alpha,
    acgidx_t nstartrows,
    const acgidx_t * __restrict startrows)
{
    acgidx_t nnzs = rowptr[n];
    acgidx_t ntasks = n+nnzs;
    acgidx_t tid = blockIdx.x*blockDim.x+threadIdx.x;

#ifdef ACG_MERGE_BASED_SPMV_SHARED
    for (acgidx_t l = tid;
         l < ((ntasks/TASKS_PER_THREAD)/blockDim.x)*blockDim.x;
         l += blockDim.x*gridDim.x)
    {
        /* prefetch data to shared memory to improve coalescing */
        extern __shared__ double smem[];
        double * __restrict sa = smem;
        acgidx_t * __restrict srowptr = (acgidx_t *) (&sa[THREADS_PER_BLOCK*TASKS_PER_THREAD]);
        acgidx_t lmin = l-threadIdx.x;
        acgidx_t imin = startrows[lmin];
        acgidx_t kmin = lmin*TASKS_PER_THREAD-imin;
        __syncthreads();
        /* for (int j = 0; j < TASKS_PER_THREAD; j++) { */
        /*     if (kmin+j*THREADS_PER_BLOCK+threadIdx.x < nnzs) { */
        /*         sa[j*THREADS_PER_BLOCK+threadIdx.x] = alpha*a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     } */
        /* } */
        /* for (int j = 0; j < TASKS_PER_THREAD; j++) { */
        /*     if (imin+j*THREADS_PER_BLOCK+threadIdx.x < n) { */
        /*         srowptr[j*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+j*THREADS_PER_BLOCK+threadIdx.x]; */
        /*     } */
        /* } */

        /* if (kmin+15*THREADS_PER_BLOCK+threadIdx.x < nnzs) { */
        /*     sa[ 0*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 1*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 2*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 3*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 4*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 5*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 6*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 7*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 8*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[ 9*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[10*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+10*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+10*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[11*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+11*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+11*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[12*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+12*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+12*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[13*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+13*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+13*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[14*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+14*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+14*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*     sa[15*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+15*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+15*THREADS_PER_BLOCK+threadIdx.x]]; */
        /* } else { */
        /*     for (int j = 0; j < TASKS_PER_THREAD; j++) { */
        /*         if (kmin+j*THREADS_PER_BLOCK+threadIdx.x < nnzs) { */
        /*             sa[j*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]]; */
        /*         } */
        /*     } */
        /* } */

        if (kmin+9*THREADS_PER_BLOCK+threadIdx.x < nnzs) {
            sa[ 0*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 0*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 1*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 1*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 2*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 2*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 3*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 3*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 4*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 4*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 5*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 5*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 6*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 6*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 7*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 7*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 8*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 8*THREADS_PER_BLOCK+threadIdx.x]];
            sa[ 9*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+ 9*THREADS_PER_BLOCK+threadIdx.x]];
        } else {
            for (int j = 0; j < TASKS_PER_THREAD; j++) {
                if (kmin+j*THREADS_PER_BLOCK+threadIdx.x < nnzs) {
                    sa[j*THREADS_PER_BLOCK+threadIdx.x] = a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]];
                }
            }
        }

        if (imin+9*THREADS_PER_BLOCK+threadIdx.x < n) {
            srowptr[0*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+0*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[1*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+1*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[2*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+2*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[3*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+3*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[4*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+4*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[5*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+5*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[6*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+6*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[7*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+7*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[8*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+8*THREADS_PER_BLOCK+threadIdx.x];
            srowptr[9*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+9*THREADS_PER_BLOCK+threadIdx.x];
        } else {
            for (int j = 0; j < TASKS_PER_THREAD; j++) {
                if (imin+j*THREADS_PER_BLOCK+threadIdx.x < n) {
                    srowptr[j*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+j*THREADS_PER_BLOCK+threadIdx.x];
                }
            }
        }
        __syncthreads();

        acgidx_t i = startrows[l];
        acgidx_t k = l*TASKS_PER_THREAD-i; /* starting nonzero */
        double sum = 0.0;
        /* #pragma unroll */
        if (k+TASKS_PER_THREAD <= srowptr[i-imin+1]) {
        /* if (k+TASKS_PER_THREAD <= rowptr[i+1]) { */
            for (int l = 0; l < TASKS_PER_THREAD; l++)
                sum += sa[k-kmin+l];
            k+=TASKS_PER_THREAD;
        } else {
            int firstrowtasks = srowptr[i-imin+1] - k;
            /* int firstrowtasks = rowptr[i+1] - k; */
            short j;
            for (j = 0; j < firstrowtasks; j++) {
                sum += sa[k-kmin]; k++;
            }
            unsafeAtomicAdd(&y[i], sum);
            i++, j++, sum = 0.0;
            for (; j < TASKS_PER_THREAD; j++) {
                if (k == srowptr[i-imin+1]) {
                /* if (k == rowptr[i+1]) { */
                    y[i] += sum;
                    i++, sum = 0.0;
                } else {
                    sum += sa[k-kmin];
                    k++;
                }
            }
        }

        if (i < n) unsafeAtomicAdd(&y[i], sum);

#if 0
        /* warp reduction of partial row results */
        long long int rowmask = __match_any_sync(0xFFFFFFFFFFFFFFFF, i);
        short lane = threadIdx.x & (warpSize-1);
        short npeers = __popcll(rowmask);
        short minlane = __ffsll(rowmask)-1;
        short maxlane = minlane + npeers;
        /* #pragma unroll */
        {
            short nextlane = lane+32 < maxlane ? lane+32 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+16 < maxlane ? lane+16 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+8 < maxlane ? lane+8 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+4 < maxlane ? lane+4 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+2 < maxlane ? lane+2 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        {
            short nextlane = lane+1 < maxlane ? lane+1 : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        /* for (short offset = warpSize/2; offset > 0; offset /= 2) { */
        /*     short nextlane = lane+offset < maxlane ? lane+offset : warpSize; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        if (lane == minlane) unsafeAtomicAdd(&y[i], sum);
#endif
    }

    /* handle remainder */
    for (acgidx_t l = ((ntasks/TASKS_PER_THREAD)/blockDim.x)*blockDim.x*TASKS_PER_THREAD+tid;
         l < ntasks;
         l += blockDim.x*gridDim.x)
    {
        /* binary search to find starting row for each thread */
        acgidx_t i = 0;
        acgidx_t count = n;
        while (count > 0) {
            acgidx_t row = i;
            acgidx_t step = count >> 1;
            row += step;
            if (rowptr[row+1] <= l-row-1) {
                i = ++row;
                count -= step + 1;
            } else { count = step; }
        }

        /* starting nonzero */
        acgidx_t k = l-i;

        /* a) per-thread atomics */
        if (k < rowptr[i+1]) unsafeAtomicAdd(&y[i], a[k]*x[colidx[k]]);

        /* b) warp reduction of partial row results */
        /* double sum = (k < rowptr[i+1]) ? a[k]*x[colidx[k]] : 0.0; */
        /* int64_t rowmask = __match_any_sync(__activemask(), i); */
        /* int lane = threadIdx.x & (warpSize-1); */
        /* int npeers = __popcll(rowmask); */
        /* int minlane = __ffsll(rowmask)-1; */
        /* int maxlane = minlane + npeers; */
        /* #pragma unroll */
        /* for (int offset = warpSize/2; offset > 0; offset /= 2) { */
        /*     int nextlane = lane+offset < maxlane ? lane+offset : warpSize; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        /* if (lane == minlane) unsafeAtomicAdd(&y[i], sum); */
    }
#else
    for (acgidx_t l = tid;
         l < ntasks/TASKS_PER_THREAD;
         l += blockDim.x*gridDim.x)
    {
        /* /\* binary search to find starting row for each thread *\/ */
        /* acgidx_t i = 0; */
        /* acgidx_t count = n; */
        /* while (count > 0) { */
        /*     acgidx_t row = i; */
        /*     acgidx_t step = count >> 1; */
        /*     row += step; */
        /*     if (rowptr[row+1] <= l*TASKS_PER_THREAD-row-1) { */
        /*         i = ++row; */
        /*         count -= step + 1; */
        /*     } else { count = step; } */
        /* } */

        acgidx_t i = startrows[l];
        acgidx_t k = l*TASKS_PER_THREAD-i; /* starting nonzero */
        double sum = 0.0;
        #pragma unroll
        for (short j = 0; j < TASKS_PER_THREAD; j++) {
            if (k == rowptr[i+1]) {
                unsafeAtomicAdd(&y[i], sum);
                i++, sum = 0.0;
            } else {
                sum += a[k]*x[colidx[k]];
                k++;
            }
        }

        /* warp reduction of partial row results */
        long long int rowmask = __match_any_sync(0xFFFFFFFFFFFFFFFF, i);
        int lane = threadIdx.x & (warpSize-1);
        int npeers = __popcll(rowmask);
        int minlane = __ffsll(rowmask)-1;
        int maxlane = minlane + npeers;
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            int nextlane = lane+offset < maxlane ? lane+offset : warpSize;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        if (lane == minlane) unsafeAtomicAdd(&y[i], sum);
    }

    /* handle remainder */
    for (acgidx_t l = (ntasks/TASKS_PER_THREAD)*TASKS_PER_THREAD+tid;
         l < ntasks;
         l += blockDim.x*gridDim.x)
    {
        /* binary search to find starting row for each thread */
        acgidx_t i = 0;
        acgidx_t count = n;
        while (count > 0) {
            acgidx_t row = i;
            acgidx_t step = count >> 1;
            row += step;
            if (rowptr[row+1] <= l-row-1) {
                i = ++row;
                count -= step + 1;
            } else { count = step; }
        }

        /* starting nonzero */
        acgidx_t k = l-i;

        /* a) per-thread atomics */
        if (k < rowptr[i+1]) unsafeAtomicAdd(&y[i], a[k]*x[colidx[k]]);

        /* b) warp reduction of partial row results */
        /* double sum = (k < rowptr[i+1]) ? a[k]*x[colidx[k]] : 0.0; */
        /* int64_t rowmask = __match_any_sync(__activemask(), i); */
        /* int lane = threadIdx.x & (warpSize-1); */
        /* int npeers = __popc(rowmask); */
        /* int minlane = __ffsll(rowmask)-1; */
        /* int maxlane = minlane + npeers; */
        /* #pragma unroll */
        /* for (int offset = warpSize/2; offset > 0; offset /= 2) { */
        /*     int nextlane = lane+offset < maxlane ? lane+offset : warpSize; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        /* if (lane == minlane) unsafeAtomicAdd(&y[i], sum); */
    }
#endif
}

int acgsolverhip_csrgemv_merge_startrows(
    acgidx_t n,
    const acgidx_t * __restrict d_rowptr,
    acgidx_t nstartrows,
    acgidx_t * d_startrows)
{
    csrgemv_merge_startrows<<<110,512>>>(n, d_rowptr, nstartrows, d_startrows);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

int acgsolverhip_csrgemv_merge(
    acgidx_t n,
    double * __restrict d_y,
    const double * __restrict d_x,
    const acgidx_t * __restrict d_rowptr,
    const acgidx_t * __restrict d_colidx,
    const double * __restrict d_a,
    double alpha,
    acgidx_t nstartrows,
    const acgidx_t * __restrict d_startrows)
{
    csrgemv_merge_global<<<110,512>>>(
        n, d_y, d_x, d_rowptr, d_colidx, d_a, alpha, nstartrows, d_startrows);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    return ACG_SUCCESS;
}

__device__ void csrgemv(
    acgidx_t n,
    double * __restrict y,
    const double * __restrict x,
    const acgidx_t * __restrict rowptr,
    const acgidx_t * __restrict colidx,
    const double * __restrict a,
    double alpha)
{
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        double yi = 0;
        for (acgidx_t k = rowptr[i]; k < rowptr[i+1]; k++)
            yi += a[k]*x[colidx[k]];
        y[i] += alpha*yi;
    }
}

__inline__ __device__ double warpReduceSum(int64_t mask, double x)
{
    for (int i = warpSize/2; i > 0; i /= 2)
        x += __shfl_down_sync(mask, x, i);
    return x;
}

__device__ void dzero(
    acgidx_t n,
    double * x)
{
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        x[i] = 0.0;
    }
}

__device__ void dcopy(
    acgidx_t n,
    double * __restrict y,
    const double * __restrict x)
{
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = x[i];
    }
}

__device__ void ddot(
    acgidx_t n,
    const double * __restrict x,
    const double * __restrict y,
    double * __restrict dot)
{
    extern __shared__ double w[];

    /* compute per-thread partial results */
    double z = 0;
#if 1
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        z += x[i]*y[i];
    }
#elif 0
    acgidx_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    for (acgidx_t i = 2*tid;
         i < (n & ~1);
         i += 2*blockDim.x*gridDim.x)
    {
        z += x[i+0]*y[i+0] + x[i+1]*y[i+1];
    }
    if (tid < (n%2)) z += x[n-tid-1]*y[n-tid-1];
#elif 0
    acgidx_t tid = blockIdx.x*blockDim.x+threadIdx.x;
    for (acgidx_t i = 4*tid;
         i < (n & ~3);
         i += 4*blockDim.x*gridDim.x)
    {
        z += x[i+0]*y[i+0] + x[i+1]*y[i+1] + x[i+2]*y[i+2] + x[i+3]*y[i+3];
    }
    if (tid < (n%4)) z += x[n-tid-1]*y[n-tid-1];
#endif

#if 0
    /* perform reduction across threads in a warp */
    z = warpReduceSum(0xffffffffffffffff, z);

    /* a) use atomics to perform reduction across warps in the grid */
    if ((threadIdx.x & (warpSize-1)) == 0)
        unsafeAtomicAdd(dot, z);
#else
    /* perform reduction across threads in a warp */
    z = warpReduceSum(0xffffffffffffffff, z);
    if ((threadIdx.x & (warpSize-1)) == 0) w[threadIdx.x>>6] = z;
    __syncthreads();
    if (threadIdx.x < 4) w[threadIdx.x] += w[threadIdx.x+4];
    __syncthreads();
    if (threadIdx.x < 2) w[threadIdx.x] += w[threadIdx.x+2];
    __syncthreads();
    if (threadIdx.x < 1) w[threadIdx.x] += w[threadIdx.x+1];
    __syncthreads();
    if (threadIdx.x == 0) unsafeAtomicAdd(dot, w[0]);
#endif
}

__device__ void daxpy(
    acgidx_t n,
    double alpha,
    const double * __restrict x,
    double * __restrict y)
{
    #pragma unroll
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = alpha*x[i]+y[i];
        // __builtin_nontemporal_store(alpha*x[i]+y[i], &y[i]);
    }
}

__device__ void daypx(
    acgidx_t n,
    double alpha,
    double * __restrict y,
    const double * __restrict x)
{
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = alpha*y[i]+x[i];
        // __builtin_nontemporal_store(alpha*y[i]+x[i], &y[i]);
    }
}

__device__ void pack_double(
    int sendbufsize,
    double * __restrict sendbuf,
    int srcbufsize,
    const double * __restrict srcbuf,
    const int * __restrict srcbufidx)
{
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < sendbufsize;
         i += blockDim.x*gridDim.x)
    {
        sendbuf[i] = srcbuf[srcbufidx[i]];
    }
}

__device__ void pack_double_block(
    int sendbufsize,
    double * __restrict sendbuf,
    int srcbufsize,
    const double * __restrict srcbuf,
    const int * __restrict srcbufidx)
{
    for (int i = threadIdx.x;
         i < sendbufsize;
         i += blockDim.x)
    {
        sendbuf[i] = srcbuf[srcbufidx[i]];
    }
}

__device__ void unpack_double(
    int recvbufsize,
    const double * __restrict recvbuf,
    int dstbufsize,
    double * __restrict dstbuf,
    const int * __restrict dstbufidx)
{
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < recvbufsize;
         i += blockDim.x*gridDim.x)
    {
        dstbuf[dstbufidx[i]] = recvbuf[i];
    }
}

__device__ void unpack_double_block(
    int recvbufsize,
    const double * __restrict recvbuf,
    int dstbufsize,
    double * __restrict dstbuf,
    const int * __restrict dstbufidx)
{
    for (int i = threadIdx.x;
         i < recvbufsize;
         i += blockDim.x)
    {
        dstbuf[dstbufidx[i]] = recvbuf[i];
    }
}

#define ACG_MERGE_BASED_SPMV
#define ACG_NVSHMEM_NOSENDRECV
// #define ACG_NVSHMEM_SENDRECV_SINGLE_BLOCK
#define ACG_NVSHMEM_NOALLREDUCE
// #define ACG_NVSHMEM_ALLREDUCE_BLOCK
// #define ACG_NVSHMEM_ALLREDUCE_WARP

__global__ void __launch_bounds__(THREADS_PER_BLOCK) acgsolverhip_cg_kernel(
    int n,
    int nghosts,
    int nborderrows,
    int borderrowoffset,
    const acgidx_t * __restrict rowptr,
    const acgidx_t * __restrict colidx,
    const double * __restrict a,
    const acgidx_t * __restrict orowptr,
    const acgidx_t * __restrict ocolidx,
    const double * __restrict oa,
    int sendsize,
    double * __restrict sendbuf,
    const int * __restrict sendbufidx,
    int nrecipients,
    const int * __restrict recipients,
    const int * __restrict sendcounts,
    const int * __restrict sdispls,
    const int * __restrict putdispls,
    const int * __restrict putranks,
    const int * __restrict getranks,
    uint64_t * __restrict received,
    uint64_t * __restrict readytoreceive,
    int recvsize,
    double * __restrict recvbuf,
    const int * __restrict recvbufidx,
    int nsenders,
    const int * __restrict senders,
    const int * __restrict recvcounts,
    const int * __restrict rdispls,
    const double * __restrict b,
    double * __restrict x,
    double * __restrict r,
    double * __restrict p,
    double * __restrict t,
    double * __restrict bnrm2sqr,
    double * __restrict r0nrm2sqr,
    double * __restrict rnrm2sqr,
    double * __restrict pdott,
    int * __restrict niterations,
    int * __restrict converged,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    acgidx_t nstartrows,
    acgidx_t * __restrict startrows)
{
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    /* int npes = nvshmem_n_pes(); */

    /* reset some variables */
    if (grid.thread_rank() == 0) {
        *bnrm2sqr = *rnrm2sqr = *r0nrm2sqr = *pdott = 0.0;
        *niterations = *converged = 0;
    }
    cg::sync(grid);

    /* compute right-hand side norm */
    ddot(n-nghosts, b, b, bnrm2sqr);
#ifndef ACG_NVSHMEM_NOALLREDUCE
    if (npes > 1) {
        cg::sync(grid);
#if defined(ACG_NVSHMEM_ALLREDUCE_BLOCK)
        if (grid.block_rank() == 0)
            nvshmemx_double_sum_reduce_block(
                NVSHMEM_TEAM_WORLD, bnrm2sqr, bnrm2sqr, 1);
#elif defined(ACG_NVSHMEM_ALLREDUCE_WARP)
        if (grid.block_rank() == 0 && block.thread_rank() < warpSize)
            nvshmemx_double_sum_reduce_warp(
                NVSHMEM_TEAM_WORLD, bnrm2sqr, bnrm2sqr, 1);
#else
        if (grid.thread_rank() == 0)
            nvshmem_double_sum_reduce(
                NVSHMEM_TEAM_WORLD, bnrm2sqr, bnrm2sqr, 1);
#endif
    }
#endif

    /* copy r₀ <- b */
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        r[i] = -b[i];
    }
    /* dcopy(n-nghosts, r, b); */
    cg::sync(grid);

    /* start halo exchange to update ghost entries of x */
#ifndef ACG_NVSHMEM_NOSENDRECV
    if (npes > 1) {
#if defined(ACG_NVSHMEM_SENDRECV_SINGLE_BLOCK)
        pack_double(sendsize, sendbuf, n, x, sendbufidx);
        if (grid.thread_rank() == 0) {
            for (int q = 0; q < nrecipients; q++) {
                nvshmem_signal_wait_until(&readytoreceive[q], NVSHMEM_CMP_EQ, 0);
                readytoreceive[q] = 1;
            }
        }
        cg::sync(grid);
        if (grid.block_rank() == 0) {
            for (int q = 0; q < nrecipients; q++) {
                const double * sendbufq = sendbuf + sdispls[q];
                double * recvbufq = recvbuf + putdispls[q];
                nvshmemx_double_put_signal_nbi_block(
                    recvbufq, sendbufq, sendcounts[q],
                    &received[putranks[q]], 1, NVSHMEM_SIGNAL_SET, recipients[q]);
            }
        }
#else
        for (int q = grid.block_rank(); q < nrecipients; q += grid.num_blocks()) {
            double * sendbufq = sendbuf + sdispls[q];
            pack_double_block(sendcounts[q], sendbufq, n, x, sendbufidx+sdispls[q]);
            if (block.thread_rank() == 0) {
                nvshmem_signal_wait_until(&readytoreceive[q], NVSHMEM_CMP_EQ, 0);
                readytoreceive[q] = 1;
            }
            cg::sync(block);
            double * recvbufq = recvbuf + putdispls[q];
            nvshmemx_double_put_signal_nbi_block(
                recvbufq, sendbufq, sendcounts[q],
                &received[putranks[q]], 1, NVSHMEM_SIGNAL_SET, recipients[q]);
        }
#endif
    }
#endif

    /* compute the initial residual, r₀ = b-A*x₀ */
#ifdef ACG_MERGE_BASED_SPMV
    csrgemv_merge(n-nghosts, r, x, rowptr, colidx, a, 1.0, nstartrows, startrows);
#else
    csrgemv(n-nghosts, r, x, rowptr, colidx, a, 1.0);
#endif

    /* wait for halo exchange to finish before multiplying
     * off-diagonal part */
#if 0
    if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
        cg::sync(grid);
        if (grid.thread_rank() == 0) nvshmem_quiet();
        cg::sync(grid);
        for (int q = grid.block_rank(); q < nsenders; q += grid.num_blocks()) {
            if (block.thread_rank() == 0) {
                nvshmem_signal_wait_until(&received[q], NVSHMEM_CMP_EQ, 1);
                received[q] = 0;
            }
            cg::sync(block);
            unpack_double_block(recvcounts[q], recvbuf+rdispls[q], n, x, recvbufidx+rdispls[q]);
            cg::sync(block);
            if (block.thread_rank() == 0) {
                nvshmemx_signal_op(&readytoreceive[getranks[q]], 0, NVSHMEM_SIGNAL_SET, senders[q]);
            }
        }
#endif
        cg::sync(grid);
        csrgemv(nborderrows, r+borderrowoffset, x+borderrowoffset, orowptr, ocolidx, oa, -1.0);
    }
#endif
    cg::sync(grid);

    /* compute initial residual norm */
    ddot(n-nghosts, r, r, rnrm2sqr);
    cg::sync(grid);
#ifndef ACG_NVSHMEM_NOALLREDUCE
    if (npes > 1) {
#if defined(ACG_NVSHMEM_ALLREDUCE_BLOCK)
        if (grid.block_rank() == 0)
            nvshmemx_double_sum_reduce_block(
                NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 1);
#elif defined(ACG_NVSHMEM_ALLREDUCE_WARP)
        if (grid.block_rank() == 0 && block.thread_rank() < warpSize)
            nvshmemx_double_sum_reduce_warp(
                NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 1);
#else
        if (grid.thread_rank() == 0)
            nvshmem_double_sum_reduce(
                NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 1);
#endif
        cg::sync(grid);
    }
#endif
    double rnrm2 = sqrt(*rnrm2sqr);
    if (grid.thread_rank() == 0) *r0nrm2sqr = *rnrm2sqr;
    residualrtol *= rnrm2;

    /* initial search direction p = r₀ */
    dcopy(n-nghosts, p, r);
    cg::sync(grid);

    /* initial convergence test */
    if ((residualatol > 0 && rnrm2 < residualatol) ||
        (residualrtol > 0 && rnrm2 < residualrtol))
    {
        if (grid.thread_rank() == 0) *converged = true;
        return;
    }

    /* iterative solver loop */
    for (int k = 0; k < maxits; k++) {

        /* set t to zero before computing t = Ap */
        dzero(n-nghosts, t);

        /* reset scalar values and wait for p to be updated */
        double alpha = *rnrm2sqr, beta = *rnrm2sqr;
        cg::sync(grid);
        if (grid.thread_rank() == 0) *pdott = *rnrm2sqr = 0;
        cg::sync(grid);

        /* start halo exchange to update ghost entries of p */
#ifndef ACG_NVSHMEM_NOSENDRECV
        if (npes > 1) {
#if defined (ACG_NVSHMEM_SENDRECV_SINGLE_BLOCK)
            pack_double(sendsize, sendbuf, n, p, sendbufidx);
            if (grid.thread_rank() == 0) {
                for (int q = 0; q < nrecipients; q++) {
                    nvshmem_signal_wait_until(&readytoreceive[q], NVSHMEM_CMP_EQ, 0);
                    readytoreceive[q] = 1;
                }
            }
            cg::sync(grid);
            if (grid.block_rank() == 0) {
                for (int q = 0; q < nrecipients; q++) {
                    const double * sendbufq = sendbuf + sdispls[q];
                    double * recvbufq = recvbuf + putdispls[q];
                    nvshmemx_double_put_signal_nbi_block(
                        recvbufq, sendbufq, sendcounts[q],
                        &received[putranks[q]], 1, NVSHMEM_SIGNAL_SET, recipients[q]);
                }
            }
#else
            for (int q = grid.block_rank(); q < nrecipients; q += grid.num_blocks()) {
                double * sendbufq = sendbuf + sdispls[q];
                pack_double_block(sendcounts[q], sendbufq, n, p, sendbufidx+sdispls[q]);
                if (block.thread_rank() == 0) {
                    nvshmem_signal_wait_until(&readytoreceive[q], NVSHMEM_CMP_EQ, 0);
                    readytoreceive[q] = 1;
                }
                cg::sync(block);
                double * recvbufq = recvbuf + putdispls[q];
                nvshmemx_double_put_signal_nbi_block(
                    recvbufq, sendbufq, sendcounts[q],
                    &received[putranks[q]], 1, NVSHMEM_SIGNAL_SET, recipients[q]);
            }
#endif
        }
#endif

        /* compute t = Ap (local part) */
#ifdef ACG_MERGE_BASED_SPMV
        csrgemv_merge(n-nghosts, t, p, rowptr, colidx, a, 1.0, nstartrows, startrows);
#else
        csrgemv(n-nghosts, t, p, rowptr, colidx, a, 1.0);
#endif

        /* wait for halo exchange to finish and compute t = Ap (remote part) */
#if 0
        if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
            cg::sync(grid);
            if (grid.thread_rank() == 0) nvshmem_quiet();
            cg::sync(grid);
            for (int q = grid.block_rank(); q < nsenders; q += grid.num_blocks()) {
                if (block.thread_rank() == 0) {
                    nvshmem_signal_wait_until(&received[q], NVSHMEM_CMP_EQ, 1);
                    received[q] = 0;
                }
                cg::sync(block);
                unpack_double_block(recvcounts[q], recvbuf+rdispls[q], n, p, recvbufidx+rdispls[q]);
                cg::sync(block);
                if (block.thread_rank() == 0) {
                    nvshmemx_signal_op(&readytoreceive[getranks[q]], 0, NVSHMEM_SIGNAL_SET, senders[q]);
                }
            }
#endif
            cg::sync(grid);
            csrgemv(nborderrows, t+borderrowoffset, p+borderrowoffset, orowptr, ocolidx, oa, 1.0);
        }
#endif
        cg::sync(grid);

        /* compute (p,t) */
        ddot(n-nghosts, p, t, pdott);
        cg::sync(grid);
#ifndef ACG_NVSHMEM_NOALLREDUCE
        if (npes > 1) {
#if defined(ACG_NVSHMEM_ALLREDUCE_BLOCK)
            if (grid.block_rank() == 0)
                nvshmemx_double_sum_reduce_block(
                    NVSHMEM_TEAM_WORLD, pdott, pdott, 1);
#elif defined(ACG_NVSHMEM_ALLREDUCE_WARP)
            if (grid.block_rank() == 0 && block.thread_rank() < warpSize)
                nvshmemx_double_sum_reduce_warp(
                    NVSHMEM_TEAM_WORLD, pdott, pdott, 1);
#else
            if (grid.thread_rank() == 0)
                nvshmem_double_sum_reduce(
                    NVSHMEM_TEAM_WORLD, pdott, pdott, 1);
#endif
            cg::sync(grid);
        }
#endif

        /* compute α = (r,r) / (p,t) */
        alpha = alpha / *pdott;

        /* update solution, x = αp + x */
        daxpy(n-nghosts, alpha, p, x);

        /* update residual, r = -αt + r */
        daxpy(n-nghosts, -alpha, t, r);

        /* compute residual norm */
        ddot(n-nghosts, r, r, rnrm2sqr);
        cg::sync(grid);
#ifndef ACG_NVSHMEM_NOALLREDUCE
        if (npes > 1) {
#if defined(ACG_NVSHMEM_ALLREDUCE_BLOCK)
            if (grid.block_rank() == 0)
                nvshmemx_double_sum_reduce_block(
                    NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 1);
#elif defined(ACG_NVSHMEM_ALLREDUCE_WARP)
            if (grid.block_rank() == 0 && block.thread_rank() < warpSize)
                nvshmemx_double_sum_reduce_warp(
                    NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 1);
#else
            if (grid.thread_rank() == 0)
                nvshmem_double_sum_reduce(
                    NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 1);
#endif
            cg::sync(grid);
        }
#endif

        /* convergence tests */
        rnrm2 = sqrt(*rnrm2sqr);
        if ((residualatol > 0 && rnrm2 < residualatol) ||
            (residualrtol > 0 && rnrm2 < residualrtol))
        {
            if (grid.thread_rank() == 0) {
                *niterations = k+1;
                *converged = true;
            }
            return;
        }

        /* compute β = (rₖ₊₁,rₖ₊₁)/(rₖ,rₖ) */
        beta = *rnrm2sqr / beta;

        /* update search direction, p = βp + r */
        daypx(n-nghosts, beta, p, r);
    }

    if (grid.thread_rank() == 0) {
        *niterations = maxits;
        *converged = false;
    }
}

/**
 * ‘acgsolverhip_solve_device()’ solves the given linear system,
 * Ax=b, using the conjugate gradient method. The linear system may be
 * distributed across multiple processes and communication is handled
 * using device-initiated rocSHMEM.
 *
 * The solver must already have been configured with ‘acgsolverhip_init()’
 * for a linear system Ax=b, and the dimensions of the vectors b and x
 * must match the number of columns and rows of A, respectively.
 *
 * The stopping criterion are:
 *
 *  - ‘maxits’, the maximum number of iterations to perform
 *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ < γₐ
 *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖ < γᵣ
 *  - ‘residualatol’, an absolute tolerance for the residual, ‖b-Ax‖ < εₐ
 *  - ‘residualrtol’, a relative tolerance for the residual, ‖b-Ax‖/‖b-Ax₀‖ < εᵣ
 *
 * The iterative solver converges if
 *
 *   ‖δx‖ < γₐ, ‖δx‖ < γᵣ‖x₀‖, ‖b-Ax‖ < εₐ or ‖b-Ax‖ < εᵣ‖b-Ax₀‖.
 *
 * To skip the convergence test for any one of the above stopping
 * criterion, the associated tolerance may be set to zero.
 */
int acgsolverhip_solve_device(
    struct acgsolverhip * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm * comm,
    int * errcode)
{
/* #if !defined(ACG_HAVE_ROCSHMEM) */
/*     return ACG_ERR_ROCSHMEM_NOT_SUPPORTED; */
/* #else */

    /* set initial state */
    cg->nsolves++; cg->niterations = 0;
    cg->bnrm2 = INFINITY;
    cg->r0nrm2 = cg->rnrm2 = INFINITY;
    cg->x0nrm2 = cg->dxnrm2 = INFINITY;
    cg->maxits = maxits;
    cg->diffatol = diffatol;
    cg->diffrtol = diffrtol;
    cg->residualatol = residualatol;
    cg->residualrtol = residualrtol;

    int err;
    if (b->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->r.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->p.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->t.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    int n = A->nprows;
    if (n != b->num_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (n != x->num_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (n != cg->r.num_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (n != cg->p.num_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (n != cg->t.num_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    int nghosts = A->nghostrows;
    if (nghosts != b->num_ghost_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (nghosts != x->num_ghost_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (nghosts != cg->r.num_ghost_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (nghosts != cg->p.num_ghost_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (nghosts != cg->t.num_ghost_nonzeros) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    int nborderrows = A->nborderrows;
    int borderrowoffset = A->borderrowoffset;

    /* not implemented */
    if (diffatol > 0 || diffrtol > 0) return ACG_ERR_NOT_SUPPORTED;

    /* only a single GPU is supported for now */
    int commsize, rank;
    acgcomm_size(comm, &commsize);
    acgcomm_rank(comm, &rank);
    if (commsize > 1) return ACG_ERR_NOT_SUPPORTED;
    /* if (comm->type != acgcomm_nvshmem) return ACG_ERR_NOT_SUPPORTED; */

    hipStream_t stream = 0;
    const struct acghalo * halo = cg->halo;
    double * d_bnrm2sqr = cg->d_bnrm2sqr;
    double * d_r0nrm2sqr = cg->d_r0nrm2sqr;
    double * d_rnrm2sqr = cg->d_rnrm2sqr;
    /* double * d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev; */
    double * d_pdott = cg->d_pdott;
    int * d_niterations = cg->d_niterations;
    int * d_converged = cg->d_converged;
    /* double * d_one = cg->d_one; */
    /* double * d_minus_one = cg->d_minus_one; */
    /* double * d_zero = cg->d_zero; */
    double * d_r = cg->d_r;
    double * d_p = cg->d_p;
    double * d_t = cg->d_t;
    acgidx_t * d_rowptr = cg->d_rowptr;
    acgidx_t * d_colidx = cg->d_colidx;
    double * d_a = cg->d_a;
    acgidx_t * d_orowptr = cg->d_orowptr;
    acgidx_t * d_ocolidx = cg->d_ocolidx;
    double * d_oa = cg->d_oa;

    /* prepare for halo exchange */
    struct acghaloexchange haloexchange;
    err = acghaloexchange_init_hip(
        &haloexchange, cg->halo, ACG_DOUBLE, ACG_DOUBLE, comm, stream);
    if (err) return err;

    /* copy right-hand side and initial guess to device */
    double * d_b;
    err = hipMalloc((void **) &d_b, b->num_nonzeros*sizeof(*d_b));
    if (err) return ACG_ERR_HIP;
    err = hipMemcpy(d_b, b->x, b->num_nonzeros*sizeof(*d_b), hipMemcpyHostToDevice);
    if (err) return ACG_ERR_HIP;
    double * d_x;
    err = hipMalloc((void **) &d_x, x->num_nonzeros*sizeof(*d_x));
    if (err) return ACG_ERR_HIP;
    err = hipMemcpy(d_x, x->x, x->num_nonzeros*sizeof(*d_x), hipMemcpyHostToDevice);
    if (err) return ACG_ERR_HIP;

    /* enable maximum amount of shared memory for merge-based spmv */
    int sharedmemsize = THREADS_PER_BLOCK*TASKS_PER_THREAD*(sizeof(double)+sizeof(acgidx_t));
    /* int sharedmemsize = THREADS_PER_BLOCK*TASKS_PER_THREAD*(sizeof(double)); */
    /* err = hipFuncSetAttribute( */
    /*     acgsolverhip_cg_kernel, */
    /*     hipFuncAttributePreferredSharedMemoryCarveout, */
    /*     hipSharedmemCarveoutMaxShared); */
    /* if (err) return ACG_ERR_HIP; */
#if 0
    /* This is ignored on AMD devices */
    err = hipFuncSetAttribute(
        acgsolverhip_cg_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        sharedmemsize);
    if (err) return ACG_ERR_HIP;
#endif

    /* determine grid and thread block size */
    int mingridsize = 0, blocksize = 0;
    hipOccupancyMaxPotentialBlockSize(
        &mingridsize, &blocksize, acgsolverhip_cg_kernel,
        sharedmemsize, 0);
    int device;
    if (blocksize > THREADS_PER_BLOCK) blocksize = THREADS_PER_BLOCK;
    hipGetDevice(&device);
    hipDeviceProp_t deviceprop;
    hipGetDeviceProperties(&deviceprop, device);
    int nmultiprocessors = deviceprop.multiProcessorCount;
    int nblockspersm = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &nblockspersm, acgsolverhip_cg_kernel, blocksize,
        sharedmemsize);
    int nblocks = nmultiprocessors*nblockspersm;
    dim3 blockDim(blocksize, 1, 1);
    dim3 gridDim(nblocks, 1, 1);

    fprintf(stderr, "\n%s: rank=%d nmultiprocessors=%d nblockspersm=%d blockDim=(%d,%d,%d) gridDim=(%d,%d,%d) n=%d nghosts=%d\n",
            __FUNCTION__, rank, nmultiprocessors, nblockspersm, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, n, nghosts);

    acgidx_t ntasks = (A->nprows-A->nghostrows)+A->fnpnzs;
    acgidx_t nstartrows = (ntasks+TASKS_PER_THREAD-1)/TASKS_PER_THREAD;
    acgidx_t * d_startrows;
    err = hipMalloc((void **) &d_startrows, nstartrows*sizeof(*d_startrows));
    if (err) return ACG_ERR_HIP;
    csrgemv_merge_startrows<<<gridDim.x,blockDim.x>>>(
        n-nghosts, d_rowptr, nstartrows, d_startrows);
    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    hipStreamSynchronize(stream);

    /* warmup */
    if (warmup > 0) {
        double zero = 0.0;
        void * kernelargs[] = {
            (void *) &n,
            (void *) &nghosts,
            (void *) &nborderrows,
            (void *) &borderrowoffset,
            (void *) &d_rowptr,
            (void *) &d_colidx,
            (void *) &d_a,
            (void *) &d_orowptr,
            (void *) &d_ocolidx,
            (void *) &d_oa,
            (void *) &halo->sendsize,
            (void *) &haloexchange.d_sendbuf,
            (void *) &haloexchange.d_sendbufidx,
            (void *) &halo->nrecipients,
            (void *) &haloexchange.d_recipients,
            (void *) &haloexchange.d_sendcounts,
            (void *) &haloexchange.d_sdispls,
            (void *) &haloexchange.d_putdispls,
            (void *) &haloexchange.d_putranks,
            (void *) &haloexchange.d_getranks,
            (void *) &haloexchange.d_received,
            (void *) &haloexchange.d_readytoreceive,
            (void *) &halo->recvsize,
            (void *) &haloexchange.d_recvbuf,
            (void *) &haloexchange.d_recvbufidx,
            (void *) &halo->nsenders,
            (void *) &haloexchange.d_senders,
            (void *) &haloexchange.d_recvcounts,
            (void *) &haloexchange.d_rdispls,
            (void *) &d_b,
            (void *) &d_x,
            (void *) &d_r,
            (void *) &d_p,
            (void *) &d_t,
            (void *) &d_bnrm2sqr,
            (void *) &d_r0nrm2sqr,
            (void *) &d_rnrm2sqr,
            (void *) &d_pdott,
            (void *) &d_niterations,
            (void *) &d_converged,
            (void *) &warmup,
            (void *) &zero /* diffatol */,
            (void *) &zero /* diffrtol */,
            (void *) &zero /* residualatol */,
            (void *) &zero /* residualrtol */,
            (void *) &nstartrows,
            (void *) &d_startrows };

        err = hipLaunchCooperativeKernel(
            (void *) acgsolverhip_cg_kernel, gridDim, blockDim,
            kernelargs, sharedmemsize, stream);

        /* err = nvshmemx_collective_launch( */
        /*     (void *) acgsolverhip_cg_kernel, gridDim, blockDim, */
        /*     kernelargs, sharedmemsize, stream); */

         if (err) { if (errcode) *errcode = err; return ACG_ERR_HIP; }
         if (hipPeekAtLastError()) return ACG_ERR_HIP;
         hipStreamSynchronize(stream);
         err = hipMemcpy(d_x, x->x, x->num_nonzeros*sizeof(*d_x), hipMemcpyHostToDevice);
         if (err) return ACG_ERR_HIP;
    }

    int converged = 0;
    acgtime_t t0, t1;
    err = acgcomm_barrier_hip(stream, comm, errcode);
    if (err) return err;
    hipStreamSynchronize(stream);
    MPI_Barrier(comm->mpicomm);
    hipStreamSynchronize(stream);
    gettime(&t0);

    /* launch device-side CG kernel */
    void * kernelargs[] = {
        (void *) &n,
        (void *) &nghosts,
        (void *) &nborderrows,
        (void *) &borderrowoffset,
        (void *) &d_rowptr,
        (void *) &d_colidx,
        (void *) &d_a,
        (void *) &d_orowptr,
        (void *) &d_ocolidx,
        (void *) &d_oa,
        (void *) &halo->sendsize,
        (void *) &haloexchange.d_sendbuf,
        (void *) &haloexchange.d_sendbufidx,
        (void *) &halo->nrecipients,
        (void *) &haloexchange.d_recipients,
        (void *) &haloexchange.d_sendcounts,
        (void *) &haloexchange.d_sdispls,
        (void *) &haloexchange.d_putdispls,
        (void *) &haloexchange.d_putranks,
        (void *) &haloexchange.d_getranks,
        (void *) &haloexchange.d_received,
        (void *) &haloexchange.d_readytoreceive,
        (void *) &halo->recvsize,
        (void *) &haloexchange.d_recvbuf,
        (void *) &haloexchange.d_recvbufidx,
        (void *) &halo->nsenders,
        (void *) &haloexchange.d_senders,
        (void *) &haloexchange.d_recvcounts,
        (void *) &haloexchange.d_rdispls,
        (void *) &d_b,
        (void *) &d_x,
        (void *) &d_r,
        (void *) &d_p,
        (void *) &d_t,
        (void *) &d_bnrm2sqr,
        (void *) &d_r0nrm2sqr,
        (void *) &d_rnrm2sqr,
        (void *) &d_pdott,
        (void *) &d_niterations,
        (void *) &d_converged,
        (void *) &maxits,
        (void *) &diffatol,
        (void *) &diffrtol,
        (void *) &residualatol,
        (void *) &residualrtol,
        (void *) &nstartrows,
        (void *) &d_startrows };

    err = hipLaunchCooperativeKernel(
        (void *) acgsolverhip_cg_kernel, gridDim, blockDim,
        kernelargs, sharedmemsize, stream);

    /* err = nvshmemx_collective_launch( */
    /*     (void *) acgsolverhip_cg_kernel, gridDim, blockDim, */
    /*     kernelargs, sharedmemsize, stream); */
    if (err) { if (errcode) *errcode = err; return ACG_ERR_HIP; }

    if (hipPeekAtLastError()) return ACG_ERR_HIP;
    hipStreamSynchronize(stream);
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

    /* copy solution back to host */
    err = hipMemcpy(x->x, d_x, x->num_nonzeros*sizeof(*d_x), hipMemcpyDeviceToHost);
    if (err) return ACG_ERR_HIP;

    /* free vectors */
    hipFree(d_startrows);
    hipFree(d_x); hipFree(d_b);
    acghaloexchange_free(&haloexchange);

    /* check for HIP errors */
    if (hipGetLastError() != hipSuccess)
        return ACG_ERR_HIP;

    /* copy results from device to host */
    err = hipMemcpy(&cg->bnrm2, d_bnrm2sqr, sizeof(cg->bnrm2), hipMemcpyDeviceToHost);
    if (err) return ACG_ERR_HIP;
    cg->bnrm2 = sqrt(cg->bnrm2);
    err = hipMemcpy(&cg->r0nrm2, d_r0nrm2sqr, sizeof(cg->r0nrm2), hipMemcpyDeviceToHost);
    if (err) return ACG_ERR_HIP;
    cg->r0nrm2 = sqrt(cg->r0nrm2);
    err = hipMemcpy(&cg->rnrm2, d_rnrm2sqr, sizeof(cg->rnrm2), hipMemcpyDeviceToHost);
    if (err) return ACG_ERR_HIP;
    cg->rnrm2 = sqrt(cg->rnrm2);
    err = hipMemcpy(&cg->niterations, d_niterations, sizeof(cg->niterations), hipMemcpyDeviceToHost);
    if (err) return ACG_ERR_HIP;
    cg->ntotaliterations += cg->niterations;
    err = hipMemcpy(&converged, d_converged, sizeof(converged), hipMemcpyDeviceToHost);
    if (err) return ACG_ERR_HIP;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged) return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
/* #endif */
}
