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
 * Authors: James D. Trotter <james@simula.no>
 *
 * Last modified: 2025-04-26
 *
 * CUDA kernels for CG solvers
 */

#include "acg/config.h"
#include "acg/cg-kernels-cuda.h"
#include "acg/cgcuda.h"
#include "acg/comm.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/error.h"
#include "acg/time.h"
#include "acg/vector.h"

#if defined(ACG_HAVE_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

__constant__ double minus_one;
__constant__ double one;
__constant__ double zero;

int acgsolvercuda_init_constants(
    double ** d_minus_one,
    double ** d_one,
    double ** d_zero)
{
    cudaError_t err;
    double h_minus_one = -1.0, h_one = 1.0, h_zero = 0.0;
    err = cudaMemcpyToSymbol(minus_one, &h_minus_one, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaGetSymbolAddress((void **) d_minus_one, minus_one);
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpyToSymbol(one, &h_one, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaGetSymbolAddress((void **) d_one, one);
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpyToSymbol(zero, &h_zero, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaGetSymbolAddress((void **) d_zero, zero);
    if (err) return ACG_ERR_CUDA;
    return ACG_SUCCESS;
}

__global__ void acgsolvercuda_alpha_kernel(
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

int acgsolvercuda_alpha(
    double * d_alpha,
    double * d_minus_alpha,
    const double * d_rnrm2sqr,
    const double * d_pdott)
{
    acgsolvercuda_alpha_kernel<<<1,1>>>(d_alpha, d_minus_alpha, d_rnrm2sqr, d_pdott);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    return ACG_SUCCESS;
}

__global__ void acgsolvercuda_beta_kernel(
    double * beta,
    const double * rnrm2sqr,
    const double * rnrm2sqr_prev)
{
    if (threadIdx.x == 0) *beta = *rnrm2sqr / *rnrm2sqr_prev;
}

int acgsolvercuda_beta(
    double * d_beta,
    const double * d_rnrm2sqr,
    const double * d_rnrm2sqr_prev)
{
    acgsolvercuda_beta_kernel<<<1,1,0>>>(d_beta, d_rnrm2sqr, d_rnrm2sqr_prev);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    return ACG_SUCCESS;
}

__global__ void acgsolvercuda_daxpy_alpha_kernel(
    int n,
    const double * rnrm2sqr,
    const double * pdott,
    const double * x,
    double * y)
{
    double a = (*rnrm2sqr) / (*pdott);
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = a*x[i]+y[i];
    }
}

int acgsolvercuda_daxpy_alpha(
    int n,
    const double * d_rnrm2sqr,
    const double * d_pdott,
    const double * d_x,
    double * d_y)
{
    static int mingridsize = 0, blocksize = 0;
    if (mingridsize == 0 && blocksize == 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &mingridsize, &blocksize, acgsolvercuda_daxpy_alpha_kernel, 0, 0);
    }
    acgsolvercuda_daxpy_alpha_kernel<<<mingridsize,blocksize>>>(
        n, d_rnrm2sqr, d_pdott, d_x, d_y);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    return ACG_SUCCESS;
}

__global__ void acgsolvercuda_daxpy_minus_alpha_kernel(
    int n,
    const double * rnrm2sqr,
    const double * pdott,
    const double * x,
    double * y)
{
    double a = -(*rnrm2sqr) / (*pdott);
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = a*x[i]+y[i];
    }
}

int acgsolvercuda_daxpy_minus_alpha(
    int n,
    const double * d_rnrm2sqr,
    const double * d_pdott,
    const double * d_x,
    double * d_y)
{
    static int mingridsize = 0, blocksize = 0;
    if (mingridsize == 0 && blocksize == 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &mingridsize, &blocksize, acgsolvercuda_daxpy_minus_alpha_kernel, 0, 0);
    }
    acgsolvercuda_daxpy_minus_alpha_kernel<<<mingridsize,blocksize>>>(
        n, d_rnrm2sqr, d_pdott, d_x, d_y);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    return ACG_SUCCESS;
}

__global__ void acgsolvercuda_pipelined_daxpy_fused_kernel(
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
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        z[i] = q[i]+beta*z[i];
        t[i] = w[i]+beta*t[i];
        p[i] = r[i]+beta*p[i];
        x[i] += alpha*p[i];
        r[i] -= alpha*t[i];
        w[i] -= alpha*z[i];
    }

    cg::grid_group grid = cg::this_grid();
    cg::sync(grid);
    if (grid.thread_rank() == 0) {
        *gamma_prev = *gamma;
        *alpha_prev = alpha;
    }
}

int acgsolvercuda_pipelined_daxpy_fused(
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
    cudaStream_t stream)
{
    static int mingridsize = 0, blocksize = 0, sharedmemsize = 0;
    if (mingridsize == 0 && blocksize == 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &mingridsize, &blocksize, acgsolvercuda_pipelined_daxpy_fused_kernel, 0, 0);
    }
    /* acgsolvercuda_pipelined_daxpy_fused_kernel<<<mingridsize,blocksize>>>( */
    /*     n, d_gamma, d_gamma_prev, d_delta, */
    /*     d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha_prev); */
    /* if (cudaPeekAtLastError()) return ACG_ERR_CUDA; */

    /* launch device-side CG kernel */
    void * kernelargs[] = {
        (void *) &n,
        (void *) &d_gamma,
        (void *) &d_gamma_prev,
        (void *) &d_delta,
        (void *) &d_q,
        (void *) &d_p,
        (void *) &d_r,
        (void *) &d_t,
        (void *) &d_x,
        (void *) &d_z,
        (void *) &d_w,
        (void *) &d_alpha_prev };
    dim3 blockDim(blocksize, 1, 1);
    dim3 gridDim(mingridsize, 1, 1);
    cudaLaunchCooperativeKernel(
        (void *) acgsolvercuda_pipelined_daxpy_fused_kernel, gridDim, blockDim,
        kernelargs, sharedmemsize, stream);

    return ACG_SUCCESS;
}

__global__ void acgsolvercuda_daypx_beta_kernel(
    int n,
    const double * rnrm2sqr,
    const double * rnrm2sqr_prev,
    double * y,
    const double * x)
{
    double a = (*rnrm2sqr) / (*rnrm2sqr_prev);
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = a*y[i]+x[i];
    }
}

int acgsolvercuda_daypx_beta(
    int n,
    const double * d_rnrm2sqr,
    const double * d_rnrm2sqr_prev,
    double * d_y,
    const double * d_x)
{
    static int mingridsize = 0, blocksize = 0;
    if (mingridsize == 0 && blocksize == 0) {
        cudaOccupancyMaxPotentialBlockSize(
            &mingridsize, &blocksize, acgsolvercuda_daypx_beta_kernel, 0, 0);
    }
    acgsolvercuda_daypx_beta_kernel<<<mingridsize,blocksize>>>(
        n, d_rnrm2sqr, d_rnrm2sqr_prev, d_y, d_x);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    return ACG_SUCCESS;
}

/*
 * Monolithic CG (CUDA)
 */

#define THREADS_PER_BLOCK 1024
#define TASKS_PER_THREAD 6

__global__ void csrgemv_merge_startrows(
    acgidx_t n,
    const acgidx_t * __restrict rowptr,
    acgidx_t nstartrows,
    acgidx_t * startrows)
{
    acgidx_t nnzs = rowptr[n];
    acgidx_t ntasks = n+nnzs;
    for (acgidx_t l = blockIdx.x*blockDim.x+threadIdx.x;
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
        #pragma unroll
        for (int j = 0; j < TASKS_PER_THREAD; j++) {
            sa[j*THREADS_PER_BLOCK+threadIdx.x] = alpha*a[kmin+j*THREADS_PER_BLOCK+threadIdx.x]*x[colidx[kmin+j*THREADS_PER_BLOCK+threadIdx.x]];
            srowptr[j*THREADS_PER_BLOCK+threadIdx.x] = rowptr[imin+j*THREADS_PER_BLOCK+threadIdx.x];
        }
        __syncthreads();

        acgidx_t i = startrows[l];
        acgidx_t k = l*TASKS_PER_THREAD-i; /* starting nonzero */
        double sum = 0.0;
        #pragma unroll
        for (short j = 0; j < TASKS_PER_THREAD; j++) {
            if (k == srowptr[i-imin+1]) {
                atomicAdd(&y[i], sum);
                i++, sum = 0.0;
            } else {
                sum += sa[k-kmin];
                k++;
            }
        }

        /* warp reduction of partial row results */
        int rowmask = __match_any_sync(0xFFFFFFFF, i);
        short lane = threadIdx.x & 31;
        short npeers = __popc(rowmask);
        short minlane = __ffs(rowmask)-1;
        short maxlane = minlane + npeers;
        #pragma unroll
        for (short offset = 16; offset > 0; offset /= 2) {
            short nextlane = lane+offset < maxlane ? lane+offset : 32;
            double x = __shfl_sync(rowmask, sum, nextlane);
            if (nextlane < maxlane) sum += x;
        }
        if (lane == minlane) atomicAdd(&y[i], sum);
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
        if (k < rowptr[i+1]) atomicAdd(&y[i], alpha*a[k]*x[colidx[k]]);

        /* b) warp reduction of partial row results */
        /* double sum = (k < rowptr[i+1]) ? a[k]*x[colidx[k]] : 0.0; */
        /* int rowmask = __match_any_sync(__activemask(), i); */
        /* short lane = threadIdx.x & 31; */
        /* short npeers = __popc(rowmask); */
        /* short minlane = __ffs(rowmask)-1; */
        /* short maxlane = minlane + npeers; */
        /* #pragma unroll */
        /* for (short offset = 16; offset > 0; offset /= 2) { */
        /*     short nextlane = lane+offset < maxlane ? lane+offset : 32; */
        /*     double x = __shfl_sync(rowmask, sum, nextlane); */
        /*     if (nextlane < maxlane) sum += x; */
        /* } */
        /* if (lane == minlane) atomicAdd(&y[i], sum); */
    }
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

__inline__ __device__ double warpReduceSum(int mask, double x)
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
    /* compute per-thread partial results */
    double z = 0;
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        z += x[i]*y[i];
    }

    /* perform reduction across threads in a warp */
    z = warpReduceSum(0xffffffff, z);

    /* a) use atomics to perform reduction across warps in the grid */
    if ((threadIdx.x & (warpSize-1)) == 0)
        atomicAdd(dot, z);

    /* b) use shared memory to perform reduction across warps in a
     * thread block, and then atomic operations to perform reduction
     * across all thread blocks in the grid */
    /* extern __shared__ double w[]; */
    /* cg::thread_block tb = cg::this_thread_block(); */
    /* cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(tb); */
    /* if (tile32.thread_rank() == 0) w[tile32.meta_group_rank()] = z; */
    /* cg::sync(tb); */
    /* if (tile32.meta_group_rank() == 0) { */
    /*     z = tile32.thread_rank() < tile32.meta_group_size() ? w[tile32.thread_rank()] : 0.0; */
    /*     z = cg::reduce(tile32, z, cg::plus<double>()); */
    /*     if (tile32.thread_rank() == 0) atomicAdd(dot, z); */
    /* } */
}

__device__ void daxpy(
    acgidx_t n,
    double alpha,
    const double * __restrict x,
    double * __restrict y)
{
    for (acgidx_t i = blockIdx.x*blockDim.x+threadIdx.x;
         i < n;
         i += blockDim.x*gridDim.x)
    {
        y[i] = alpha*x[i]+y[i];
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
    }
}

#if defined(ACG_HAVE_NVSHMEM)
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

// #define ACG_NVSHMEM_NOSENDRECV
// #define ACG_NVSHMEM_SENDRECV_SINGLE_BLOCK
// #define ACG_NVSHMEM_NOALLREDUCE
// #define ACG_NVSHMEM_ALLREDUCE_BLOCK
// #define ACG_NVSHMEM_ALLREDUCE_WARP

__global__ void __launch_bounds__(1024) acgsolvercuda_cg_kernel(
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
    int npes = nvshmem_n_pes();

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
    dcopy(n-nghosts, r, b);
    cg::sync(grid);

    /* start halo exchange to update ghost entries of x */
    if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
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
#endif
    }

    /* compute the initial residual, r₀ = b-A*x₀ */
    csrgemv_merge(n-nghosts, r, x, rowptr, colidx, a, -1.0, nstartrows, startrows);

    /* wait for halo exchange to finish before multiplying
     * off-diagonal part */
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
        if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
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
#endif
        }

        /* compute t = Ap (local part) */
        csrgemv_merge(n-nghosts, t, p, rowptr, colidx, a, 1.0, nstartrows, startrows);

        /* wait for halo exchange to finish and compute t = Ap (remote part) */
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
#endif

/**
 * ‘acgsolvercuda_solve_device()’ solves the given linear system,
 * Ax=b, using the conjugate gradient method. The linear system may be
 * distributed across multiple processes and communication is handled
 * using device-initiated NVSHMEM.
 *
 * The solver must already have been configured with ‘acgsolvercuda_init()’
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
int acgsolvercuda_solve_device(
    struct acgsolvercuda * cg,
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
#if !defined(ACG_HAVE_NVSHMEM)
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#else
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

    int commsize, rank;
    acgcomm_size(comm, &commsize);
    acgcomm_rank(comm, &rank);
    if (comm->type != acgcomm_nvshmem) return ACG_ERR_NOT_SUPPORTED;

    cudaStream_t stream = 0;
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
    err = acghaloexchange_init_cuda(
        &haloexchange, cg->halo, ACG_DOUBLE, ACG_DOUBLE, comm, stream);
    if (err) return err;

    /* copy right-hand side and initial guess to device */
    double * d_b;
    err = cudaMalloc((void **) &d_b, b->num_nonzeros*sizeof(*d_b));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_b, b->x, b->num_nonzeros*sizeof(*d_b), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    double * d_x;
    err = cudaMalloc((void **) &d_x, x->num_nonzeros*sizeof(*d_x));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_x, x->x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;

    /* enable maximum amount of shared memory for merge-based spmv */
    int sharedmemsize = THREADS_PER_BLOCK*TASKS_PER_THREAD*(sizeof(double)+sizeof(acgidx_t));
    /* err = cudaFuncSetAttribute( */
    /*     acgsolvercuda_cg_kernel, */
    /*     cudaFuncAttributePreferredSharedMemoryCarveout, */
    /*     cudaSharedmemCarveoutMaxShared); */
    /* if (err) return ACG_ERR_CUDA; */
    err = cudaFuncSetAttribute(
        acgsolvercuda_cg_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedmemsize);
    if (err) return ACG_ERR_CUDA;

    /* determine grid and thread block size */
    int mingridsize = 0, blocksize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &mingridsize, &blocksize, acgsolvercuda_cg_kernel,
        sharedmemsize, 0);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, device);
    int nmultiprocessors = deviceprop.multiProcessorCount;
    int nblockspersm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nblockspersm, acgsolvercuda_cg_kernel, blocksize,
        sharedmemsize);
    int nblocks = nmultiprocessors*nblockspersm;
    dim3 blockDim(blocksize, 1, 1);
    dim3 gridDim(nblocks, 1, 1);

    fprintf(stderr, "\n%s: rank=%d blockDim=(%d,%d,%d) gridDim=(%d,%d,%d) n=%d nghosts=%d\n",
            __FUNCTION__, rank, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, n, nghosts);

    acgidx_t ntasks = (A->nprows-A->nghostrows)+A->fnpnzs;
    acgidx_t nstartrows = ntasks/TASKS_PER_THREAD;
    acgidx_t * d_startrows;
    err = cudaMalloc((void **) &d_startrows, nstartrows*sizeof(*d_startrows));
    if (err) return ACG_ERR_CUDA;
    csrgemv_merge_startrows<<<gridDim.x,blockDim.x>>>(
        n-nghosts, d_rowptr, nstartrows, d_startrows);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    cudaStreamSynchronize(stream);

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
        err = nvshmemx_collective_launch(
            (void *) acgsolvercuda_cg_kernel, gridDim, blockDim,
            kernelargs, sharedmemsize, stream);
         if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
         if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
         cudaStreamSynchronize(stream);
         err = cudaMemcpy(d_x, x->x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyHostToDevice);
         if (err) return ACG_ERR_CUDA;
    }

    int converged = 0;
    acgtime_t t0, t1;
    err = acgcomm_barrier(stream, comm, errcode);
    if (err) return err;
    cudaStreamSynchronize(stream);
    MPI_Barrier(comm->mpicomm);
    cudaStreamSynchronize(stream);
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

    /* cudaLaunchCooperativeKernel( */
    /*     (void *) acgsolvercuda_cg_kernel, gridDim, blockDim, */
    /*     kernelargs, sharedmemsize, stream); */

    err = nvshmemx_collective_launch(
        (void *) acgsolvercuda_cg_kernel, gridDim, blockDim,
        kernelargs, sharedmemsize, stream);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }

    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    cudaStreamSynchronize(stream);
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

    /* copy solution back to host */
    err = cudaMemcpy(x->x, d_x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;

    /* free vectors */
    cudaFree(d_startrows);
    cudaFree(d_x); cudaFree(d_b);
    acghaloexchange_free(&haloexchange);

    /* check for CUDA errors */
    if (cudaGetLastError() != cudaSuccess)
        return ACG_ERR_CUDA;

    /* copy results from device to host */
    err = cudaMemcpy(&cg->bnrm2, d_bnrm2sqr, sizeof(cg->bnrm2), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->bnrm2 = sqrt(cg->bnrm2);
    err = cudaMemcpy(&cg->r0nrm2, d_r0nrm2sqr, sizeof(cg->r0nrm2), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->r0nrm2 = sqrt(cg->r0nrm2);
    err = cudaMemcpy(&cg->rnrm2, d_rnrm2sqr, sizeof(cg->rnrm2), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->rnrm2 = sqrt(cg->rnrm2);
    err = cudaMemcpy(&cg->niterations, d_niterations, sizeof(cg->niterations), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->ntotaliterations += cg->niterations;
    err = cudaMemcpy(&converged, d_converged, sizeof(converged), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged) return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
#endif
}

#if defined(ACG_HAVE_NVSHMEM)
__global__ void __launch_bounds__(1024) acgsolvercuda_cg_pipelined_kernel(
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
    double * __restrict w,
    double * __restrict q,
    double * __restrict z,
    double * __restrict bnrm2sqr,
    double * __restrict r0nrm2sqr,
    double * __restrict rnrm2sqr,
    double * __restrict delta,
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
    int npes = nvshmem_n_pes();

    /* reset scalar values for accumulating dot products */
    double alpha = INFINITY, beta = INFINITY;
    if (grid.thread_rank() == 0) {
        *bnrm2sqr = *rnrm2sqr = *r0nrm2sqr = *delta = 0.0;
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
    dcopy(n-nghosts, r, b);
    cg::sync(grid);

    /* start halo exchange to update ghost entries of x */
    if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
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
#endif
    }

    /* compute the initial residual, r₀ = b-A*x₀ */
    csrgemv_merge(n-nghosts, r, x, rowptr, colidx, a, -1.0, nstartrows, startrows);

    /* wait for halo exchange to finish before multiplying
     * off-diagonal part */
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
    cg::sync(grid);

    /* set w to zero before computing w = Ar */
    dzero(n-nghosts, w);
    cg::sync(grid);

    /* start halo exchange to update ghost entries of r */
    if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
#if defined (ACG_NVSHMEM_SENDRECV_SINGLE_BLOCK)
        pack_double(sendsize, sendbuf, n, r, sendbufidx);
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
            pack_double_block(sendcounts[q], sendbufq, n, r, sendbufidx+sdispls[q]);
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
#endif
    }

    /* compute w = Ar (local part) */
    csrgemv_merge(n-nghosts, w, r, rowptr, colidx, a, 1.0, nstartrows, startrows);

    /* wait for halo exchange to finish and compute w = Ar (remote part) */
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
            unpack_double_block(recvcounts[q], recvbuf+rdispls[q], n, r, recvbufidx+rdispls[q]);
            cg::sync(block);
            if (block.thread_rank() == 0) {
                nvshmemx_signal_op(&readytoreceive[getranks[q]], 0, NVSHMEM_SIGNAL_SET, senders[q]);
            }
        }
#endif
        cg::sync(grid);
        csrgemv(nborderrows, w+borderrowoffset, r+borderrowoffset, orowptr, ocolidx, oa, 1.0);
    }
    cg::sync(grid);

    /* set the vectors z, t, p and q to zero */
    dzero(n-nghosts, z);
    dzero(n-nghosts, t);
    dzero(n-nghosts, p);
    dzero(n-nghosts, q);

    /* iterative solver loop */
    for (int k = 0; k < maxits; k++) {

        /* reset scalar values */
        if (grid.thread_rank() == 0) *delta = *rnrm2sqr = 0;
        cg::sync(grid);

        /* compute residual norm (r,r) and (w,r) */
        ddot(n-nghosts, r, r, rnrm2sqr);
        ddot(n-nghosts, w, r, delta);

        /* perform a single reduction for the two dot products */
#ifndef ACG_NVSHMEM_NOALLREDUCE
        if (npes > 1) {
            cg::sync(grid);
#if defined(ACG_NVSHMEM_ALLREDUCE_BLOCK)
            if (grid.block_rank() == 0)
                nvshmemx_double_sum_reduce_block(
                    NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 2);
#elif defined(ACG_NVSHMEM_ALLREDUCE_WARP)
            if (grid.block_rank() == 0 && block.thread_rank() < warpSize)
                nvshmemx_double_sum_reduce_warp(
                    NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 2);
#else
            if (grid.thread_rank() == 0)
                nvshmem_double_sum_reduce(
                    NVSHMEM_TEAM_WORLD, rnrm2sqr, rnrm2sqr, 2);
#endif
        }
#endif

        /* start halo exchange to update ghost entries of w */
        if (npes > 1) {
#ifndef ACG_NVSHMEM_NOSENDRECV
#if defined (ACG_NVSHMEM_SENDRECV_SINGLE_BLOCK)
            pack_double(sendsize, sendbuf, n, w, sendbufidx);
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
                pack_double_block(sendcounts[q], sendbufq, n, w, sendbufidx+sdispls[q]);
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
#endif
        }

        /* compute q = Aw (local part) */
        csrgemv_merge(n-nghosts, q, w, rowptr, colidx, a, 1.0, nstartrows, startrows);

        /* wait for halo exchange to finish and compute q = Aw (remote part) */
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
                unpack_double_block(recvcounts[q], recvbuf+rdispls[q], n, w, recvbufidx+rdispls[q]);
                cg::sync(block);
                if (block.thread_rank() == 0) {
                    nvshmemx_signal_op(&readytoreceive[getranks[q]], 0, NVSHMEM_SIGNAL_SET, senders[q]);
                }
            }
#endif
            cg::sync(grid);
            csrgemv(nborderrows, q+borderrowoffset, w+borderrowoffset, orowptr, ocolidx, oa, 1.0);
        }
        cg::sync(grid);

        /* convergence tests */
        double rnrm2 = sqrt(*rnrm2sqr);
        if (k == 0 && grid.thread_rank() == 0) *r0nrm2sqr = *rnrm2sqr;
        if (k == 0) residualrtol *= rnrm2;
        if ((residualatol > 0 && rnrm2 < residualatol) ||
            (residualrtol > 0 && rnrm2 < residualrtol))
        {
            if (grid.thread_rank() == 0) {
                *niterations = k+1;
                *converged = true;
            }
            return;
        }

        /* compute scalars: alpha, beta */
        beta = (*rnrm2sqr) / beta;
        alpha = (*rnrm2sqr) / ((*delta) - beta*(*rnrm2sqr)/alpha);

        /* update vectors */
        for (int i = blockIdx.x*blockDim.x+threadIdx.x;
             i < n-nghosts;
             i += blockDim.x*gridDim.x)
        {
            z[i] = q[i]+beta*z[i];
            t[i] = w[i]+beta*t[i];
            p[i] = r[i]+beta*p[i];
            x[i] += alpha*p[i];
            r[i] -= alpha*t[i];
            w[i] -= alpha*z[i];
            q[i] = 0.0;
        }

        beta = *rnrm2sqr;
        cg::sync(grid);
    }

    if (grid.thread_rank() == 0) {
        *niterations = maxits;
        *converged = false;
    }
}
#endif

/**
 * ‘acgsolvercuda_solve_device_pipelined()’ solves the given linear
 * system, Ax=b, using a pipelined conjugate gradient method. The
 * linear system may be distributed across multiple processes and
 * communication is handled using device-initiated NVSHMEM.
 *
 * The solver must already have been configured with ‘acgsolvercuda_init()’
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
int acgsolvercuda_solve_device_pipelined(
    struct acgsolvercuda * cg,
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
#if !defined(ACG_HAVE_NVSHMEM)
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#else
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

    int commsize, rank;
    acgcomm_size(comm, &commsize);
    acgcomm_rank(comm, &rank);
    if (comm->type != acgcomm_nvshmem) return ACG_ERR_NOT_SUPPORTED;

    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    if (!supportsCoopLaunch) {
      fprintf(stderr, "cooperative group launch not supported!\n");
      return ACG_ERR_NOT_SUPPORTED;
    }

    /* allocate extra vectors needed for pipelined CG */
    if (!cg->w) {
        cg->w = (struct acgvector *) malloc(sizeof(*cg->w)); if (!cg->w) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->w, x); if (err) return err;
        err = cudaMalloc((void **) &cg->d_w, cg->w->num_nonzeros*sizeof(*cg->d_w));
        if (err) return ACG_ERR_CUDA;
    }
    if (!cg->q) {
        cg->q = (struct acgvector *) malloc(sizeof(*cg->q)); if (!cg->q) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->q, x); if (err) return err;
        err = cudaMalloc((void **) &cg->d_q, cg->q->num_nonzeros*sizeof(*cg->d_q));
        if (err) return ACG_ERR_CUDA;
    }
    if (!cg->z) {
        cg->z = (struct acgvector *) malloc(sizeof(*cg->z)); if (!cg->z) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->z, x); if (err) return err;
        err = cudaMalloc((void **) &cg->d_z, cg->z->num_nonzeros*sizeof(*cg->d_z));
        if (err) return ACG_ERR_CUDA;
    }

    cudaStream_t stream = 0;
    const struct acghalo * halo = cg->halo;
    double * d_bnrm2sqr = cg->d_bnrm2sqr;
    double * d_r0nrm2sqr = cg->d_r0nrm2sqr;
    double * d_rnrm2sqr = &cg->d_rnrm2sqr[0];
    /* double * d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev; */
    double * d_delta = &cg->d_rnrm2sqr[1];
    int * d_niterations = cg->d_niterations;
    int * d_converged = cg->d_converged;
    /* double * d_one = cg->d_one; */
    /* double * d_minus_one = cg->d_minus_one; */
    /* double * d_zero = cg->d_zero; */
    double * d_r = cg->d_r;
    double * d_p = cg->d_p;
    double * d_t = cg->d_t;
    double * d_w = cg->d_w;
    double * d_q = cg->d_q;
    double * d_z = cg->d_z;
    acgidx_t * d_rowptr = cg->d_rowptr;
    acgidx_t * d_colidx = cg->d_colidx;
    double * d_a = cg->d_a;
    acgidx_t * d_orowptr = cg->d_orowptr;
    acgidx_t * d_ocolidx = cg->d_ocolidx;
    double * d_oa = cg->d_oa;

    /* prepare for halo exchange */
    struct acghaloexchange haloexchange;
    err = acghaloexchange_init_cuda(
        &haloexchange, cg->halo, ACG_DOUBLE, ACG_DOUBLE, comm, stream);
    if (err) return err;

    /* copy right-hand side and initial guess to device */
    double * d_b;
    err = cudaMalloc((void **) &d_b, b->num_nonzeros*sizeof(*d_b));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_b, b->x, b->num_nonzeros*sizeof(*d_b), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    double * d_x;
    err = cudaMalloc((void **) &d_x, x->num_nonzeros*sizeof(*d_x));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_x, x->x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;

    /* enable maximum amount of shared memory for merge-based spmv */
    int sharedmemsize = THREADS_PER_BLOCK*TASKS_PER_THREAD*(sizeof(double)+sizeof(acgidx_t));
    /* err = cudaFuncSetAttribute( */
    /*     acgsolvercuda_cg_pipelined_kernel, */
    /*     cudaFuncAttributePreferredSharedMemoryCarveout, */
    /*     cudaSharedmemCarveoutMaxShared); */
    /* if (err) return ACG_ERR_CUDA; */
    err = cudaFuncSetAttribute(
        acgsolvercuda_cg_pipelined_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedmemsize);
    if (err) return ACG_ERR_CUDA;

    /* determine grid and thread block size */
    int mingridsize = 0, blocksize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &mingridsize, &blocksize, acgsolvercuda_cg_pipelined_kernel,
        sharedmemsize, 0);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceprop;
    cudaGetDeviceProperties(&deviceprop, device);
    int nmultiprocessors = deviceprop.multiProcessorCount;
    int nblockspersm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nblockspersm, acgsolvercuda_cg_pipelined_kernel, blocksize,
        sharedmemsize);
    int nblocks = nmultiprocessors*nblockspersm;
    dim3 blockDim(blocksize, 1, 1);
    dim3 gridDim(nblocks, 1, 1);

    fprintf(stderr, "\n%s: rank=%d blockDim=(%d,%d,%d) gridDim=(%d,%d,%d) n=%d nghosts=%d\n",
            __FUNCTION__, rank, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, n, nghosts);

    acgidx_t ntasks = (A->nprows-A->nghostrows)+A->fnpnzs;
    acgidx_t nstartrows = ntasks/TASKS_PER_THREAD;
    acgidx_t * d_startrows;
    err = cudaMalloc((void **) &d_startrows, nstartrows*sizeof(*d_startrows));
    if (err) return ACG_ERR_CUDA;
    csrgemv_merge_startrows<<<gridDim.x,blockDim.x>>>(
        n-nghosts, d_rowptr, nstartrows, d_startrows);
    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    cudaStreamSynchronize(stream);

    /* warmup iterations */
    if (warmup > 0) {
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
            (void *) &d_w,
            (void *) &d_q,
            (void *) &d_z,
            (void *) &d_bnrm2sqr,
            (void *) &d_r0nrm2sqr,
            (void *) &d_rnrm2sqr,
            (void *) &d_delta,
            (void *) &d_niterations,
            (void *) &d_converged,
            (void *) &warmup,
            (void *) &diffatol,
            (void *) &diffrtol,
            (void *) &residualatol,
            (void *) &residualrtol,
            (void *) &nstartrows,
            (void *) &d_startrows };
        err = nvshmemx_collective_launch(
            (void *) acgsolvercuda_cg_pipelined_kernel, gridDim, blockDim,
            kernelargs, sharedmemsize, stream);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
        if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
        cudaStreamSynchronize(stream);
        err = cudaMemcpy(d_x, x->x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyHostToDevice);
        if (err) return ACG_ERR_CUDA;
    }

    int converged = 0;
    acgtime_t t0, t1;
    err = acgcomm_barrier(stream, comm, errcode);
    if (err) return err;
    cudaStreamSynchronize(stream);
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
        (void *) &d_w,
        (void *) &d_q,
        (void *) &d_z,
        (void *) &d_bnrm2sqr,
        (void *) &d_r0nrm2sqr,
        (void *) &d_rnrm2sqr,
        (void *) &d_delta,
        (void *) &d_niterations,
        (void *) &d_converged,
        (void *) &maxits,
        (void *) &diffatol,
        (void *) &diffrtol,
        (void *) &residualatol,
        (void *) &residualrtol,
        (void *) &nstartrows,
        (void *) &d_startrows };

    /* cudaLaunchCooperativeKernel( */
    /*     (void *) acgsolvercuda_cg_pipelined_kernel, gridDim, blockDim, */
    /*     kernelargs, sharedmemsize, stream); */

    err = nvshmemx_collective_launch(
        (void *) acgsolvercuda_cg_pipelined_kernel, gridDim, blockDim,
        kernelargs, sharedmemsize, stream);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }

    if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
    cudaStreamSynchronize(stream);
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

    /* copy solution back to host */
    err = cudaMemcpy(x->x, d_x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;

    /* free vectors */
    cudaFree(d_startrows);
    cudaFree(d_x); cudaFree(d_b);
    acghaloexchange_free(&haloexchange);

    /* check for CUDA errors */
    if (cudaGetLastError() != cudaSuccess)
        return ACG_ERR_CUDA;

    /* copy results from device to host */
    err = cudaMemcpy(&cg->bnrm2, d_bnrm2sqr, sizeof(cg->bnrm2), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->bnrm2 = sqrt(cg->bnrm2);
    err = cudaMemcpy(&cg->r0nrm2, d_r0nrm2sqr, sizeof(cg->r0nrm2), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->r0nrm2 = sqrt(cg->r0nrm2);
    err = cudaMemcpy(&cg->rnrm2, d_rnrm2sqr, sizeof(cg->rnrm2), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->rnrm2 = sqrt(cg->rnrm2);
    err = cudaMemcpy(&cg->niterations, d_niterations, sizeof(cg->niterations), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->ntotaliterations += cg->niterations;
    err = cudaMemcpy(&converged, d_converged, sizeof(converged), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged) return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
#endif
}
