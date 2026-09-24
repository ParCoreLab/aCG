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
 * conjugate gradient (CG) solver using CUDA
 */

#include "acg/cgcuda.h"
#include "acg/cg-kernels-cuda.h"
#include "acg/comm.h"
#include "acg/config.h"
#include "acg/error.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/time.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef ACG_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef ACG_HAVE_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef ACG_HAVE_CUSPARSE
#include <cusparse.h>
#endif

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef ACG_HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#endif
#include <driver_types.h>

/*
 * profiling
 */
#define TASKS_PER_THREAD 6
#define ACG_ENABLE_PROFILING 1
#define DEBUG 1
#define ACG_USE_CUSPARSE 1
#define ACG_USE_CUBLAS 1

#ifdef ACG_HAVE_NVTX
#define acgSetStreamName(stream, name) nvtxNameCudaStreamA(stream, name)
#else
#define acgSetStreamName(stream, name)
#endif

#ifdef ACG_ENABLE_PROFILING
#define acgEventRecord(event, stream) cudaEventRecord((event), (stream))
#else
#define acgEventRecord(event, stream)
#endif

#define ACG_PIPELINED_ALLREDUCE_SIZE 2

/*
 * memory management
 */

/**
 * ‘acgsolvercuda_free()’ frees storage allocated for a solver.
 */
void acgsolvercuda_free(struct acgsolvercuda *cg)
{
  acgvector_free(&cg->r);
  acgvector_free(&cg->p);
  acgvector_free(&cg->t);
  if (cg->w)
  {
    acgvector_free(cg->w);
  }
  free(cg->w);
  if (cg->q)
  {
    acgvector_free(cg->q);
  }
  free(cg->q);
  if (cg->z)
  {
    acgvector_free(cg->z);
  }
  free(cg->z);
  if (cg->dx)
  {
    acgvector_free(cg->dx);
  }
  free(cg->dx);
  if (cg->halo)
    acghalo_free(cg->halo);
  free(cg->halo);
  if (cg->haloexchange)
    acghaloexchange_free(cg->haloexchange);
  free(cg->haloexchange);
  if (!cg->use_nvshmem)
  {
    cudaFree(cg->d_bnrm2sqr);
    cudaFree(cg->d_r0nrm2sqr);
    cudaFree(cg->d_rnrm2sqr);
    cudaFree(cg->d_pdott);
  }
  else
  {
    acgcomm_nvshmem_free(cg->d_bnrm2sqr);
    acgcomm_nvshmem_free(cg->d_r0nrm2sqr);
    acgcomm_nvshmem_free(cg->d_rnrm2sqr);
    acgcomm_nvshmem_free(cg->d_pdott);
  }
  cudaFree(cg->d_rnrm2sqr_prev);
  cudaFree(cg->d_alpha);
  cudaFree(cg->d_minus_alpha);
  cudaFree(cg->d_beta);
  cudaFree(cg->d_niterations);
  cudaFree(cg->d_converged);
  cudaFree(cg->d_r);
  cudaFree(cg->d_p);
  cudaFree(cg->d_t);
  if (cg->d_w)
    cudaFree(cg->d_w);
  if (cg->d_q)
    cudaFree(cg->d_q);
  if (cg->d_z)
    cudaFree(cg->d_z);
  cudaFree(cg->d_rowptr);
  cudaFree(cg->d_colidx);
  cudaFree(cg->d_a);
  cudaFree(cg->d_orowptr);
  cudaFree(cg->d_ocolidx);
  cudaFree(cg->d_oa);
}

/*
 * initialise a solver
 */

#if defined(ACG_HAVE_CUBLAS) && defined(ACG_HAVE_CUSPARSE)
/**
 * ‘acgsolvercuda_init()’ sets up a conjugate gradient solver for a given
 * sparse matrix in CSR format.
 *
 * The matrix may be partitioned and distributed.
 */
int acgsolvercuda_init(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    const struct acgcomm *comm)
{
  int err = acgsymcsrmatrix_vector(A, &cg->r);
  if (err)
    return err;
  acgvector_setzero(&cg->r);
  err = acgsymcsrmatrix_vector(A, &cg->p);
  if (err)
  {
    acgvector_free(&cg->r);
    return err;
  }
  acgvector_setzero(&cg->p);
  err = acgsymcsrmatrix_vector(A, &cg->t);
  if (err)
  {
    acgvector_free(&cg->p);
    acgvector_free(&cg->r);
    return err;
  }
  acgvector_setzero(&cg->t);
  cg->w = cg->q = cg->z = cg->m = cg->n = cg->u = cg->y = NULL;
  cg->dx = NULL;
  cg->halo = malloc(sizeof(*cg->halo));
  if (!cg->halo)
  {
    acgvector_free(&cg->t);
    acgvector_free(&cg->p);
    acgvector_free(&cg->r);
    return ACG_ERR_ERRNO;
  }
  err = acgsymcsrmatrix_halo(A, cg->halo);
  if (err)
  {
    free(cg->haloexchange);
    free(cg->halo);
    acgvector_free(&cg->t);
    acgvector_free(&cg->p);
    acgvector_free(&cg->r);
    return err;
  }
  cg->haloexchange = malloc(sizeof(*cg->haloexchange));
  if (!cg->haloexchange)
  {
    acghalo_free(cg->halo);
    free(cg->halo);
    acgvector_free(&cg->t);
    acgvector_free(&cg->p);
    acgvector_free(&cg->r);
    return ACG_ERR_ERRNO;
  }
  cudaStream_t stream = 0;
  err = acghaloexchange_init_cuda(
      cg->haloexchange, cg->halo, ACG_DOUBLE, ACG_DOUBLE, comm, stream);
  if (err)
  {
    free(cg->haloexchange);
    acghalo_free(cg->halo);
    free(cg->halo);
    acgvector_free(&cg->t);
    acgvector_free(&cg->p);
    acgvector_free(&cg->r);
    return err;
  }

  /* initialise device-side data */
  /* err = acgsolvercuda_init_constants( */
  /*     &cg->d_minus_one, &cg->d_one, &cg->d_zero); */
  /* if (err) return err; */

  double one = 1.0, minus_one = -1.0, zero = 0.0, inf = INFINITY;
  err = cudaMalloc((void **)&cg->d_one, sizeof(*cg->d_one));
  if (err)
    return ACG_ERR_CUDA;
  err =
      cudaMemcpy(cg->d_one, &one, sizeof(*cg->d_one), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_minus_one, sizeof(*cg->d_minus_one));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      cg->d_minus_one, &minus_one, sizeof(*cg->d_minus_one),
      cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_zero, sizeof(*cg->d_zero));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      cg->d_zero, &zero, sizeof(*cg->d_zero), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_inf, sizeof(*cg->d_inf));
  if (err)
    return ACG_ERR_CUDA;
  err =
      cudaMemcpy(cg->d_inf, &inf, sizeof(*cg->d_inf), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;

  cg->use_nvshmem = comm->type == acgcomm_nvshmem || comm->type == acgcomm_nvshmem_split;
  cg->use_nccl_split = comm->type == acgcomm_nccl_split;
  if (!cg->use_nvshmem)
  {
    err = cudaMalloc((void **)&cg->d_bnrm2sqr, sizeof(*cg->d_bnrm2sqr));
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMalloc((void **)&cg->d_r0nrm2sqr, sizeof(*cg->d_r0nrm2sqr));
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMalloc((void **)&cg->d_rnrm2sqr, ACG_PIPELINED_ALLREDUCE_SIZE * sizeof(*cg->d_rnrm2sqr));
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMalloc((void **)&cg->d_pdott, sizeof(*cg->d_pdott));
    if (err)
      return ACG_ERR_CUDA;
  }
  else
  {
    int errcode;
    err = acgcomm_nvshmem_malloc(
        (void **)&cg->d_bnrm2sqr, sizeof(*cg->d_bnrm2sqr), &errcode);
    if (err)
      return err;
    err = acgcomm_nvshmem_malloc(
        (void **)&cg->d_r0nrm2sqr, sizeof(*cg->d_r0nrm2sqr), &errcode);
    if (err)
      return err;
    err = acgcomm_nvshmem_malloc(
        (void **)&cg->d_rnrm2sqr, 2 * sizeof(*cg->d_rnrm2sqr), &errcode);
    if (err)
      return err;
    err = acgcomm_nvshmem_malloc(
        (void **)&cg->d_pdott, sizeof(*cg->d_pdott), &errcode);
    if (err)
      return err;
  }
  err =
      cudaMalloc((void **)&cg->d_rnrm2sqr_prev, sizeof(*cg->d_rnrm2sqr_prev));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_niterations, sizeof(*cg->d_niterations));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_converged, sizeof(*cg->d_converged));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_alpha, sizeof(*cg->d_alpha));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_minus_alpha, sizeof(*cg->d_minus_alpha));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_beta, sizeof(*cg->d_beta));
  if (err)
    return ACG_ERR_CUDA;

  /* allocate storage for auxiliary vectors on device */
  err = cudaMalloc((void **)&cg->d_r, cg->r.num_nonzeros * sizeof(*cg->d_r));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_p, cg->p.num_nonzeros * sizeof(*cg->d_p));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_t, cg->t.num_nonzeros * sizeof(*cg->d_t));
  if (err)
    return ACG_ERR_CUDA;
  cg->d_w = cg->d_q = cg->d_z = cg->d_m = cg->d_n = cg->d_u = cg->d_y = NULL;

  /* copy sparse matrix to device */
  err = cudaMalloc(
      (void **)&cg->d_rowptr, (A->nprows + 1) * sizeof(*cg->d_rowptr));
  if (err)
    return ACG_ERR_CUDA;
  if (sizeof(*cg->d_rowptr) == sizeof(*A->frowptr))
  {
    err = cudaMemcpy(
        cg->d_rowptr, A->frowptr, (A->nprows + 1) * sizeof(*cg->d_rowptr),
        cudaMemcpyHostToDevice);
    if (err)
      return ACG_ERR_CUDA;
  }
  else
  {
    acgidx_t *tmprowptr = malloc((A->nprows + 1) * sizeof(*tmprowptr));
    if (!tmprowptr)
      return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i <= A->nprows; i++)
    {
      if (A->frowptr[i] > ACGIDX_T_MAX)
      {
        return ACG_ERR_INDEX_OUT_OF_BOUNDS;
      }
      tmprowptr[i] = A->frowptr[i];
    }
    err = cudaMemcpy(
        cg->d_rowptr, tmprowptr, (A->nprows + 1) * sizeof(*cg->d_rowptr),
        cudaMemcpyHostToDevice);
    if (err)
      return ACG_ERR_CUDA;
    free(tmprowptr);
  }
  err = cudaMalloc((void **)&cg->d_colidx, A->fnpnzs * sizeof(*cg->d_colidx));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      cg->d_colidx, A->fcolidx, A->fnpnzs * sizeof(*cg->d_colidx),
      cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_a, A->fnpnzs * sizeof(*cg->d_a));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      cg->d_a, A->fa, A->fnpnzs * sizeof(*cg->d_a), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;

  /* copy sparse matrix to device */
  err = cudaMalloc(
      (void **)&cg->d_orowptr,
      (A->nborderrows + A->nghostrows + 1) * sizeof(*cg->d_orowptr));
  if (err)
    return ACG_ERR_CUDA;
  if (sizeof(*cg->d_orowptr) == sizeof(*A->orowptr))
  {
    err = cudaMemcpy(
        cg->d_orowptr, A->orowptr,
        (A->nborderrows + A->nghostrows + 1) * sizeof(*cg->d_orowptr),
        cudaMemcpyHostToDevice);
    if (err)
      return ACG_ERR_CUDA;
  }
  else
  {
    acgidx_t *tmprowptr =
        malloc((A->nborderrows + A->nghostrows + 1) * sizeof(*tmprowptr));
    if (!tmprowptr)
      return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i <= A->nborderrows + A->nghostrows; i++)
    {
      if (A->orowptr[i] > ACGIDX_T_MAX)
      {
        return ACG_ERR_INDEX_OUT_OF_BOUNDS;
      }
      tmprowptr[i] = A->orowptr[i];
    }
    err = cudaMemcpy(
        cg->d_orowptr, tmprowptr,
        (A->nborderrows + A->nghostrows + 1) * sizeof(*cg->d_orowptr),
        cudaMemcpyHostToDevice);
    if (err)
      return ACG_ERR_CUDA;
    free(tmprowptr);
  }
  err =
      cudaMalloc((void **)&cg->d_ocolidx, A->onpnzs * sizeof(*cg->d_ocolidx));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      cg->d_ocolidx, A->ocolidx, A->onpnzs * sizeof(*cg->d_ocolidx),
      cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&cg->d_oa, A->onpnzs * sizeof(*cg->d_oa));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      cg->d_oa, A->oa, A->onpnzs * sizeof(*cg->d_oa), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;

  cg->maxits = 0;
  cg->diffatol = 0;
  cg->diffrtol = 0;
  cg->residualatol = 0;
  cg->residualrtol = 0;
  cg->bnrm2 = 0;
  cg->r0nrm2 = cg->rnrm2 = 0;
  cg->x0nrm2 = cg->dxnrm2 = 0;
  cg->nsolves = 0;
  cg->ntotaliterations = cg->niterations = 0;
  cg->nflops = 0;
  cg->tsolve = 0;
  cg->tgemv = cg->tdot = cg->tnrm2 = cg->taxpy = cg->tcopy = cg->tallreduce =
      cg->thalo = 0;
  cg->ngemv = cg->ndot = cg->nnrm2 = cg->naxpy = cg->ncopy = cg->nallreduce =
      cg->nhalo = 0;
  cg->Bgemv = cg->Bdot = cg->Bnrm2 = cg->Baxpy = cg->Bcopy = cg->Ballreduce =
      cg->Bhalo = 0;
  cg->nhalomsgs = 0;
  return ACG_SUCCESS;
}
#endif

/*
 * iterative solution procedure
 */

/**
 * ‘acgsolvercuda_solve()’ solves the given linear system, Ax=b, using the
 * conjugate gradient method.
 *
 * The solver must already have been configured with ‘acgsolvercuda_init()’
 * for a linear system Ax=b, and the dimensions of the vectors b and x
 * must match the number of columns and rows of A, respectively.
 *
 * The stopping criterion are:
 *
 *  - ‘maxits’, the maximum number of iterations to perform
 *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ < γₐ
 *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖ <
 * γᵣ
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
int acgsolvercuda_solve(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup);

/*
 * iterative solution procedure in distributed memory using MPI
 */
/*
#if defined(ACG_HAVE_MPI) && defined(ACG_HAVE_CUBLAS)                          \
    && defined(ACG_HAVE_CUSPARSE)
*/
/**
 * ‘acgsolvercuda_solvempi()’ solves the given linear system, Ax=b, using
 * the conjugate gradient method. The linear system may be distributed
 * across multiple processes and communication is handled using MPI.
 *
 * The solver must already have been configured with ‘acgsolvercuda_init()’
 * for a linear system Ax=b, and the dimensions of the vectors b and x
 * must match the number of columns and rows of A, respectively.
 *
 * The stopping criterion are:
 *
 *  - ‘maxits’, the maximum number of iterations to perform
 *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ < γₐ
 *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖ <
 * γᵣ
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
int acgsolvercuda_solvempi(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    cusparseSpMVAlg_t cusparse_spmv_alg)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

  /* /\* If the stopping criterion is based on the difference in */
  /*  * solution from one iteration to the next, then allocate */
  /*  * additional storage for storing the difference. *\/ */
  /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
  /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO;
   */
  /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
  /* } */

  cudaStream_t stream = 0;
  const struct acghalo *halo = cg->halo;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  double *d_rnrm2sqr = cg->d_rnrm2sqr;
  double *d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
  double *d_pdott = cg->d_pdott;
  double *d_alpha = cg->d_alpha;
  double *d_minus_alpha = cg->d_minus_alpha;
  double *d_beta = cg->d_beta;
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_r = cg->d_r;
  double *d_p = cg->d_p;
  double *d_t = cg->d_t;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  double *rnrm2sqr;
  err = cudaMallocHost((void **)&rnrm2sqr, sizeof(*rnrm2sqr));
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t copystream;
  err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t rnrm2sqrready;
  cudaEventCreateWithFlags(&rnrm2sqrready, cudaEventDisableTiming);
  /* cudaEvent_t rnrm2sqrreceived; */
  /* err = cudaEventCreateWithFlags(&rnrm2sqrreceived,
   * cudaEventDisableTiming); if (err) return ACG_ERR_CUDA; */

  /* copy right-hand side and initial guess to device */
  double *d_b;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  double *d_x;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;

  /* used to overlap P2P communication with SpMV */
  cudaStream_t commstream;
  err = cudaStreamCreateWithFlags(&commstream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t xreadytosend, xreceived;
  err = cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(xreadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t preadytosend, preceived;
  err = cudaEventCreateWithFlags(&preadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(preadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&preceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;

  /* create cusparse matrix and vectors */
  cusparseDnVecDescr_t vecx, vecr, vecp, vect;
  err = cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecp, A->nownedrows, d_p, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  cusparseDnVecDescr_t vecxo, vecro, vecpo, vecto;
  if (commsize > 1)
  {
    err = cusparseCreateDnVec(
        &vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecpo, A->nborderrows + A->nghostrows, d_p + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecto, A->nborderrows + A->nghostrows, d_t + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }

  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, cusparse_spmv_alg, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
  err = cusparseSpMV_preprocess(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, cusparse_spmv_alg, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
#endif

  cusparseSpMatDescr_t matO;
  void *d_obuffer;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows,
        A->nborderrows + A->nghostrows, A->onpnzs, d_orowptr, d_ocolidx,
        d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, cusparse_spmv_alg, &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, cusparse_spmv_alg, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#endif
  }

  /* create timing events for profiling */
  acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0,
           nallreduce = 0, nhalo = 0;
  cudaEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
  tgemv = malloc(2 * (maxits + 1) * sizeof(*tgemv));
  if (!tgemv)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tgemv[i]);
  tdot = malloc(2 * maxits * sizeof(*tdot));
  if (!tdot)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * maxits; i++)
    cudaEventCreate(&tdot[i]);
  tnrm2 = malloc(2 * (maxits + 2) * sizeof(*tnrm2));
  if (!tnrm2)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&tnrm2[i]);
  taxpy = malloc(2 * (3 * maxits) * sizeof(*taxpy));
  if (!taxpy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (3 * maxits); i++)
    cudaEventCreate(&taxpy[i]);
  tcopy = malloc(2 * 2 * sizeof(*tcopy));
  if (!tcopy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * 2; i++)
    cudaEventCreate(&tcopy[i]);
  tallreduce = malloc(2 * (2 * maxits + 2) * sizeof(*tallreduce));
  if (!tallreduce)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (2 * maxits + 2); i++)
    cudaEventCreate(&tallreduce[i]);
  thalo = malloc(2 * (maxits + 1) * sizeof(*thalo));
  if (!thalo)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&thalo[i]);
#endif

  /* warmup iterations for dot/allreduce */
  for (int i = 0; i < warmup; i++)
  {
    cudaMemcpy(
        d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice);
    err = cublasDdot(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
        d_bnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = cublasDdot(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t,
        1, d_pdott);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
  }
  cudaMemcpy(
      d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
  cudaMemcpy(
      d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice);

  /* warmup iterations for halo exchange/SpMV */
  for (int i = 0; i < warmup; i++)
  {
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, xreadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
        d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
      err = cudaEventRecord(xreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, xreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
          vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
          d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cudaEventRecord(xreadytosend, stream);
      if (err)
        return ACG_ERR_CUDA;
    }

    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
      err = cudaEventRecord(preceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, preceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cudaEventRecord(preadytosend, stream);
      if (err)
        return ACG_ERR_CUDA;
    }
  }

  /* warmup iterations for axpy */
  for (int i = 0; i < warmup; i++)
  {
    err = acgsolvercuda_daxpy_alpha(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_zero, d_one, d_p,
        d_x);
    if (err)
      return err;
    err = acgsolvercuda_daypx_beta(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_one, d_p,
        d_r);
    if (err)
      return err;
  }

  /* warmup iterations for copy */
  for (int i = 0; i < warmup; i++)
  {
    err = cublasDcopy(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
    err = cublasDcopy(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p,
        1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
  }

  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;
  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  gettime(&t0);

  /* compute right-hand side norm */
  double bnrm2sqr;
  acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
  err = cublasDdot(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
      d_bnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
  nnrm2++;
  cg->nnrm2++;
  cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
  cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
  if (commsize > 1)
  {
    acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
        errcode);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    nallreduce++;
    cg->nallreduce++;
    cg->Ballreduce += sizeof(bnrm2sqr);
  }
  err = cudaMemcpy(
      &bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* /\* compute norm of initial guess *\/ */
  /* if (diffatol > 0 || diffrtol > 0) { */
  /*     gettime(&tnrm20); */
  /*     double x0nrm2sqr; */
  /*     err = acgvector_dnrm2sqr(x, &x0nrm2sqr, &cg->nflops, &cg->Bnrm2); */
  /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
   */
  /*     gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
   */
  /*     gettime(&tallreduce0); */
  /*     err = MPI_Allreduce(MPI_IN_PLACE, &x0nrm2sqr, 1, MPI_DOUBLE, MPI_SUM,
   * comm->mpicomm); */
  /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *errcode =
   * err; return ACG_ERR_MPI; } */
  /*     cg->Ballreduce += sizeof(x0nrm2sqr); */
  /*     gettime(&tallreduce1); cg->nallreduce++; cg->tallreduce +=
   * elapsed(tallreduce0,tallreduce1); */
  /*     cg->x0nrm2 = sqrt(x0nrm2sqr); */
  /*     diffrtol *= cg->x0nrm2; */
  /* } */

  /* compute initial residual, r₀ = b-A*x₀ */
  acgEventRecord(tcopy[2 * ncopy + 0], 0);
  err = cublasDcopy(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tcopy[2 * ncopy + 1], 0);
  ncopy++;
  cg->ncopy++;
  cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

  if (commsize > 1)
  {
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  if (commsize > 1)
  {
    acgEventRecord(thalo[2 * nhalo + 0], 0);
    err = acghalo_exchange_cuda_end(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 1], 0);
    nhalo++;
    cg->nhalo++;
    cg->Bhalo += cg->halo->sendsize * sizeof(*x->x);
    cg->nhalomsgs += cg->halo->nrecipients;
    err = cudaEventRecord(xreceived, commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(stream, xreceived, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 1], 0);
  ngemv++;
  cg->ngemv++;
  cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
  cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

  /* compute initial search direction: p = r₀ */
  acgEventRecord(tcopy[2 * ncopy + 0], 0);
  err = cublasDcopy(
      cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p, 1);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tcopy[2 * ncopy + 1], 0);
  ncopy++;
  cg->ncopy++;
  cg->Bcopy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->r.x));
  err = cudaEventRecord(preadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;

  /* compute initial residual norm */
  acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
  err = cublasDdot(
      cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1,
      d_rnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
  nnrm2++;
  cg->nnrm2++;
  cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
  cg->Bnrm2 +=
      (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);
  if (commsize > 1)
  {
    acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
        errcode);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    nallreduce++;
    cg->nallreduce++;
    cg->Ballreduce += sizeof(*rnrm2sqr);
  }
  err = cudaMemcpy(
      rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->rnrm2 = cg->r0nrm2 = sqrt(*rnrm2sqr);
  residualrtol *= cg->r0nrm2;

  /* initial convergence test */
  if ((residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_SUCCESS;
  }

  /* iterative solver loop */
  for (int k = 0; k < maxits; k++)
  {
    /* compute t = Ap */
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      acgEventRecord(thalo[2 * nhalo + 0], 0);
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(thalo[2 * nhalo + 1], 0);
      nhalo++;
      cg->nhalo++;
      cg->Bhalo += cg->halo->sendsize * sizeof(*cg->p.x);
      cg->nhalomsgs += cg->halo->nrecipients;
      err = cudaEventRecord(preceived, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(stream, preceived, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->t.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->p.num_nonzeros * sizeof(*cg->p.x);

    /* compute (p,Ap) */
    acgEventRecord(tdot[2 * ndot + 0], 0);
    err = cublasDdot(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t,
        1, d_pdott);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    acgEventRecord(tdot[2 * ndot + 1], 0);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));
    if (commsize > 1)
    {
      acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
#ifndef HOST_ALLREDUCE
      err = acgcomm_allreduce(
          ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          errcode);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
#else
      double pdott;
      cudaMemcpy(
          &pdott, d_pdott, sizeof(*d_pdott), cudaMemcpyDeviceToHost);
      MPI_Allreduce(
          MPI_IN_PLACE, &pdott, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
      cudaMemcpy(
          d_pdott, &pdott, sizeof(*d_pdott), cudaMemcpyHostToDevice);
#endif
      acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += sizeof(*d_pdott);
    }
    err = cudaMemcpyAsync(
        d_rnrm2sqr_prev, d_rnrm2sqr, sizeof(*d_rnrm2sqr),
        cudaMemcpyDeviceToDevice, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }

    /* update residual, r = -αt + r */
    acgEventRecord(taxpy[2 * naxpy + 0], 0);
#ifndef NO_FUSED_KERNELS
    err = acgsolvercuda_daxpy_minus_alpha(
        cg->t.num_nonzeros - cg->t.num_ghost_nonzeros, d_rnrm2sqr, d_pdott,
        d_t, d_r);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
#else
    err = acgsolvercuda_alpha(d_alpha, d_minus_alpha, d_rnrm2sqr, d_pdott);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    err = cublasDaxpy(
        cublas, cg->t.num_nonzeros - cg->t.num_ghost_nonzeros,
        d_minus_alpha, d_t, 1, d_r, 1);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
#endif
    acgEventRecord(taxpy[2 * naxpy + 1], 0);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 2 * (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros);
    cg->Baxpy += (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * (sizeof(*cg->t.x) + sizeof(*cg->r.x));

    /* compute residual norm */
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
    cg->Bnrm2 +=
        (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);
#ifndef HOST_ALLREDUCE
    if (commsize > 1)
    {
      acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
      err = acgcomm_allreduce(
          ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          errcode);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += sizeof(*rnrm2sqr);
    }
    err = cudaEventRecord(rnrm2sqrready, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaMemcpyAsync(
        rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost,
        copystream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    /* err = cudaEventRecord(rnrm2sqrreceived, copystream); */
    /* if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return
     * ACG_ERR_CUDA; } */
#else
    if (commsize > 1)
    {
      acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
      cudaMemcpy(
          rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr),
          cudaMemcpyDeviceToHost);
      MPI_Allreduce(
          MPI_IN_PLACE, rnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
      err = cudaMemcpyAsync(
          d_rnrm2sqr, rnrm2sqr, sizeof(*d_rnrm2sqr),
          cudaMemcpyHostToDevice, copystream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += sizeof(*rnrm2sqr);
    }
    else
    {
      err = cudaEventRecord(rnrm2sqrready, stream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaMemcpyAsync(
          rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr),
          cudaMemcpyDeviceToHost, copystream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
    }
#endif

    /* update solution, x = αp + x, where α = (r,r)/(p,t) */
    acgEventRecord(taxpy[2 * naxpy + 0], 0);
#ifndef NO_FUSED_KERNELS
    err = acgsolvercuda_daxpy_alpha(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_rnrm2sqr_prev,
        d_pdott, d_p, d_x);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
#else
    err = cublasDaxpy(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_alpha, d_p,
        1, d_x, 1);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
#endif
    acgEventRecord(taxpy[2 * naxpy + 1], 0);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Baxpy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*x->x));

    /* update search direction, p = βp + r, where β = (rₖ,rₖ)/(rₖ₋₁,rₖₖ₋₁)
     */
    acgEventRecord(taxpy[2 * naxpy + 0], 0);
#ifndef NO_FUSED_KERNELS
    err = acgsolvercuda_daypx_beta(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_rnrm2sqr,
        d_rnrm2sqr_prev, d_p, d_r);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
#else
    err = acgsolvercuda_beta(d_beta, d_rnrm2sqr, d_rnrm2sqr_prev);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    err = cublasDscal(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_beta, d_p,
        1);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    err = cublasDaxpy(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_r,
        1, d_p, 1);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
#endif
    acgEventRecord(taxpy[2 * naxpy + 1], 0);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Baxpy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->r.x));
    err = cudaEventRecord(preadytosend, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }

    /* convergence tests */
    /* cudaEventSynchronize(rnrm2sqrreceived); */
    cudaStreamSynchronize(copystream);
    cg->rnrm2 = sqrt(*rnrm2sqr);
    if ((diffatol > 0 && cg->dxnrm2 < diffatol) || (diffrtol > 0 && cg->dxnrm2 < diffrtol) || (residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      cudaStreamSynchronize(stream);
      cg->ntotaliterations++;
      cg->niterations++;
      converged = true;
      break;
    }
    cg->ntotaliterations++;
    cg->niterations++;
  }
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

#if defined(ACG_ENABLE_PROFILING)
  /* record profiling information */
  float t;
  for (acgidx_t i = 0; i < ngemv; i++)
  {
    cudaEventSynchronize(tgemv[2 * i + 1]);
    cudaEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
    cg->tgemv += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ndot; i++)
  {
    cudaEventSynchronize(tdot[2 * i + 1]);
    cudaEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
    cg->tdot += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nnrm2; i++)
  {
    cudaEventSynchronize(tnrm2[2 * i + 1]);
    cudaEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
    cg->tnrm2 += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < naxpy; i++)
  {
    cudaEventSynchronize(taxpy[2 * i + 1]);
    cudaEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
    cg->taxpy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ncopy; i++)
  {
    cudaEventSynchronize(tcopy[2 * i + 1]);
    cudaEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
    cg->tcopy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nallreduce; i++)
  {
    cudaEventSynchronize(tallreduce[2 * i + 1]);
    cudaEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
    cg->tallreduce += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nhalo; i++)
  {
    cudaEventSynchronize(thalo[2 * i + 1]);
    cudaEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
    cg->thalo += 1.0e-3 * t;
  }
#endif

  /* copy solution back to host */
  err = cudaMemcpy(
      x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

  /* free cusparse matrix and vectors */
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecp);
  cusparseDestroyDnVec(vect);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecpo);
    cusparseDestroyDnVec(vecto);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
  cudaFree(d_x);
  cudaFree(d_b);
  cudaFreeHost(rnrm2sqr);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(copystream);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  /* check for CUDA errors */
  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  /* if the solver converged or the only stopping criteria is a
   * maximum number of iterations, then the solver succeeded */
  if (converged)
    return ACG_SUCCESS;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;

  /* otherwise, the solver failed to converge with the given number
   * of maximum iterations */
  return ACG_ERR_NOT_CONVERGED;
}

int acgsolvercuda_solve_preconditioned(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    int preconditioner,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    cusparseSpMVAlg_t cusparse_spmv_alg)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

  /* /\* If the stopping criterion is based on the difference in */
  /*  * solution from one iteration to the next, then allocate */
  /*  * additional storage for storing the difference. *\/ */
  /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
  /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO;
   */
  /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
  /* } */

  /* allocate extra vectors needed for pipelined CG */
  if (!cg->w)
  {
    cg->w = malloc(sizeof(*cg->w));
    if (!cg->w)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->w, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->q)
  {
    cg->q = malloc(sizeof(*cg->q));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->q, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->z)
  {
    cg->z = malloc(sizeof(*cg->z));
    if (!cg->z)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->z, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->m)
  {
    cg->m = malloc(sizeof(*cg->m));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->m, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_m, cg->q->num_nonzeros * sizeof(*cg->d_m));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->n)
  {
    cg->n = malloc(sizeof(*cg->n));
    if (!cg->n)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->n, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_n, cg->n->num_nonzeros * sizeof(*cg->d_n));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->u)
  {
    cg->u = malloc(sizeof(*cg->u));
    if (!cg->u)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->u, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_u, cg->u->num_nonzeros * sizeof(*cg->d_u));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->y)
  {
    cg->y = malloc(sizeof(*cg->y));
    if (!cg->y)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->y, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_y, cg->y->num_nonzeros * sizeof(*cg->d_y));
    if (err)
      return ACG_ERR_CUDA;
  }

  cudaStream_t stream = 0;
  const struct acghalo *halo = cg->halo;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  double *d_rnrm2sqr = cg->d_rnrm2sqr;
  double *d_delta = cg->d_pdott;
  double *d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
  double *d_pdott = cg->d_pdott;
  double *d_alpha = cg->d_alpha;
  double *d_minus_alpha = cg->d_minus_alpha;
  double *d_beta = cg->d_beta;
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_r = cg->d_r;
  double *d_p = cg->d_p;
  double *d_t = cg->d_t;
  double *d_inf = cg->d_inf;
  double *d_z = cg->d_z;
  double *d_w = cg->d_w;
  double *d_m = cg->d_m;
  double *d_n = cg->d_n;
  double *d_q = cg->d_q;
  double *d_u = cg->d_u;
  double *d_y = cg->d_y;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;
  MPI_Request request;

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  cudaStream_t collective_stream;
  err = cudaStreamCreateWithFlags(&collective_stream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  double *rnrm2sqr;
  err = cudaMallocHost((void **)&rnrm2sqr, sizeof(*rnrm2sqr));
  if (err)
    return ACG_ERR_CUDA;

  double *rnrm2sqr_prev;
  err = cudaMallocHost((void **)&rnrm2sqr_prev, sizeof(*rnrm2sqr_prev));
  if (err)
    return ACG_ERR_CUDA;

  cudaStream_t copystream;
  err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t rnrm2sqrready;
  cudaEventCreateWithFlags(&rnrm2sqrready, cudaEventDisableTiming);
  /* cudaEvent_t rnrm2sqrreceived; */
  /* err = cudaEventCreateWithFlags(&rnrm2sqrreceived,
   * cudaEventDisableTiming); if (err) return ACG_ERR_CUDA; */

  /* copy right-hand side and initial guess to device */
  double *d_b;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  double *d_x;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;

  /* used to overlap P2P communication with SpMV */
  cudaStream_t commstream;
  err = cudaStreamCreateWithFlags(&commstream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t xreadytosend, xreceived;
  err = cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(xreadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t preadytosend, preceived, reduced;
  err = cudaEventCreateWithFlags(&preadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(preadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&preceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t mreadytosend, mreceived;
  err = cudaEventCreateWithFlags(&mreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&mreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&reduced, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t ureadytosend, ureceived;
  err = cudaEventCreateWithFlags(&ureadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  // err = cudaEventRecord(ureadytosend, stream);
  // if (err)
  //     return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&ureceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;

  cudaEvent_t dotEvent;
  err = cudaEventCreateWithFlags(&dotEvent, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;

  /* create cusparse matrix and vectors */
  cusparseDnVecDescr_t vecx, vecr, vecp, vect;
  err = cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecp, A->nownedrows, d_p, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  cusparseDnVecDescr_t vecxo, vecro, vecpo, vecto;
  if (commsize > 1)
  {
    err = cusparseCreateDnVec(
        &vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecpo, A->nborderrows + A->nghostrows, d_p + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecto, A->nborderrows + A->nghostrows, d_t + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }

  cusparseDnVecDescr_t vecz, vecn, vecm, vecw, vecy, vecu;
  err = cusparseCreateDnVec(&vecz, A->nownedrows, d_z, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  err = cusparseCreateDnVec(&vecm, A->nownedrows, d_m, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  err = cusparseCreateDnVec(&vecn, A->nownedrows, d_n, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  err = cusparseCreateDnVec(&vecw, A->nownedrows, d_w, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecy, A->nownedrows, d_y, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecu, A->nownedrows, d_u, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  cusparseDnVecDescr_t vecno, vecmo, vecwo, veczo, vecuo;
  if (commsize > 1)
  {
    err = cusparseCreateDnVec(
        &vecno, A->nborderrows + A->nghostrows, d_n + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecmo, A->nborderrows + A->nghostrows, d_m + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &veczo, A->nborderrows + A->nghostrows, d_z + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecuo, A->nborderrows + A->nghostrows, d_u + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }

  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;

#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
  err = cusparseSpMV_preprocess(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, cusparse_spmv_alg, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
#endif

  cusparseSpMatDescr_t matO;
  void *d_obuffer;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows,
        A->nborderrows + A->nghostrows, A->onpnzs, d_orowptr, d_ocolidx,
        d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, cusparse_spmv_alg, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#endif
  }

  /* create timing events for profiling */
  acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0,
           nallreduce = 0, nhalo = 0;
  cudaEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
  tgemv = malloc(2 * (maxits + 2) * sizeof(*tgemv));
  if (!tgemv)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tgemv[i]);
  tdot = malloc(2 * (maxits) * sizeof(*tdot));
  if (!tdot)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits); i++)
    cudaEventCreate(&tdot[i]);
  tnrm2 = malloc(2 * (maxits + 2) * sizeof(*tnrm2));
  if (!tnrm2)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&tnrm2[i]);
  taxpy = malloc(2 * maxits * sizeof(*taxpy));
  if (!taxpy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * maxits; i++)
    cudaEventCreate(&taxpy[i]);
  tcopy = malloc(2 * 2 * sizeof(*tcopy));
  if (!tcopy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * 2; i++)
    cudaEventCreate(&tcopy[i]);
  tallreduce = malloc(2 * (maxits + 1) * sizeof(*tallreduce));
  if (!tallreduce)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tallreduce[i]);
  thalo = malloc(2 * (maxits + 2) * sizeof(*thalo));
  if (!thalo)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&thalo[i]);
#endif

  /* warmup iterations for dot/allreduce */
  for (int i = 0; i < warmup; i++)
  {
    cudaMemcpy(
        d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice);
    err = cublasDdot(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
        d_bnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = cublasDdot(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t,
        1, d_pdott);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
  }
  cudaMemcpy(
      d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
  cudaMemcpy(
      d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice);

  /* warmup iterations for halo exchange/SpMV */
  for (int i = 0; i < warmup; i++)
  {
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, xreadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
        d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
      err = cudaEventRecord(xreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, xreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
          vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
          d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cudaEventRecord(xreadytosend, stream);
      if (err)
        return ACG_ERR_CUDA;
    }

    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
      err = cudaEventRecord(preceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, preceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cudaEventRecord(preadytosend, stream);
      if (err)
        return ACG_ERR_CUDA;
    }
  }

  /* warmup iterations for axpy */
  for (int i = 0; i < warmup; i++)
  {
    err = acgsolvercuda_daxpy_alpha(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_zero, d_one, d_p,
        d_x);
    if (err)
      return err;
    err = acgsolvercuda_daypx_beta(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_one, d_p,
        d_r);
    if (err)
      return err;
  }

  /* warmup iterations for copy */
  for (int i = 0; i < warmup; i++)
  {
    err = cublasDcopy(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
    err = cublasDcopy(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p,
        1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
  }

  /* set scalars to infinity (needed to produce correct results on
   * the first call to acgsolvercuda_pipelined_daxpy_fused) */
  err =
      cudaMemcpy(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev),
      cudaMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_CUDA;

  double *d_M_inv;
  double *h_M_inv;
  cusparseSpMatDescr_t matM_lower, matM_upper;
  cusparseSpSVDescr_t spSVDescrL, spSVDescrU;
  size_t bufferSizeL, bufferSizeU;
  void *d_bufferL, *d_bufferU;
  err = cusparseSpSV_createDescr(&spSVDescrU);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSpSV_createDescr(&spSVDescrL);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  /**
   * Preconditioner setup
   */
  if (preconditioner == 1)
  {
    err = cudaMalloc(&d_M_inv, (A->nprows) * sizeof(double));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    h_M_inv = (double *)malloc((A->nprows) * sizeof(double));
    if (!h_M_inv)
    {
      return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < (A->nprows); i++)
    {
      double diag = 0.0;
      for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
      {
        if (A->colidx[j] == i + A->rowidxbase)
        {
          diag += A->a[j];
        }
      }
      h_M_inv[i] = diag;
    }

    err = cudaMemcpy(d_M_inv, h_M_inv, (A->nprows) * sizeof(double), cudaMemcpyHostToDevice);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    free(h_M_inv);
  }
  else if (preconditioner == 2)
  {
    cusparseMatDescr_t matLU;
    acgidx_t *d_M_rowptr = d_rowptr;
    acgidx_t *d_M_colidx = d_colidx;
    double *d_M_values;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t diag_nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    err = cudaMalloc(
        &d_M_values, A->fnpnzs * sizeof(*d_M_values));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cudaMemcpy(
        d_M_values, d_a, A->fnpnzs * sizeof(*d_M_values),
        cudaMemcpyDeviceToDevice);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    // matM_lower
    err = cusparseCreateCsr(
        &matM_lower, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
        d_M_colidx, d_M_values, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower,
        sizeof(fill_lower));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    // matM_upper
    err = cusparseCreateCsr(
        &matM_upper, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
        d_M_colidx, d_M_values, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper,
        sizeof(fill_upper));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_upper, CUSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit,
        sizeof(diag_nonunit));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    // ILU factorization part
    csrilu02Info_t infoM = NULL;
    int bufferSizeLU = 0;
    void *d_bufferLU;
    err = cusparseCreateMatDescr(&matLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    err = cusparseCreateCsrilu02Info(&infoM);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    err = cusparseDcsrilu02_bufferSize(
        cusparse, A->nownedrows, A->fnpnzs, matLU,
        d_M_values, d_rowptr, d_colidx, infoM, &bufferSizeLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_bufferLU, bufferSizeLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    err = cusparseDcsrilu02_analysis(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    int structural_zero;
    err = cusparseXcsrilu02_zeroPivot(
        cusparse, infoM, &structural_zero);
    if (structural_zero >= 0)
    {
      fprintf(stderr, "ACG: structural zero at index %d\n", structural_zero);
      // return ACG_ERR_CUSPARSE;
    }
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     return ACG_ERR_CUSPARSE;
    // }

    err = cusparseDcsrilu02(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values,
        d_rowptr, d_colidx, infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    int numerical_zero;
    err = cusparseXcsrilu02_zeroPivot(
        cusparse, infoM, &numerical_zero);
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     return ACG_ERR_CUSPARSE;
    // }
    if (numerical_zero >= 0)
    {
      fprintf(stderr, "ACG: numerical zero at index %d\n", numerical_zero);
      // return ACG_ERR_CUSPARSE;
    }

    err = cusparseDestroyCsrilu02Info(infoM);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseDestroyMatDescr(matLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaFree(d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    // lower

    err = cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spSVDescrL, &bufferSizeL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_bufferL, bufferSizeL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL,
        d_bufferL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMemset(d_y, 0x0, A->nownedrows * sizeof(*d_y));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    // upper
    err = cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spSVDescrU, &bufferSizeU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_bufferU, bufferSizeU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrU,
        d_bufferU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMemset(d_u, 0x0, A->nownedrows * sizeof(*d_u));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
  }

  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;
  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  gettime(&t0);

  /* compute right-hand side norm */
  double bnrm2sqr;
  acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
  err = cublasDdot(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
      d_bnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
  nnrm2++;
  cg->nnrm2++;
  cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
  cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
  if (commsize > 1)
  {
    acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
        errcode);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    nallreduce++;
    cg->nallreduce++;
    cg->Ballreduce += sizeof(bnrm2sqr);
  }
  err = cudaMemcpy(
      &bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);
  //     cg->r0nrm2 = cg->bnrm2;
  //     cg->rnrm2 = *rnrm2sqr;
  //     *rnrm2sqr *= cg->bnrm2;

  /* /\* compute norm of initial guess *\/ */
  /* if (diffatol > 0 || diffrtol > 0) { */
  /*     gettime(&tnrm20); */
  /*     double x0nrm2sqr; */
  /*     err = acgvector_dnrm2sqr(x, &x0nrm2sqr, &cg->nflops, &cg->Bnrm2); */
  /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
   */
  /*     gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
   */
  /*     gettime(&tallreduce0); */
  /*     err = MPI_Allreduce(MPI_IN_PLACE, &x0nrm2sqr, 1, MPI_DOUBLE, MPI_SUM,
   * comm->mpicomm); */
  /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *errcode =
   * err; return ACG_ERR_MPI; } */
  /*     cg->Ballreduce += sizeof(x0nrm2sqr); */
  /*     gettime(&tallreduce1); cg->nallreduce++; cg->tallreduce +=
   * elapsed(tallreduce0,tallreduce1); */
  /*     cg->x0nrm2 = sqrt(x0nrm2sqr); */
  /*     diffrtol *= cg->x0nrm2; */
  /* } */

  /* compute initial residual, r₀ = b-A*x₀ */
  acgEventRecord(tcopy[2 * ncopy + 0], 0);
  err = cublasDcopy(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tcopy[2 * ncopy + 1], 0);
  ncopy++;
  cg->ncopy++;
  cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

  if (commsize > 1)
  {
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  if (commsize > 1)
  {
    acgEventRecord(thalo[2 * nhalo + 0], 0);
    err = acghalo_exchange_cuda_end(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 1], 0);
    nhalo++;
    cg->nhalo++;
    cg->Bhalo += cg->halo->sendsize * sizeof(*x->x);
    cg->nhalomsgs += cg->halo->nrecipients;
    err = cudaEventRecord(xreceived, commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(stream, xreceived, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 1], 0);
  ngemv++;
  cg->ngemv++;
  cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
  cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

  /**
   * Preconditioner
   */
  if (preconditioner == 0)
  {
    err = cudaMemcpy(d_u, d_r, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r), cudaMemcpyDeviceToDevice);

    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
  }
  else if (preconditioner == 1)
  {
    acgsolvercuda_apply_jacobi_preconditioner((A->nprows - A->nghostrows), d_M_inv, d_r, d_u, numSMs, stream);
  }
  else if (preconditioner == 2)
  {

    err = cusparseSpSV_solve(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    err = cusparseSpSV_solve(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
  /* compute (r, u) */
  acgEventRecord(tdot[2 * ndot + 0], 0);
  err = cublasDdot(
      cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_u,
      1, d_rnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tdot[2 * ndot + 1], 0);
  ndot++;
  cg->ndot++;
  cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
  cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) *
              (sizeof(*cg->p.x) + sizeof(*cg->t.x));

  if (commsize > 1)
  {
    if (comm->type == acgcomm_nccl || comm->type == acgcomm_nvshmem || comm->type == acgcomm_nvshmem_split)
    {

      acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
      err = acgcomm_allreduce(
          ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM,
          stream, comm, errcode);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    }
    else if (comm->type == acgcomm_mpi)
    {
      cudaStreamSynchronize(stream);
      /**
       * TODO: will change the following part. That will go to the
       * acgcomm_allreduce() method once I find a clear way of doing it.
       **/
      err = MPI_Iallreduce(MPI_IN_PLACE, d_rnrm2sqr, 1, MPI_DOUBLE,
                           MPI_SUM, comm->mpicomm, &request);
      if (err)
      {
        if (errcode)
          *errcode = err;
        ACG_ERR_MPI;
      }
    }

    nallreduce++;
    cg->nallreduce++;
    cg->Ballreduce += sizeof(*d_pdott);
  }

  err = cudaMemcpy(
      rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->rnrm2 = cg->r0nrm2 = sqrt(*rnrm2sqr);
  residualrtol *= cg->r0nrm2;

  err = cublasDcopy(
      cublas, cg->u->num_nonzeros - cg->u->num_ghost_nonzeros,
      d_u, 1, d_p, 1);
  if (err)
  {
    if (errcode)
    {
      *errcode = err;
    }
    return ACG_ERR_CUBLAS;
  }

  err = cudaEventRecord(preadytosend, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUDA;
  }
  /* iterative solver loop */
  for (int k = 0; k < maxits; k++)
  {
    // SpMV
    /* compute t = Ap */
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      acgEventRecord(thalo[2 * nhalo + 0], 0);
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(thalo[2 * nhalo + 1], 0);
      nhalo++;
      cg->nhalo++;
      cg->Bhalo += cg->halo->sendsize * sizeof(*cg->p.x);
      cg->nhalomsgs += cg->halo->nrecipients;
      /**
       * TODO: mrecevied -> preceived
       **/
      err = cudaEventRecord(mreceived, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(stream, mreceived, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->t.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->p.num_nonzeros * sizeof(*cg->p.x);

    // alpha = (r, u)/(t, p) -> gamma/delta

    // dot (t, p)
    acgEventRecord(tdot[2 * ndot + 0], 0);
    err = cublasDdot(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_t, 1, d_p,
        1, d_delta);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    acgEventRecord(tdot[2 * ndot + 1], 0);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

    if (commsize > 1)
    {
      if (comm->type == acgcomm_nccl || comm->type == acgcomm_nvshmem || comm->type == acgcomm_nvshmem_split)
      {

        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce(
            ACG_IN_PLACE, d_delta, 1, ACG_DOUBLE, ACG_SUM,
            stream, comm, errcode);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return err;
        }
        acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
      }
      else if (comm->type == acgcomm_mpi)
      {
        cudaStreamSynchronize(stream);
        /**
         * TODO: will change the following part. That will go to the
         * acgcomm_allreduce() method once I find a clear way of doing it.
         **/
        err = MPI_Iallreduce(MPI_IN_PLACE, d_delta, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm, &request);
        if (err)
        {
          if (errcode)
            *errcode = err;
          ACG_ERR_MPI;
        }
      }

      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += sizeof(*d_pdott);
    }

    acgEventRecord(taxpy[2 * naxpy + 0], 0);
    acgsolvercuda_preconditioned_daxpy_fused(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_rnrm2sqr,
        d_rnrm2sqr_prev, d_delta, d_p, d_r, d_t, d_x, stream);

    acgEventRecord(taxpy[2 * naxpy + 1], 0);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 4 * (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros);
    cg->Baxpy += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*cg->p.x);

    // preconditioner u = M^{-1}r
    if (preconditioner == 0)
    {
      err = cudaMemcpy(
          d_u, d_r,
          (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r),
          cudaMemcpyDeviceToDevice);

      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUDA;
      }
    }
    else if (preconditioner == 1)
    {
      acgsolvercuda_apply_jacobi_preconditioner((A->nprows - A->nghostrows),
                                                d_M_inv, d_r, d_u,
                                                numSMs, stream);
    }
    else if (preconditioner == 2)
    {

      // err = cudaMemset(d_y, 0x0, A->nownedrows * sizeof(*d_y));
      // if (err)
      // {
      //     if (errcode)
      //         *errcode = err;
      //     return ACG_ERR_CUDA;
      // }
      // err = cudaMemset(d_u, 0x0, A->nownedrows * sizeof(*d_y));
      // if (err)
      // {
      //     if (errcode)
      //         *errcode = err;
      //     return ACG_ERR_CUDA;
      // }

      err = cusparseSpSV_solve(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
          vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cusparseSpSV_solve(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
          vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
    }

    // dot (r, u) -> gamma
    acgEventRecord(tdot[2 * ndot + 0], 0);
    err = cublasDdot(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_u,
        1, d_rnrm2sqr);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    acgEventRecord(tdot[2 * ndot + 1], 0);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

    if (commsize > 1)
    {
      if (comm->type == acgcomm_nccl || comm->type == acgcomm_nvshmem || comm->type == acgcomm_nvshmem_split)
      {

        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce(
            ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM,
            stream, comm, errcode);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return err;
        }
        acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
      }
      else if (comm->type == acgcomm_mpi)
      {
        cudaStreamSynchronize(stream);
        /**
         * TODO: will change the following part. That will go to the
         * acgcomm_allreduce() method once I find a clear way of doing it.
         **/
        err = MPI_Iallreduce(MPI_IN_PLACE, d_rnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm, &request);
        if (err)
        {
          if (errcode)
            *errcode = err;
          ACG_ERR_MPI;
        }
      }

      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += sizeof(*d_pdott);
    }

    err = cudaEventRecord(rnrm2sqrready, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }

    err = cudaMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr),
                          cudaMemcpyDeviceToHost, copystream);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    /* update search direction, p = βp + r, where β = (rₖ,rₖ)/(rₖ₋₁,rₖₖ₋₁)
     */
    acgEventRecord(taxpy[2 * naxpy + 0], 0);
    err = acgsolvercuda_daypx_beta(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_rnrm2sqr,
        d_rnrm2sqr_prev, d_p, d_u);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(taxpy[2 * naxpy + 1], 0);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Baxpy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->r.x));
    err = cudaEventRecord(preadytosend, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }

    /* convergence tests */
    /* cudaEventSynchronize(rnrm2sqrreceived); */
    cudaStreamSynchronize(copystream);
    cg->rnrm2 = sqrt(*rnrm2sqr);
    if ((diffatol > 0 && cg->dxnrm2 < diffatol) || (diffrtol > 0 && cg->dxnrm2 < diffrtol) || (residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      cudaStreamSynchronize(stream);
      cg->ntotaliterations++;
      cg->niterations++;
      converged = true;
      break;
    }
    cg->ntotaliterations++;
    cg->niterations++;
  }
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

#if defined(ACG_ENABLE_PROFILING)
  /* record profiling information */
  float t;
  for (acgidx_t i = 0; i < ngemv; i++)
  {
    cudaEventSynchronize(tgemv[2 * i + 1]);
    cudaEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
    cg->tgemv += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ndot; i++)
  {
    cudaEventSynchronize(tdot[2 * i + 1]);
    cudaEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
    cg->tdot += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nnrm2; i++)
  {
    cudaEventSynchronize(tnrm2[2 * i + 1]);
    cudaEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
    cg->tnrm2 += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < naxpy; i++)
  {
    cudaEventSynchronize(taxpy[2 * i + 1]);
    cudaEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
    cg->taxpy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ncopy; i++)
  {
    cudaEventSynchronize(tcopy[2 * i + 1]);
    cudaEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
    cg->tcopy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nallreduce; i++)
  {
    cudaEventSynchronize(tallreduce[2 * i + 1]);
    cudaEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
    cg->tallreduce += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nhalo; i++)
  {
    cudaEventSynchronize(thalo[2 * i + 1]);
    cudaEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
    cg->thalo += 1.0e-3 * t;
  }
#endif

  /* copy solution back to host */
  err = cudaMemcpy(
      x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

  /* free cusparse matrix and vectors */
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecp);
  cusparseDestroyDnVec(vect);
  cusparseDestroyDnVec(vecw);
  cusparseDestroyDnVec(vecz);
  cusparseDestroyDnVec(vecn);
  cusparseDestroyDnVec(vecm);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecpo);
    cusparseDestroyDnVec(vecto);
    cusparseDestroyDnVec(vecno);
    cusparseDestroyDnVec(vecmo);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_z);
  cudaFree(d_w);
  cudaFree(d_n);
  cudaFree(d_m);
  cudaFree(d_q);
  cudaFree(d_u);
  //    cudaFree(d_merged_dots);
  cudaFreeHost(rnrm2sqr);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(copystream);
  cudaStreamDestroy(collective_stream);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  /* check for CUDA errors */
  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  /* if the solver converged or the only stopping criteria is a
   * maximum number of iterations, then the solver succeeded */
  if (converged)
    return ACG_SUCCESS;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;

  /* otherwise, the solver failed to converge with the given number
   * of maximum iterations */
  return ACG_ERR_NOT_CONVERGED;
}

int acgsolvercuda_solve_pipelined_preconditioned(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    int preconditioner,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

#if defined(ACG_NOCOMM)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE) && defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE)
  int nocomm_allreduce = 1, nocomm_p2p = 0;
#elif defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 0, nocomm_p2p = 1;
#else
  int nocomm_allreduce = 0, nocomm_p2p = 0;
#endif

  /* /\* If the stopping criterion is based on the difference in */
  /*  * solution from one iteration to the next, then allocate */
  /*  * additional storage for storing the difference. *\/ */
  /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
  /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO;
   */
  /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
  /* } */

  /* allocate extra vectors needed for pipelined CG */
  if (!cg->w)
  {
    cg->w = malloc(sizeof(*cg->w));
    if (!cg->w)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->w, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->q)
  {
    cg->q = malloc(sizeof(*cg->q));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->q, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->z)
  {
    cg->z = malloc(sizeof(*cg->z));
    if (!cg->z)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->z, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->m)
  {
    cg->m = malloc(sizeof(*cg->m));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->m, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_m, cg->q->num_nonzeros * sizeof(*cg->d_m));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->n)
  {
    cg->n = malloc(sizeof(*cg->n));
    if (!cg->n)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->n, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_n, cg->n->num_nonzeros * sizeof(*cg->d_n));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->u)
  {
    cg->u = malloc(sizeof(*cg->u));
    if (!cg->u)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->u, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_u, cg->u->num_nonzeros * sizeof(*cg->d_u));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->y)
  {
    cg->y = malloc(sizeof(*cg->y));
    if (!cg->y)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->y, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_y, cg->y->num_nonzeros * sizeof(*cg->d_y));
    if (err)
      return ACG_ERR_CUDA;
  }

  const struct acghalo *halo = cg->halo;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  double *d_rnrm2sqr = &cg->d_rnrm2sqr[0];
  double *d_delta = &cg->d_rnrm2sqr[1];
  double *d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
  double *d_pdott = cg->d_pdott;
  double *d_alpha = cg->d_alpha;
  double *d_minus_alpha = cg->d_minus_alpha;
  double *d_beta = cg->d_beta;
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_r = cg->d_r;
  double *d_p = cg->d_p;
  double *d_t = cg->d_t;
  double *d_inf = cg->d_inf;
  double *d_z = cg->d_z;
  double *d_w = cg->d_w;
  double *d_m = cg->d_m;
  double *d_n = cg->d_n;
  double *d_q = cg->d_q;
  double *d_u = cg->d_u;
  double *d_y = cg->d_y;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;
  MPI_Request request;

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  int leastPriority, greatestPriority;
  err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_CUDA;

  cudaStream_t stream;
  err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_CUDA;

  cudaStream_t collective_stream;
  err = cudaStreamCreateWithPriority(&collective_stream, cudaStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_CUDA;

  cudaStream_t commstream;
  err = cudaStreamCreateWithPriority(&commstream, cudaStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_CUDA;

  acgSetStreamName(commstream, "P2P");
  acgSetStreamName(collective_stream, "Allreduce");
  acgSetStreamName(stream, "Compute");

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  double *rnrm2sqr;
  err = cudaMallocHost((void **)&rnrm2sqr, sizeof(*rnrm2sqr));
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t copystream;
  err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t rnrm2sqrready;
  cudaEventCreateWithFlags(&rnrm2sqrready, cudaEventDisableTiming);
  /* cudaEvent_t rnrm2sqrreceived; */
  /* err = cudaEventCreateWithFlags(&rnrm2sqrreceived,
   * cudaEventDisableTiming); if (err) return ACG_ERR_CUDA; */

  /* copy right-hand side and initial guess to device */
  double *d_b;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  double *d_x;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;

  cudaEvent_t xreadytosend, xreceived;
  err = cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(xreadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t preadytosend, preceived, reduced;
  err = cudaEventCreateWithFlags(&preadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(preadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&preceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t mreadytosend, mreceived;
  err = cudaEventCreateWithFlags(&mreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&mreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&reduced, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t ureadytosend, ureceived;
  err = cudaEventCreateWithFlags(&ureadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  // err = cudaEventRecord(ureadytosend, stream);
  // if (err)
  //     return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&ureceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;

  cudaEvent_t dotEvent;
  err = cudaEventCreateWithFlags(&dotEvent, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
#if defined(ACG_USE_CUSPARSE)
  /* create cusparse matrix and vectors */
  cusparseDnVecDescr_t vecx, vecr, vecp, vect;
  err = cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecp, A->nownedrows, d_p, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  cusparseDnVecDescr_t vecxo, vecro, vecpo, vecto;
  if (commsize > 1)
  {
    err = cusparseCreateDnVec(
        &vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecpo, A->nborderrows + A->nghostrows, d_p + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecto, A->nborderrows + A->nghostrows, d_t + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }

  cusparseDnVecDescr_t vecz, vecn, vecm, vecw, vecy, vecu;
  err = cusparseCreateDnVec(&vecz, A->nownedrows, d_z, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  err = cusparseCreateDnVec(&vecm, A->nownedrows, d_m, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  err = cusparseCreateDnVec(&vecn, A->nownedrows, d_n, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  err = cusparseCreateDnVec(&vecw, A->nownedrows, d_w, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecy, A->nownedrows, d_y, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecu, A->nownedrows, d_u, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  cusparseDnVecDescr_t vecno, vecmo, vecwo, veczo, vecuo;
  if (commsize > 1)
  {
    err = cusparseCreateDnVec(
        &vecno, A->nborderrows + A->nghostrows, d_n + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecmo, A->nborderrows + A->nghostrows, d_m + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &veczo, A->nborderrows + A->nghostrows, d_z + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecuo, A->nborderrows + A->nghostrows, d_u + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;

  cusparseSpMatDescr_t matO;
  void *d_obuffer;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows,
        A->nborderrows + A->nghostrows, A->onpnzs, d_orowptr, d_ocolidx,
        d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
  }
#else
  /* setup for merge-based SpMV */
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows = ntasks / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = cudaMalloc((void **)&d_startrows, nstartrows * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_CUDA;

  err = acgsolvercuda_csrgemv_merge_init((A->nprows - A->nghostrows), d_rowptr, nstartrows, d_startrows, stream);
  if (err)
    return err;

  err = cudaStreamSynchronize(stream);
  if (err)
    return ACG_ERR_CUDA;
#endif

  /* create timing events for profiling */
  acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0,
           nallreduce = 0, nhalo = 0, nprecond = 0, ngemv_o = 0;
  cudaEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo, *tprecond, *tgemv_o;
#if defined(ACG_ENABLE_PROFILING)
  tgemv = malloc(2 * (maxits + 2) * sizeof(*tgemv));
  if (!tgemv)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&tgemv[i]);
  tgemv_o = malloc(2 * (maxits + 2) * sizeof(*tgemv_o));
  if (!tgemv)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&tgemv_o[i]);
  tdot = malloc(2 * (2 * maxits) * sizeof(*tdot));
  if (!tdot)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (2 * maxits); i++)
    cudaEventCreate(&tdot[i]);
  tnrm2 = malloc(2 * (maxits + 2) * sizeof(*tnrm2));
  if (!tnrm2)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&tnrm2[i]);
  taxpy = malloc(2 * maxits * sizeof(*taxpy));
  if (!taxpy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * maxits; i++)
    cudaEventCreate(&taxpy[i]);
  tcopy = malloc(2 * 2 * sizeof(*tcopy));
  if (!tcopy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * 2; i++)
    cudaEventCreate(&tcopy[i]);
  tallreduce = malloc(2 * (maxits + 1) * sizeof(*tallreduce));
  if (!tallreduce)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tallreduce[i]);
  thalo = malloc(4 * (2 * maxits + 2) * sizeof(*thalo));
  if (!thalo)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 4 * (2 * maxits + 2); i++)
    cudaEventCreate(&thalo[i]);
  tprecond = malloc(2 * (maxits + 1) * sizeof(*tprecond));
  if (!tprecond)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tprecond[i]);
#endif

  /* warmup iterations for dot/allreduce */
  for (int i = 0; i < warmup; i++)
  {
    cudaMemcpyAsync(
        d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(
        d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice, stream);
#if defined(ACG_USE_CUBLAS)
    err = cublasDdot(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
        d_bnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
#else
    err = acgsolvercuda_ddot(
        b->num_nonzeros - b->num_ghost_nonzeros, d_b, d_b, d_bnrm2sqr, stream);
    if (err)
      return err;
#endif

    if (commsize > 1 && !nocomm_allreduce)
    {
      if (comm->type == acgcomm_nccl_split)
      {
        cudaEventRecord(dotEvent, stream);
        cudaStreamWaitEvent(collective_stream, dotEvent, 0);
        acgcomm_allreduce(
            ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, collective_stream, comm,
            NULL);
      }
      else if (comm->type == acgcomm_nccl)
      {
        acgcomm_allreduce(
            ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
            NULL);
      }
    }
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1 && !nocomm_allreduce)
    {
      if (comm->type == acgcomm_nccl_split)
      {
        cudaEventRecord(dotEvent, stream);
        cudaStreamWaitEvent(collective_stream, dotEvent, 0);
        acgcomm_allreduce(
            ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, collective_stream, comm,
            NULL);
      }
      else if (comm->type == acgcomm_nccl)
      {
        acgcomm_allreduce(
            ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
            NULL);
      }
    }
#if defined(ACG_USE_CUBLAS)
    err = cublasDdot(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t,
        1, d_pdott);
    if (err)
      return ACG_ERR_CUBLAS;
#else
    err = acgsolvercuda_ddot(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, d_t, d_pdott, stream);
    if (err)
      return err;
#endif

    if (commsize > 1 && !nocomm_allreduce)
    {
      if (comm->type == acgcomm_nccl_split)
      {
        cudaEventRecord(dotEvent, stream);
        cudaStreamWaitEvent(collective_stream, dotEvent, 0);
        acgcomm_allreduce(
            ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, collective_stream, comm,
            NULL);
      }
      else if (comm->type == acgcomm_nccl)
      {
        acgcomm_allreduce(
            ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
            NULL);
      }
    }
  }
  cudaMemcpyAsync(
      d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(
      d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice, stream);

  /* warmup iterations for halo exchange/SpMV */
  for (int i = 0; i < warmup; i++)
  {
    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaStreamWaitEvent(commstream, xreadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
        d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1,
        nstartrows, d_startrows, numSMs, stream);
    if (err)
      return err;
#endif
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
            commstream);
        if (err)
          return err;
        err = cudaEventRecord(xreceived, commstream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(stream, xreceived, 0);
        if (err)
          return ACG_ERR_CUDA;
      }
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
          vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
          d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(A->nborderrows, d_r + A->borderrowoffset, d_x + A->borderrowoffset,
                         d_orowptr, d_ocolidx, d_oa, -1.0, numSMs, stream);
      if (err)
        return err;
#endif
      if (!nocomm_p2p)
      {
        err = cudaEventRecord(xreadytosend, stream);
        if (err)
          return ACG_ERR_CUDA;
      }
    }

    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_t, d_p, d_rowptr, d_colidx, d_a, 1.0, 0,
        nstartrows, d_startrows, numSMs, stream);
    if (err)
      return err;
#endif
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
            cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
            commstream);
        if (err)
          return err;
        err = cudaEventRecord(preceived, commstream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(stream, preceived, 0);
        if (err)
          return ACG_ERR_CUDA;
      }
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(A->nborderrows, d_t + A->borderrowoffset, d_p + A->borderrowoffset,
                         d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
      if (err)
        return err;
#endif
      if (!nocomm_p2p)
      {
        err = cudaEventRecord(preadytosend, stream);
        if (err)
          return ACG_ERR_CUDA;
      }
    }
  }

  /* warmup iterations for axpy */
  for (int i = 0; i < warmup; i++)
  {
    err = acgsolvercuda_daxpy_alpha(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_zero, d_one, d_p,
        d_x);
    if (err)
      return err;
    err = acgsolvercuda_daypx_beta(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_one, d_p,
        d_r);
    if (err)
      return err;
  }

  /* warmup iterations for copy */
  for (int i = 0; i < warmup; i++)
  {
#if defined(ACG_USE_CUBLAS)
    err = cublasDcopy(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
    err = cublasDcopy(
        cublas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p,
        1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
#else
    err = acgsolvercuda_dcopy(
        b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
    if (err)
      return err;
    err = acgsolvercuda_dcopy(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, d_r, stream);
    if (err)
      return err;
#endif
  }

  /* set scalars to infinity (needed to produce correct results on
   * the first call to acgsolvercuda_pipelined_daxpy_fused) */
  err =
      cudaMemcpyAsync(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev),
      cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;

  double *d_M_inv, *h_M_inv;
  cusparseSpMatDescr_t matM_lower, matM_upper;
  size_t bufferSizeL, bufferSizeU;
  void *d_bufferL, *d_bufferU;
  cusparseSpSVDescr_t spSVDescrL, spSVDescrU;
  err = cusparseSpSV_createDescr(&spSVDescrU);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSpSV_createDescr(&spSVDescrL);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  /**
   * Preconditioner preparation
   */

  if (preconditioner == 1)
  {
    err = cudaMalloc(&d_M_inv, (A->nprows) * sizeof(double));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    h_M_inv = (double *)malloc((A->nprows) * sizeof(double));
    if (!h_M_inv)
    {
      return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < (A->nprows); i++)
    {
      double diag = 0.0;
      for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
      {
        if (A->colidx[j] == i + A->rowidxbase)
        {
          diag += A->a[j];
        }
      }
      h_M_inv[i] = diag;
    }

    err = cudaMemcpyAsync(d_M_inv, h_M_inv, (A->nprows) * sizeof(double), cudaMemcpyHostToDevice, stream);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cudaStreamSynchronize(stream); /* ‘h_M_inv’ is freed below */
    if (err)
      return ACG_ERR_CUDA;
    free(h_M_inv);
  }
  else if (preconditioner == 2)
  {
    cusparseMatDescr_t matLU;
    acgidx_t *d_M_rowptr = d_rowptr;
    acgidx_t *d_M_colidx = d_colidx;
    double *d_M_values;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t diag_nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    err = cudaMalloc(
        &d_M_values, A->fnpnzs * sizeof(*d_M_values));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cudaMemcpyAsync(
        d_M_values, d_a, A->fnpnzs * sizeof(*d_M_values),
        cudaMemcpyDeviceToDevice, stream);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    // matM_lower
    err = cusparseCreateCsr(
        &matM_lower, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
        d_M_colidx, d_M_values, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower,
        sizeof(fill_lower));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    // matM_upper
    err = cusparseCreateCsr(
        &matM_upper, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
        d_M_colidx, d_M_values, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper,
        sizeof(fill_upper));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSpMatSetAttribute(
        matM_upper, CUSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit,
        sizeof(diag_nonunit));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    // ILU factorization part
    csrilu02Info_t infoM = NULL;
    int bufferSizeLU = 0;
    void *d_bufferLU;
    err = cusparseCreateMatDescr(&matLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    err = cusparseCreateCsrilu02Info(&infoM);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    err = cusparseDcsrilu02_bufferSize(
        cusparse, A->nownedrows, A->fnpnzs, matLU,
        d_M_values, d_rowptr, d_colidx, infoM, &bufferSizeLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_bufferLU, bufferSizeLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }

    err = cusparseDcsrilu02_analysis(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    int structural_zero;
    err = cusparseXcsrilu02_zeroPivot(
        cusparse, infoM, &structural_zero);
    if (structural_zero >= 0)
    {
      fprintf(stderr, "ACG: structural zero at index %d\n", structural_zero);
      // return ACG_ERR_CUSPARSE;
    }
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     return ACG_ERR_CUSPARSE;
    // }

    err = cusparseDcsrilu02(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values,
        d_rowptr, d_colidx, infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
        d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    int numerical_zero;
    err = cusparseXcsrilu02_zeroPivot(
        cusparse, infoM, &numerical_zero);
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     return ACG_ERR_CUSPARSE;
    // }
    if (numerical_zero >= 0)
    {
      fprintf(stderr, "ACG: numerical zero at index %d\n", numerical_zero);
      // return ACG_ERR_CUSPARSE;
    }

    err = cusparseDestroyCsrilu02Info(infoM);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseDestroyMatDescr(matLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaFree(d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
  }

  err = cublasSetStream(cublas, stream);
  if (err)
  {
    if (errcode)
      *errcode = err;
  }

  err = cusparseSetStream(cusparse, stream);
  if (err)
  {
    if (errcode)
      *errcode = err;
  }

  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;

  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  cudaStreamSynchronize(commstream);
  cudaStreamSynchronize(collective_stream);
  gettime(&t0);

  /* compute right-hand side norm */
  double bnrm2sqr;
  acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
#if defined(ACG_USE_CUBLAS)
  err = cublasDdot(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
      d_bnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
#else
  err = acgsolvercuda_ddot(
      (b->num_nonzeros - b->num_ghost_nonzeros), d_b, d_b, d_bnrm2sqr, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return err;
  }
#endif
  acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
  nnrm2++;
  cg->nnrm2++;
  cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
  cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);

  if (commsize > 1 && !nocomm_allreduce)
  {
    acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    if (comm->type == acgcomm_nccl_split)
    {
      cudaEventRecord(dotEvent, stream);
      cudaStreamWaitEvent(collective_stream, dotEvent, 0);
      err = acgcomm_allreduce(
          ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, collective_stream, comm,
          errcode);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }
    else if (comm->type == acgcomm_nccl)
    {
      err = acgcomm_allreduce(
          ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          errcode);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }
    acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    nallreduce++;
    cg->nallreduce++;
    cg->Ballreduce += sizeof(bnrm2sqr);
  }
  if (commsize > 1 && !nocomm_allreduce)
    cudaStreamSynchronize(collective_stream);
  err = cudaMemcpy(
      &bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* compute initial residual, r₀ = b-A*x₀ */
  acgEventRecord(tcopy[2 * ncopy + 0], 0);

#if defined(ACG_USE_CUBLAS)
  err = cublasDcopy(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
#else
  err = acgsolvercuda_dcopy(
      (b->num_nonzeros - b->num_ghost_nonzeros), d_r, d_b, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return err;
  }
#endif
  acgEventRecord(tcopy[2 * ncopy + 1], 0);
  ncopy++;
  cg->ncopy++;
  cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

  if (commsize > 1 && !nocomm_p2p)
  {
    acgEventRecord(thalo[2 * nhalo + 0], commstream);
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 1], commstream);
    nhalo++;
    cg->nhalo++;
    cg->Bhalo += cg->halo->sendsize * sizeof(*x->x);
    cg->nhalomsgs += cg->halo->nrecipients;
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
#if defined(ACG_USE_CUSPARSE)
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
#else
  err = acgsolvercuda_csrgemv_merge(
      (A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1,
      nstartrows, d_startrows, numSMs, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return err;
  }
#endif
  acgEventRecord(tgemv[2 * ngemv + 1], 0);
  ngemv++;
  cg->ngemv++;
  cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
  cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      acgEventRecord(thalo[2 * nhalo + 0], commstream);
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(thalo[2 * nhalo + 1], commstream);
      nhalo++;
      cg->nhalo++;
      err = cudaEventRecord(xreceived, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(stream, xreceived, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
    }
    acgEventRecord(tgemv_o[2 * ngemv_o + 0], stream);
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = csrgemv_host(
        A->nborderrows, d_r + A->borderrowoffset, d_x + A->borderrowoffset,
        d_orowptr, d_ocolidx, d_oa, -1.0, numSMs, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
#endif
  }
  acgEventRecord(tgemv_o[2 * ngemv_o + 1], stream);
  ngemv_o++;
  cg->ngemv_o++;

  if (preconditioner == 0)
  {
    err = cudaMemcpyAsync(d_u, d_r, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r), cudaMemcpyDeviceToDevice, stream);

    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
  }
  else if (preconditioner == 1)
  {
    acgEventRecord(tprecond[2 * nprecond + 0], 0);
    acgsolvercuda_apply_jacobi_preconditioner((A->nprows - A->nghostrows), d_M_inv, d_r, d_u, numSMs, stream);
    acgEventRecord(tprecond[2 * nprecond + 1], 0);
    nprecond++;
    cg->nprecond++;
    // cg->Bprecon += (A->nprows - A->nghostrows) * (sizeof(*d_M_inv) + 2 * sizeof(*d_r));
  }
  else if (preconditioner == 2)
  {
#if defined(ACG_USE_CUSPARSE)
    acgEventRecord(tprecond[2 * nprecond + 0], 0);
    err = cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spSVDescrL, &bufferSizeL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_bufferL, bufferSizeL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL,
        d_bufferL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMemsetAsync(d_y, 0x0, A->nownedrows * sizeof(*d_y), stream);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cusparseSpSV_solve(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }

    // upper

    err = cusparseSpSV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
        spSVDescrU, &bufferSizeU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_bufferU, bufferSizeU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cusparseSpSV_analysis(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrU,
        d_bufferU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMemsetAsync(d_u, 0x0, A->nownedrows * sizeof(*d_u), stream);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUDA;
    }
    err = cusparseSpSV_solve(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    acgEventRecord(tprecond[2 * nprecond + 1], 0);
    nprecond++;
    cg->nprecond++;
    // cg->nprecon++;
#endif
  }

  if (commsize > 1 && !nocomm_p2p)
    err = cudaEventRecord(ureadytosend, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUDA;
  }

  // w = Au
  if (commsize > 1 && !nocomm_p2p)
  {
    err = cudaStreamWaitEvent(commstream, ureadytosend, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 0], commstream);
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_u, ACG_DOUBLE,
        x->num_nonzeros, d_u, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 1], commstream);
    nhalo++;
    cg->nhalo++;
    cg->Bhalo += cg->halo->sendsize * sizeof(*x->x);
    cg->nhalomsgs += cg->halo->nrecipients;
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
#if defined(ACG_USE_CUSPARSE)
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecu, d_one,
      vecw, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
#else
  err = acgsolvercuda_csrgemv_merge(
      (A->nprows - A->nghostrows), d_w, d_u, d_rowptr, d_colidx, d_a, 1.0, 1.0,
      nstartrows, d_startrows, numSMs, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return err;
  }
#endif
  acgEventRecord(tgemv[2 * ngemv + 1], 0);
  ngemv++;
  cg->ngemv++;
  cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
  cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      acgEventRecord(thalo[2 * nhalo + 0], commstream);
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_u, ACG_DOUBLE,
          x->num_nonzeros, d_u, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(thalo[2 * nhalo + 1], commstream);
      nhalo++;
      cg->nhalo++;
      err = cudaEventRecord(ureceived, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(stream, ureceived, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
    }
    acgEventRecord(tgemv_o[2 * ngemv_o + 0], stream);
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecuo,
        d_one, vecwo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = csrgemv_host(
        A->nborderrows, d_w + A->borderrowoffset, d_u + A->borderrowoffset,
        d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
#endif
    acgEventRecord(tgemv_o[2 * ngemv_o + 1], stream);
    ngemv_o++;
    cg->ngemv_o++;
  }

  /* iterative solver loop */
  for (int k = 0; k < maxits; k++)
  {

    /* compute (r, u) */
    acgEventRecord(tdot[2 * ndot + 0], stream);
#if defined(ACG_USE_CUBLAS)
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_u,
        1, d_rnrm2sqr);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
#else
    err = acgsolvercuda_ddot(
        (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros), d_r, d_u,
        d_rnrm2sqr, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
#endif
    acgEventRecord(tdot[2 * ndot + 1], stream);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

    // dot (w, u)
    acgEventRecord(tdot[2 * ndot + 0], stream);
#if defined(ACG_USE_CUBLAS)
    err = cublasDdot(
        cublas, cg->w->num_nonzeros - cg->w->num_ghost_nonzeros, d_w, 1, d_u,
        1, d_delta);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
#else
    err = acgsolvercuda_ddot(
        (cg->w->num_nonzeros - cg->w->num_ghost_nonzeros), d_w, d_u,
        d_delta, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
#endif
    acgEventRecord(tdot[2 * ndot + 1], stream);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

    if (commsize > 1 && !nocomm_allreduce)
    {
      err = cudaEventRecord(dotEvent, stream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }

    // preconditioning
    if (preconditioner == 0) // fake preconditioner
    {
      err = cudaMemcpyAsync(d_m, d_w,
                       (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t),
                       cudaMemcpyDeviceToDevice, stream);

      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUDA;
      }
    }
    else if (preconditioner == 1)
    {
      acgEventRecord(tprecond[2 * nprecond + 0], stream);
      acgsolvercuda_apply_jacobi_preconditioner((A->nprows - A->nghostrows),
                                                d_M_inv, d_w, d_m,
                                                numSMs, stream);
      acgEventRecord(tprecond[2 * nprecond + 1], stream);
      nprecond++;
      cg->nprecond++;
      // cg->Bprecon += (A->nprows - A->nghostrows) * (sizeof(*d_M_inv) + 2 * sizeof(*d_w));
    }
    else if (preconditioner == 2)
    {
#if defined(ACG_USE_CUSPARSE)
      acgEventRecord(tprecond[2 * nprecond + 0], 0);
      if (k == 0)
      {
        // lower
        err = cusparseSpSV_bufferSize(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecw, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrL, &bufferSizeL);
        if (err)
        {
          if (errcode)
            *errcode = err;
          return ACG_ERR_CUSPARSE;
        }
        err = cudaMalloc(&d_bufferL, bufferSizeL);
        if (err)
        {
          if (errcode)
            *errcode = err;
          return ACG_ERR_CUDA;
        }
        err = cusparseSpSV_analysis(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecw, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL,
            d_bufferL);
        if (err)
        {
          if (errcode)
            *errcode = err;
          return ACG_ERR_CUSPARSE;
        }

        // upper
        err = cusparseSpSV_bufferSize(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecy, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrL, &bufferSizeL);
        if (err)
        {
          if (errcode)
            *errcode = err;
          return ACG_ERR_CUSPARSE;
        }
        err = cudaMalloc(&d_bufferL, bufferSizeL);
        if (err)
        {
          if (errcode)
            *errcode = err;
          return ACG_ERR_CUDA;
        }
        err = cusparseSpSV_analysis(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecy, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL,
            d_bufferL);
        if (err)
        {
          if (errcode)
            *errcode = err;
          return ACG_ERR_CUSPARSE;
        }
      }

      err = cudaMemsetAsync(d_y, 0x0, A->nownedrows * sizeof(*d_y), stream);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUDA;
      }
      err = cudaMemsetAsync(d_m, 0x0, A->nownedrows * sizeof(*d_m), stream);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUDA;
      }
      err = cusparseSpSV_solve(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
          vecw, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cusparseSpSV_solve(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
          vecy, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      acgEventRecord(tprecond[2 * nprecond + 1], 0);
      nprecond++;
      cg->nprecond++;
#endif
    }

    if (commsize > 1 && !nocomm_allreduce)
    {
      if (comm->type == acgcomm_nccl_split)
      {
        err = cudaStreamWaitEvent(collective_stream, dotEvent, 0);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return err;
        }

        acgEventRecord(tallreduce[2 * nallreduce + 0], collective_stream);
        err = acgcomm_allreduce(
            ACG_IN_PLACE, d_rnrm2sqr, ACG_PIPELINED_ALLREDUCE_SIZE, ACG_DOUBLE, ACG_SUM,
            collective_stream, comm, errcode);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return err;
        }
        acgEventRecord(tallreduce[2 * nallreduce + 1], collective_stream);
      }
      else if (comm->type == acgcomm_nccl)
      {
        acgEventRecord(tallreduce[2 * nallreduce + 0], stream);
        err = acgcomm_allreduce(
            ACG_IN_PLACE, d_rnrm2sqr, ACG_PIPELINED_ALLREDUCE_SIZE, ACG_DOUBLE, ACG_SUM,
            stream, comm, errcode);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return err;
        }
        acgEventRecord(tallreduce[2 * nallreduce + 1], stream);
      }
      else if (comm->type == acgcomm_mpi)
      {
        cudaStreamSynchronize(stream);
        /**
         * TODO: will change the following part. That will go to the
         * acgcomm_allreduce() method once I find a clear way of doing it.
         **/
        err = MPI_Iallreduce(MPI_IN_PLACE, d_rnrm2sqr, 2, MPI_DOUBLE, MPI_SUM, comm->mpicomm, &request);
        if (err)
        {
          if (errcode)
            *errcode = err;
          ACG_ERR_MPI;
        }
      }

      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += sizeof(*d_pdott);
    }

    if (comm->type == acgcomm_nccl)
    {
      /* start copying residual norm from device to host,
       * overlapping it with the matrix-vector product */
      err = cudaEventRecord(rnrm2sqrready, stream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaMemcpyAsync(
          rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost,
          copystream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
    }

    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaEventRecord(mreadytosend, stream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
    }
    // SpMV
    /* compute n = Am */
    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaStreamWaitEvent(commstream, mreadytosend, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      acgEventRecord(thalo[2 * nhalo + 0], commstream);
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_m, ACG_DOUBLE,
          cg->p.num_nonzeros, d_m, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      acgEventRecord(thalo[2 * nhalo + 1], commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      nhalo++;
      cg->nhalo++;
      cg->Bhalo += cg->halo->sendsize * sizeof(*cg->p.x);
      cg->nhalomsgs += cg->halo->nrecipients;
    }
    acgEventRecord(tgemv[2 * ngemv + 0], stream);
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecm,
        d_zero, vecn, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_n, d_m, d_rowptr, d_colidx, d_a, 1.0, 0.0,
        nstartrows, d_startrows, numSMs, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
#endif
    acgEventRecord(tgemv[2 * ngemv + 1], stream);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->t.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->p.num_nonzeros * sizeof(*cg->p.x);

    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        acgEventRecord(thalo[2 * nhalo + 0], commstream);
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_m, ACG_DOUBLE,
            cg->p.num_nonzeros, d_m, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
            commstream);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return err;
        }
        acgEventRecord(thalo[2 * nhalo + 0], commstream);
        nhalo++;
        cg->nhalo++;
        err = cudaEventRecord(mreceived, commstream);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return ACG_ERR_CUDA;
        }
        err = cudaStreamWaitEvent(stream, mreceived, 0);
        if (err)
        {
          gettime(&t1);
          cg->tsolve += elapsed(t0, t1);
          return ACG_ERR_CUDA;
        }
      }
      acgEventRecord(tgemv_o[2 * ngemv_o + 0], stream);
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecmo,
          d_one, vecno, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(
          A->nborderrows, d_n + A->borderrowoffset, d_m + A->borderrowoffset,
          d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
#endif
    }
    acgEventRecord(tgemv_o[2 * ngemv_o + 1], stream);
    ngemv_o++;
    cg->ngemv_o++;

    if (commsize > 1 && !nocomm_allreduce && comm->type == acgcomm_nccl_split)
      cudaStreamSynchronize(collective_stream);
    else
      cudaStreamSynchronize(copystream);

    if (comm->type == acgcomm_nccl_split)
    {
      err = cudaMemcpy(
          rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
    }

    /* wait for host to receive updated residual norm */
    // cudaStreamSynchronize(stream);
    cg->rnrm2 = sqrt(*rnrm2sqr);
    if (k == 0)
    {
      cg->r0nrm2 = cg->rnrm2;
      residualrtol *= cg->r0nrm2;
    }

    // /* convergence tests */
    // if ((diffatol > 0 && cg->dxnrm2 < diffatol) || (diffrtol > 0 && cg->dxnrm2 < diffrtol) || (residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    // {
    //     cudaStreamSynchronize(stream);
    //     converged = true;
    //     break;
    // }
    if ((residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      cudaStreamSynchronize(stream);
      converged = true;
      break;
    }

    /* update vectors */
    acgEventRecord(taxpy[2 * naxpy + 0], stream);
    err = acgsolvercuda_preconditioned_pipelined_daxpy_fused(
        cg->t.num_nonzeros - cg->t.num_ghost_nonzeros, k,
        d_rnrm2sqr, d_rnrm2sqr_prev, d_delta,
        d_p, d_r, d_t, d_x, d_z, d_w, d_q, d_n, d_m, d_u,
        d_alpha, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(taxpy[2 * naxpy + 1], stream);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 16 * (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros);
    cg->Baxpy += 8 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*cg->p.x);

    cg->ntotaliterations++;
    cg->niterations++;
  }
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

#if defined(ACG_ENABLE_PROFILING)
  /* record profiling information */
  float t;
  for (acgidx_t i = 0; i < ngemv; i++)
  {
    cudaEventSynchronize(tgemv[2 * i + 1]);
    cudaEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
    cg->tgemv += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ngemv_o; i++)
  {
    cudaEventSynchronize(tgemv_o[2 * i + 1]);
    cudaEventElapsedTime(&t, tgemv_o[2 * i + 0], tgemv_o[2 * i + 1]);
    cg->tgemv_o += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ndot; i++)
  {
    cudaEventSynchronize(tdot[2 * i + 1]);
    cudaEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
    cg->tdot += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nnrm2; i++)
  {
    cudaEventSynchronize(tnrm2[2 * i + 1]);
    cudaEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
    cg->tnrm2 += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < naxpy; i++)
  {
    cudaEventSynchronize(taxpy[2 * i + 1]);
    cudaEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
    cg->taxpy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ncopy; i++)
  {
    cudaEventSynchronize(tcopy[2 * i + 1]);
    cudaEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
    cg->tcopy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nallreduce; i++)
  {
    cudaEventSynchronize(tallreduce[2 * i + 1]);
    cudaEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
    cg->tallreduce += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nhalo; i++)
  {
    cudaEventSynchronize(thalo[2 * i + 1]);
    cudaEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
    cg->thalo += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nprecond; i++)
  {
    cudaEventSynchronize(tprecond[2 * i + 1]);
    cudaEventElapsedTime(&t, tprecond[2 * i + 0], tprecond[2 * i + 1]);
    cg->tprecond += 1.0e-3 * t;
  }
#endif

  /* copy solution back to host. The streams are non-blocking, so the
   * default-stream copy is not ordered against the solver loop; drain it
   * first. */
  cudaStreamSynchronize(stream);
  err = cudaMemcpy(
      x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  /* free cusparse matrix and vectors */
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecp);
  cusparseDestroyDnVec(vect);
  cusparseDestroyDnVec(vecw);
  cusparseDestroyDnVec(vecz);
  cusparseDestroyDnVec(vecn);
  cusparseDestroyDnVec(vecm);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecpo);
    cusparseDestroyDnVec(vecto);
    cusparseDestroyDnVec(vecno);
    cusparseDestroyDnVec(vecmo);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
#endif

#if !defined(ACG_USE_CUSPARSE)
  cudaFree(d_startrows);
#endif

  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_z);
  cudaFree(d_w);
  cudaFree(d_n);
  cudaFree(d_m);
  cudaFree(d_q);
  cudaFree(d_u);
  cudaFreeHost(rnrm2sqr);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(copystream);
  cudaStreamDestroy(collective_stream);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  /* check for CUDA errors */
  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  /* if the solver converged or the only stopping criteria is a
   * maximum number of iterations, then the solver succeeded */
  if (converged)
    return ACG_SUCCESS;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;

  /* otherwise, the solver failed to converge with the given number
   * of maximum iterations */
  return ACG_ERR_NOT_CONVERGED;
}

/*
 * Select how the BiCGStab stabilisation scalar ω is computed:
 *   0 -> plain,          ω = (t,s)/(t,t)
 *   1 -> preconditioned, ω = (M⁻¹t,M⁻¹s)/(M⁻¹t,M⁻¹t)
 * Both variants are implemented below; flip this define to switch.
 */
#ifndef ACG_BICGSTAB_OMEGA_PRECONDITIONED
#define ACG_BICGSTAB_OMEGA_PRECONDITIONED 0
#endif

/*
 * ‘bicgstab_allreduce()’ performs an in-place sum-allreduce of a small
 * device-side buffer, handling the supported communicator types. It is a
 * no-op for a single process or when collective communication is disabled.
 */
static int bicgstab_allreduce(
    double *d_buf,
    int count,
    struct acgcomm *comm,
    int commsize,
    int nocomm_allreduce,
    cudaStream_t stream,
    cudaStream_t collective_stream,
    cudaEvent_t dotEvent,
    int *errcode)
{
  int err;
  if (!(commsize > 1 && !nocomm_allreduce))
    return ACG_SUCCESS;
  if (comm->type == acgcomm_nccl_split)
  {
    cudaEventRecord(dotEvent, stream);
    cudaStreamWaitEvent(collective_stream, dotEvent, 0);
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, collective_stream,
        comm, errcode);
    if (err)
      return err;
    cudaStreamSynchronize(collective_stream);
  }
  else if (comm->type == acgcomm_nccl)
  {
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
    if (err)
      return err;
  }
  else if (comm->type == acgcomm_mpi)
  {
    cudaStreamSynchronize(stream);
    err = MPI_Allreduce(
        MPI_IN_PLACE, d_buf, count, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_MPI;
    }
  }
  return ACG_SUCCESS;
}

/*
 * ‘bicgstab_spmv()’ computes d_out = A·d_in for the distributed matrix A,
 * overlapping the halo exchange of d_in with the local (owned-row)
 * matrix-vector product, then accumulating the off-diagonal contribution
 * on the border rows. Mirrors the SpMV path used by the pipelined CG
 * solvers (both cuSPARSE and merge-based backends).
 */
static int bicgstab_spmv(
    const struct acgsymcsrmatrix *A,
    struct acgcomm *comm,
    int commsize,
    int nocomm_p2p,
    struct acghalo *halo,
    struct acghaloexchange *haloexchange,
    int tag,
    int *errcode,
    int numSMs,
    cudaStream_t stream,
    cudaStream_t commstream,
    int nnz_full,
    double *d_in,
    double *d_out,
    cudaEvent_t inreadytosend,
    cudaEvent_t inreceived,
    double *d_one,
    double *d_zero,
    cusparseHandle_t cusparse,
    cusparseSpMatDescr_t matA,
    cusparseSpMatDescr_t matO,
    cusparseDnVecDescr_t vecin,
    cusparseDnVecDescr_t vecout,
    cusparseDnVecDescr_t vecino,
    cusparseDnVecDescr_t vecouto,
    void *d_buffer,
    void *d_obuffer,
    acgidx_t *d_rowptr,
    acgidx_t *d_colidx,
    double *d_a,
    acgidx_t *d_orowptr,
    acgidx_t *d_ocolidx,
    double *d_oa,
    acgidx_t nstartrows,
    acgidx_t *d_startrows)
{
  int err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = cudaEventRecord(inreadytosend, stream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaStreamWaitEvent(commstream, inreadytosend, 0);
    if (err)
      return ACG_ERR_CUDA;
    err = acghalo_exchange_cuda_begin(
        halo, haloexchange, nnz_full, d_in, ACG_DOUBLE, nnz_full, d_in,
        ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_CUSPARSE)
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecin, d_zero,
      vecout, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
#else
  err = acgsolvercuda_csrgemv_merge(
      (A->nprows - A->nghostrows), d_out, d_in, d_rowptr, d_colidx, d_a, 1.0,
      0.0, nstartrows, d_startrows, numSMs, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_cuda_end(
          halo, haloexchange, nnz_full, d_in, ACG_DOUBLE, nnz_full, d_in,
          ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
      err = cudaEventRecord(inreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, inreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
    }
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecino, d_one,
        vecouto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = csrgemv_host(
        A->nborderrows, d_out + A->borderrowoffset, d_in + A->borderrowoffset,
        d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
    if (err)
      return err;
#endif
  }
  return ACG_SUCCESS;
}

/**
 * ‘acgsolvercuda_solve_preconditioned_bicgstab()’ solves the given linear
 * system, Ax=b, using a preconditioned stabilised bi-conjugate gradient
 * (BiCGStab) method. The linear system may be distributed across multiple
 * processes and communication is handled using MPI. The preconditioner is
 * selected by ‘preconditioner’: 0 -> none, 1 -> Jacobi, 2 -> ILU(0).
 *
 * The solver infrastructure (CUDA streams, halo exchange, SpMV and the
 * cuSPARSE/cuBLAS handles) mirrors the pipelined preconditioned CG solver.
 */
int acgsolvercuda_solve_preconditioned_bicgstab(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    int preconditioner,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

#if defined(ACG_NOCOMM)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE) && defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE)
  int nocomm_allreduce = 1, nocomm_p2p = 0;
#elif defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 0, nocomm_p2p = 1;
#else
  int nocomm_allreduce = 0, nocomm_p2p = 0;
#endif

  /* allocate the extra work vectors (reuses the pipelined-CG storage) */
  if (!cg->w)
  {
    cg->w = malloc(sizeof(*cg->w));
    if (!cg->w)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->w, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->q)
  {
    cg->q = malloc(sizeof(*cg->q));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->q, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->z)
  {
    cg->z = malloc(sizeof(*cg->z));
    if (!cg->z)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->z, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->m)
  {
    cg->m = malloc(sizeof(*cg->m));
    if (!cg->m)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->m, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_m, cg->m->num_nonzeros * sizeof(*cg->d_m));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->n)
  {
    cg->n = malloc(sizeof(*cg->n));
    if (!cg->n)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->n, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_n, cg->n->num_nonzeros * sizeof(*cg->d_n));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->u)
  {
    cg->u = malloc(sizeof(*cg->u));
    if (!cg->u)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->u, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_u, cg->u->num_nonzeros * sizeof(*cg->d_u));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->y)
  {
    cg->y = malloc(sizeof(*cg->y));
    if (!cg->y)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->y, x);
    if (err)
      return err;
    err = cudaMalloc((void **)&cg->d_y, cg->y->num_nonzeros * sizeof(*cg->d_y));
    if (err)
      return ACG_ERR_CUDA;
  }

  /*
   * Vector role mapping for BiCGStab:
   *   d_r  = r (residual)      d_w = r̂ (shadow residual, constant)
   *   d_p  = p                 d_q = v = A·ŷ
   *   d_n  = s                 d_t = t = A·ẑ
   *   d_y  = ŷ = M⁻¹p          d_z = ẑ = M⁻¹s
   *   d_u  = M⁻¹t (ω precond)  d_m = triangular-solve intermediate (ILU)
   */
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_r = cg->d_r;
  double *d_w = cg->d_w; /* r̂ */
  double *d_p = cg->d_p;
  double *d_q = cg->d_q; /* v */
  double *d_n = cg->d_n; /* s */
  double *d_t = cg->d_t; /* t */
  double *d_y = cg->d_y; /* ŷ */
  double *d_z = cg->d_z; /* ẑ */
  double *d_u = cg->d_u; /* M⁻¹t */
  double *d_m = cg->d_m; /* intermediate */
  double *d_alpha = cg->d_alpha;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;
  int nnz_full = x->num_nonzeros;
  int n_owned = cg->r.num_nonzeros - cg->r.num_ghost_nonzeros;

  /* BiCGStab device-side scalars */
  double *d_rho, *d_rho_prev, *d_omega, *d_rhat_v, *d_nrm2sqr, *d_dots;
  err = cudaMalloc((void **)&d_rho, sizeof(*d_rho));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_rho_prev, sizeof(*d_rho_prev));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_omega, sizeof(*d_omega));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_rhat_v, sizeof(*d_rhat_v));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_nrm2sqr, sizeof(*d_nrm2sqr));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_dots, 2 * sizeof(*d_dots));
  if (err)
    return ACG_ERR_CUDA;
  double *d_dot_num = &d_dots[0]; /* (t,s) or (M⁻¹t,M⁻¹s) */
  double *d_dot_den = &d_dots[1]; /* (t,t) or (M⁻¹t,M⁻¹t) */

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  int leastPriority, greatestPriority;
  err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_CUDA;

  cudaStream_t stream;
  err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t collective_stream;
  err = cudaStreamCreateWithPriority(&collective_stream, cudaStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t commstream;
  err = cudaStreamCreateWithPriority(&commstream, cudaStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_CUDA;
  acgSetStreamName(commstream, "P2P");
  acgSetStreamName(collective_stream, "Allreduce");
  acgSetStreamName(stream, "Compute");

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  cudaEvent_t dotEvent;
  cudaEventCreateWithFlags(&dotEvent, cudaEventDisableTiming);
  cudaEvent_t yreadytosend, yreceived, zreadytosend, zreceived, xreadytosend, xreceived;
  cudaEventCreateWithFlags(&yreadytosend, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&yreceived, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&zreadytosend, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&zreceived, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming);

  /* copy right-hand side and initial guess to device */
  double *d_b, *d_x;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  /* create cusparse dense vectors and matrices */
  cusparseDnVecDescr_t vecx, vecr, vecp, vecn, vecq, vect, vecy, vecz, vecm, vecu;
  cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  cusparseCreateDnVec(&vecp, A->nownedrows, d_p, CUDA_R_64F);
  cusparseCreateDnVec(&vecn, A->nownedrows, d_n, CUDA_R_64F);
  cusparseCreateDnVec(&vecq, A->nownedrows, d_q, CUDA_R_64F);
  cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
  cusparseCreateDnVec(&vecy, A->nownedrows, d_y, CUDA_R_64F);
  cusparseCreateDnVec(&vecz, A->nownedrows, d_z, CUDA_R_64F);
  cusparseCreateDnVec(&vecm, A->nownedrows, d_m, CUDA_R_64F);
  cusparseCreateDnVec(&vecu, A->nownedrows, d_u, CUDA_R_64F);
  cusparseDnVecDescr_t vecxo, vecro, vecyo, vecqo, veczo, vecto;
  if (commsize > 1)
  {
    int no = A->nborderrows + A->nghostrows;
    cusparseCreateDnVec(&vecxo, no, d_x + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecro, no, d_r + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecyo, no, d_y + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecqo, no, d_q + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&veczo, no, d_z + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecto, no, d_t + A->borderrowoffset, CUDA_R_64F);
  }
  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;
  cusparseSpMatDescr_t matO = NULL;
  void *d_obuffer = NULL;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows,
        A->onpnzs, d_orowptr, d_ocolidx, d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo,
        d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
  }
#else
  /* dummy descriptors for the merge-based backend */
  cusparseSpMatDescr_t matA = NULL, matO = NULL;
  cusparseDnVecDescr_t vecx = NULL, vecr = NULL, vecp = NULL, vecn = NULL,
                       vecq = NULL, vect = NULL, vecy = NULL, vecz = NULL,
                       vecm = NULL, vecu = NULL;
  cusparseDnVecDescr_t vecxo = NULL, vecro = NULL, vecyo = NULL, vecqo = NULL,
                       veczo = NULL, vecto = NULL;
  void *d_buffer = NULL, *d_obuffer = NULL;

  /* setup for merge-based SpMV */
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows_v = ntasks / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = cudaMalloc((void **)&d_startrows, nstartrows_v * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_CUDA;
  err = acgsolvercuda_csrgemv_merge_init(
      (A->nprows - A->nghostrows), d_rowptr, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
  err = cudaStreamSynchronize(stream);
  if (err)
    return ACG_ERR_CUDA;
#endif
#if defined(ACG_USE_CUSPARSE)
  acgidx_t nstartrows_v = 0;
  acgidx_t *d_startrows = NULL;
#endif

  /*
   * Preconditioner setup (mirrors the pipelined preconditioned solver).
   */
  double *d_M_inv = NULL;
  cusparseSpMatDescr_t matM_lower = NULL, matM_upper = NULL;
  double *d_M_values = NULL;
  cusparseSpSVDescr_t spL_p = NULL, spU_y = NULL, spL_s = NULL, spU_z = NULL,
                      spL_t = NULL, spU_u = NULL;
  void *d_bufL_p = NULL, *d_bufU_y = NULL, *d_bufL_s = NULL, *d_bufU_z = NULL,
       *d_bufL_t = NULL, *d_bufU_u = NULL;

  if (preconditioner == 1)
  {
    err = cudaMalloc(&d_M_inv, (A->nprows) * sizeof(double));
    if (err)
      return ACG_ERR_CUDA;
    double *h_M_inv = (double *)malloc((A->nprows) * sizeof(double));
    if (!h_M_inv)
      return ACG_ERR_ERRNO;
    for (int i = 0; i < (A->nprows); i++)
    {
      double diag = 0.0;
      for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; j++)
        if (A->colidx[j] == i + A->rowidxbase)
          diag += A->a[j];
      h_M_inv[i] = diag;
    }
    err = cudaMemcpyAsync(d_M_inv, h_M_inv, (A->nprows) * sizeof(double), cudaMemcpyHostToDevice, stream);
    if (!err)
      err = cudaStreamSynchronize(stream); /* ‘h_M_inv’ is freed below */
    free(h_M_inv);
    if (err)
      return ACG_ERR_CUDA;
  }
#if defined(ACG_USE_CUSPARSE)
  else if (preconditioner == 2)
  {
    cusparseMatDescr_t matLU;
    cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
    cusparseFillMode_t fill_upper = CUSPARSE_FILL_MODE_UPPER;
    cusparseDiagType_t diag_unit = CUSPARSE_DIAG_TYPE_UNIT;
    cusparseDiagType_t diag_nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;

    err = cudaMalloc(&d_M_values, A->fnpnzs * sizeof(*d_M_values));
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMemcpyAsync(d_M_values, d_a, A->fnpnzs * sizeof(*d_M_values), cudaMemcpyDeviceToDevice, stream);
    if (err)
      return ACG_ERR_CUDA;

    err = cusparseCreateCsr(
        &matM_lower, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr,
        d_colidx, d_M_values, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
      return ACG_ERR_CUSPARSE;
    cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
    cusparseSpMatSetAttribute(matM_lower, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));

    err = cusparseCreateCsr(
        &matM_upper, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr,
        d_colidx, d_M_values, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
      return ACG_ERR_CUSPARSE;
    cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper));
    cusparseSpMatSetAttribute(matM_upper, CUSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit, sizeof(diag_nonunit));

    /* incomplete-LU factorisation, in place on d_M_values */
    csrilu02Info_t infoM = NULL;
    int bufferSizeLU = 0;
    void *d_bufferLU;
    cusparseCreateMatDescr(&matLU);
    cusparseSetMatType(matLU, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matLU, CUSPARSE_INDEX_BASE_ZERO);
    cusparseCreateCsrilu02Info(&infoM);
    err = cusparseDcsrilu02_bufferSize(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, &bufferSizeLU);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cudaMalloc(&d_bufferLU, bufferSizeLU);
    if (err)
      return ACG_ERR_CUDA;
    err = cusparseDcsrilu02_analysis(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
      return ACG_ERR_CUSPARSE;
    int structural_zero;
    if (cusparseXcsrilu02_zeroPivot(cusparse, infoM, &structural_zero) == CUSPARSE_STATUS_ZERO_PIVOT)
      fprintf(stderr, "ACG: structural zero at index %d\n", structural_zero);
    err = cusparseDcsrilu02(
        cusparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
      return ACG_ERR_CUSPARSE;
    int numerical_zero;
    if (cusparseXcsrilu02_zeroPivot(cusparse, infoM, &numerical_zero) == CUSPARSE_STATUS_ZERO_PIVOT)
      fprintf(stderr, "ACG: numerical zero at index %d\n", numerical_zero);
    cusparseDestroyCsrilu02Info(infoM);
    cusparseDestroyMatDescr(matLU);
    cudaFree(d_bufferLU);

    /*
     * Create and analyse a separate triangular-solve descriptor for each
     * (matrix, input, output) site used in the iteration:
     *   ŷ = M⁻¹p : (L) p->m, (U) m->y
     *   ẑ = M⁻¹s : (L) s->m, (U) m->z
     *   M⁻¹t     : (L) t->m, (U) m->u   (only for the preconditioned ω)
     */
    size_t bsz;
    cusparseSpSV_createDescr(&spL_p);
    cusparseSpSV_createDescr(&spU_y);
    cusparseSpSV_createDescr(&spL_s);
    cusparseSpSV_createDescr(&spU_z);
    err = cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecp, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_p, &bsz);
    if (err)
      return ACG_ERR_CUSPARSE;
    cudaMalloc(&d_bufL_p, bsz);
    err = cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecp, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_p, d_bufL_p);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_y, &bsz);
    if (err)
      return ACG_ERR_CUSPARSE;
    cudaMalloc(&d_bufU_y, bsz);
    err = cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_y, d_bufU_y);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecn, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_s, &bsz);
    if (err)
      return ACG_ERR_CUSPARSE;
    cudaMalloc(&d_bufL_s, bsz);
    err = cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecn, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_s, d_bufL_s);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecz, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_z, &bsz);
    if (err)
      return ACG_ERR_CUSPARSE;
    cudaMalloc(&d_bufU_z, bsz);
    err = cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecz, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_z, d_bufU_z);
    if (err)
      return ACG_ERR_CUSPARSE;
#if ACG_BICGSTAB_OMEGA_PRECONDITIONED
    cusparseSpSV_createDescr(&spL_t);
    cusparseSpSV_createDescr(&spU_u);
    err = cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vect, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_t, &bsz);
    if (err)
      return ACG_ERR_CUSPARSE;
    cudaMalloc(&d_bufL_t, bsz);
    err = cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vect, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_t, d_bufL_t);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_u, &bsz);
    if (err)
      return ACG_ERR_CUSPARSE;
    cudaMalloc(&d_bufU_u, bsz);
    err = cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_u, d_bufU_u);
    if (err)
      return ACG_ERR_CUSPARSE;
#endif
  }
#endif

  err = cublasSetStream(cublas, stream);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cusparseSetStream(cusparse, stream);
  if (err)
    return ACG_ERR_CUSPARSE;

  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;

  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  cudaStreamSynchronize(commstream);
  cudaStreamSynchronize(collective_stream);
  gettime(&t0);

  /* ‖b‖₂ */
  err = cublasDdot(cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
  if (err)
    return ACG_ERR_CUBLAS;
  err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  double bnrm2sqr;
  err = cudaMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* r₀ = b − A·x₀ */
  err = acgsolvercuda_dcopy(b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
  if (err)
    return err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = cudaEventRecord(xreadytosend, stream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaStreamWaitEvent(commstream, xreadytosend, 0);
    if (err)
      return ACG_ERR_CUDA;
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x,
        ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_CUSPARSE)
  err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
    return ACG_ERR_CUSPARSE;
#else
  err = acgsolvercuda_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1, nstartrows_v, d_startrows, numSMs, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_cuda_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
      err = cudaEventRecord(xreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, xreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
    }
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
      return ACG_ERR_CUSPARSE;
#else
    err = csrgemv_host(A->nborderrows, d_r + A->borderrowoffset, d_x + A->borderrowoffset, d_orowptr, d_ocolidx, d_oa, -1.0, numSMs, stream);
    if (err)
      return err;
#endif
  }
  cg->ngemv++;

  /* r̂ = r₀ (shadow residual) */
  err = acgsolvercuda_dcopy(n_owned, d_w, d_r, stream);
  if (err)
    return err;

  /* p = 0, v = 0 */
  err = cudaMemsetAsync(d_p, 0, nnz_full * sizeof(*d_p), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_q, 0, nnz_full * sizeof(*d_q), stream);
  if (err)
    return ACG_ERR_CUDA;

  /* scalars: ρ_prev = ω = α = 1 */
  err = cudaMemcpyAsync(d_rho_prev, d_one, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(d_omega, d_one, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(d_alpha, d_one, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;

  /* ‖r₀‖₂ */
  err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, d_nrm2sqr);
  if (err)
    return ACG_ERR_CUBLAS;
  err = bicgstab_allreduce(d_nrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  double nrm2sqr;
  err = cudaMemcpy(&nrm2sqr, d_nrm2sqr, sizeof(nrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->r0nrm2 = cg->rnrm2 = sqrt(nrm2sqr);
  if (residualrtol > 0)
    residualrtol *= cg->r0nrm2;

  /* iterative solver loop */
  bool breakdown = false;
  /* fixed-iteration benchmark mode: no stopping criteria are set, so run
   * exactly ‘maxits’ iterations and disable the breakdown early-out */
  bool fixed_iterations =
      (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0);
  for (int k = 0; k < maxits; k++)
  {
    /* ρ = (r̂, r) */
    err = cublasDdot(cublas, n_owned, d_w, 1, d_r, 1, d_rho);
    if (err)
      return ACG_ERR_CUBLAS;
    err = bicgstab_allreduce(d_rho, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* p = r + β(p − ω·v),  β = (ρ/ρ_prev)(α/ω) */
    err = acgsolvercuda_bicgstab_p_update(n_owned, k, d_rho, d_rho_prev, d_alpha, d_omega, d_p, d_r, d_q, stream);
    if (err)
      return err;

    /* ŷ = M⁻¹p */
    if (preconditioner == 0)
    {
      err = cudaMemcpyAsync(d_y, d_p, n_owned * sizeof(*d_y), cudaMemcpyDeviceToDevice, stream);
      if (err)
        return ACG_ERR_CUDA;
    }
    else if (preconditioner == 1)
    {
      err = acgsolvercuda_apply_jacobi_preconditioner(n_owned, d_M_inv, d_p, d_y, numSMs, stream);
      if (err)
        return err;
    }
#if defined(ACG_USE_CUSPARSE)
    else if (preconditioner == 2)
    {
      cudaMemsetAsync(d_m, 0, A->nownedrows * sizeof(*d_m), stream);
      cudaMemsetAsync(d_y, 0, A->nownedrows * sizeof(*d_y), stream);
      err = cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecp, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_p);
      if (err)
        return ACG_ERR_CUSPARSE;
      err = cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecy, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_y);
      if (err)
        return ACG_ERR_CUSPARSE;
    }
#endif

    /* v = A·ŷ */
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        numSMs, stream, commstream, nnz_full, d_y, d_q, yreadytosend, yreceived,
        d_one, d_zero, cusparse, matA, matO, vecy, vecq, vecyo, vecqo, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
        nstartrows_v, d_startrows);
    if (err)
      return err;
    cg->ngemv++;

    /* (r̂, v) */
    err = cublasDdot(cublas, n_owned, d_w, 1, d_q, 1, d_rhat_v);
    if (err)
      return ACG_ERR_CUBLAS;
    err = bicgstab_allreduce(d_rhat_v, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* α = ρ/(r̂,v),  s = r − α·v */
    err = acgsolvercuda_bicgstab_s_update(n_owned, d_alpha, d_rho, d_rhat_v, d_n, d_r, d_q, stream);
    if (err)
      return err;

    /* ‖s‖₂ — half-step convergence test */
    err = cublasDdot(cublas, n_owned, d_n, 1, d_n, 1, d_nrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = bicgstab_allreduce(d_nrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;
    cudaStreamSynchronize(stream);
    err = cudaMemcpy(&nrm2sqr, d_nrm2sqr, sizeof(nrm2sqr), cudaMemcpyDeviceToHost);
    if (err)
      return ACG_ERR_CUDA;
    double snrm2 = sqrt(nrm2sqr);
    cg->ndot++;

    if ((residualatol > 0 && snrm2 < residualatol) || (residualrtol > 0 && snrm2 < residualrtol))
    {
      /* x = x + α·ŷ */
      err = acgsolvercuda_bicgstab_x_halfstep(n_owned, d_alpha, d_x, d_y, stream);
      if (err)
        return err;
      cudaStreamSynchronize(stream);
      cg->rnrm2 = snrm2;
      cg->ntotaliterations++;
      cg->niterations++;
      converged = true;
      break;
    }

    /* ẑ = M⁻¹s */
    if (preconditioner == 0)
    {
      err = cudaMemcpyAsync(d_z, d_n, n_owned * sizeof(*d_z), cudaMemcpyDeviceToDevice, stream);
      if (err)
        return ACG_ERR_CUDA;
    }
    else if (preconditioner == 1)
    {
      err = acgsolvercuda_apply_jacobi_preconditioner(n_owned, d_M_inv, d_n, d_z, numSMs, stream);
      if (err)
        return err;
    }
#if defined(ACG_USE_CUSPARSE)
    else if (preconditioner == 2)
    {
      cudaMemsetAsync(d_m, 0, A->nownedrows * sizeof(*d_m), stream);
      cudaMemsetAsync(d_z, 0, A->nownedrows * sizeof(*d_z), stream);
      err = cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecn, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_s);
      if (err)
        return ACG_ERR_CUSPARSE;
      err = cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecz, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_z);
      if (err)
        return ACG_ERR_CUSPARSE;
    }
#endif

    /* t = A·ẑ */
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        numSMs, stream, commstream, nnz_full, d_z, d_t, zreadytosend, zreceived,
        d_one, d_zero, cusparse, matA, matO, vecz, vect, veczo, vecto, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
        nstartrows_v, d_startrows);
    if (err)
      return err;
    cg->ngemv++;

    /* ω numerator/denominator */
#if ACG_BICGSTAB_OMEGA_PRECONDITIONED
    /* M⁻¹t into d_u, then ω = (M⁻¹t,M⁻¹s)/(M⁻¹t,M⁻¹t) with M⁻¹s = ẑ = d_z */
    if (preconditioner == 0)
    {
      err = cudaMemcpyAsync(d_u, d_t, n_owned * sizeof(*d_u), cudaMemcpyDeviceToDevice, stream);
      if (err)
        return ACG_ERR_CUDA;
    }
    else if (preconditioner == 1)
    {
      err = acgsolvercuda_apply_jacobi_preconditioner(n_owned, d_M_inv, d_t, d_u, numSMs, stream);
      if (err)
        return err;
    }
#if defined(ACG_USE_CUSPARSE)
    else if (preconditioner == 2)
    {
      cudaMemsetAsync(d_m, 0, A->nownedrows * sizeof(*d_m), stream);
      cudaMemsetAsync(d_u, 0, A->nownedrows * sizeof(*d_u), stream);
      err = cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vect, vecm, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spL_t);
      if (err)
        return ACG_ERR_CUSPARSE;
      err = cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecu, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spU_u);
      if (err)
        return ACG_ERR_CUSPARSE;
    }
#endif
    err = cublasDdot(cublas, n_owned, d_u, 1, d_z, 1, d_dot_num);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_u, 1, d_u, 1, d_dot_den);
    if (err)
      return ACG_ERR_CUBLAS;
#else
    /* plain ω = (t,s)/(t,t) */
    err = cublasDdot(cublas, n_owned, d_t, 1, d_n, 1, d_dot_num);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_t, 1, d_t, 1, d_dot_den);
    if (err)
      return ACG_ERR_CUBLAS;
#endif
    err = bicgstab_allreduce(d_dots, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* ω = num/den,  x += α·ŷ + ω·ẑ,  r = s − ω·t,  ρ_prev = ρ */
    err = acgsolvercuda_bicgstab_xr_update(n_owned, d_omega, d_rho_prev, d_dot_num, d_dot_den, d_rho, d_alpha, d_x, d_y, d_z, d_r, d_n, d_t, stream);
    if (err)
      return err;

    /* ‖r‖₂ */
    err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, d_nrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = bicgstab_allreduce(d_nrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;
    cudaStreamSynchronize(stream);
    err = cudaMemcpy(&nrm2sqr, d_nrm2sqr, sizeof(nrm2sqr), cudaMemcpyDeviceToHost);
    if (err)
      return ACG_ERR_CUDA;
    cg->rnrm2 = sqrt(nrm2sqr);

    cg->ntotaliterations++;
    cg->niterations++;
    cg->ndot += 4;

    if ((residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      converged = true;
      break;
    }

    /*
     * Breakdown / stagnation detection. The next iteration's scalar updates
     * divide by ω and by ρ_prev = (r̂,r); a classic BiCGStab breakdown makes
     * those collapse toward zero, and rounding drift past the attainable
     * accuracy can make the recurrence residual non-finite. Stop here rather
     * than iterating into Inf/NaN — this mirrors PETSc's
     * KSP_DIVERGED_BREAKDOWN / KSP_DIVERGED_NANORINF.
     *
     * Skipped entirely in fixed-iteration benchmark mode (all tolerances
     * zero): there the caller wants exactly ‘maxits’ iterations for timing,
     * and the per-iteration cost is value-independent, so we let it run to
     * completion without the early break or the extra device-to-host copies.
     */
    if (!fixed_iterations)
    {
      double omega, rho_prev;
      err = cudaMemcpy(&omega, d_omega, sizeof(omega), cudaMemcpyDeviceToHost);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaMemcpy(&rho_prev, d_rho_prev, sizeof(rho_prev), cudaMemcpyDeviceToHost);
      if (err)
        return ACG_ERR_CUDA;
      if (!isfinite(cg->rnrm2))
      {
        breakdown = true;
        fprintf(stderr,
                "%s: non-finite residual norm at iteration %d; stopping "
                "(KSP_DIVERGED_NANORINF analogue)\n",
                __func__, cg->niterations);
        break;
      }
      if (omega == 0.0 || !isfinite(omega) || rho_prev == 0.0 || !isfinite(rho_prev))
      {
        breakdown = true;
        fprintf(stderr,
                "%s: BiCGStab breakdown at iteration %d "
                "(omega=%.*g, rho=%.*g, residual norm=%.*g); stopping "
                "(KSP_DIVERGED_BREAKDOWN analogue)\n",
                __func__, cg->niterations, DBL_DIG, omega, DBL_DIG, rho_prev,
                DBL_DIG, cg->rnrm2);
        break;
      }
    }
  }
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

  /* copy solution back to host */
  /* the streams are non-blocking, so the default-stream copy below is not
   * ordered against the solver loop; drain it first */
  cudaStreamSynchronize(stream);
  err = cudaMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecp);
  cusparseDestroyDnVec(vecn);
  cusparseDestroyDnVec(vecq);
  cusparseDestroyDnVec(vect);
  cusparseDestroyDnVec(vecy);
  cusparseDestroyDnVec(vecz);
  cusparseDestroyDnVec(vecm);
  cusparseDestroyDnVec(vecu);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecyo);
    cusparseDestroyDnVec(vecqo);
    cusparseDestroyDnVec(veczo);
    cusparseDestroyDnVec(vecto);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
  if (preconditioner == 2)
  {
    if (matM_lower)
      cusparseDestroySpMat(matM_lower);
    if (matM_upper)
      cusparseDestroySpMat(matM_upper);
    if (spL_p)
      cusparseSpSV_destroyDescr(spL_p);
    if (spU_y)
      cusparseSpSV_destroyDescr(spU_y);
    if (spL_s)
      cusparseSpSV_destroyDescr(spL_s);
    if (spU_z)
      cusparseSpSV_destroyDescr(spU_z);
    if (spL_t)
      cusparseSpSV_destroyDescr(spL_t);
    if (spU_u)
      cusparseSpSV_destroyDescr(spU_u);
    cudaFree(d_bufL_p);
    cudaFree(d_bufU_y);
    cudaFree(d_bufL_s);
    cudaFree(d_bufU_z);
    cudaFree(d_bufL_t);
    cudaFree(d_bufU_u);
    cudaFree(d_M_values);
  }
#else
  cudaFree(d_startrows);
#endif
  if (preconditioner == 1)
    cudaFree(d_M_inv);

  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_rho);
  cudaFree(d_rho_prev);
  cudaFree(d_omega);
  cudaFree(d_rhat_v);
  cudaFree(d_nrm2sqr);
  cudaFree(d_dots);
  cudaEventDestroy(dotEvent);
  cudaEventDestroy(yreadytosend);
  cudaEventDestroy(yreceived);
  cudaEventDestroy(zreadytosend);
  cudaEventDestroy(zreceived);
  cudaEventDestroy(xreadytosend);
  cudaEventDestroy(xreceived);
  cudaStreamDestroy(stream);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(collective_stream);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  if (converged)
    return ACG_SUCCESS;
  /* a breakdown is an early stop, so report it as not converged even in
   * fixed-iteration benchmark mode (mirrors the PETSc wrapper's handling of
   * a negative KSPConvergedReason) */
  if (breakdown)
    return ACG_ERR_NOT_CONVERGED;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;
  return ACG_ERR_NOT_CONVERGED;
}

/*
 * ‘pbicgstab_reduce_begin()’ starts an in-place sum-allreduce of a small
 * device buffer that is meant to be overlapped with a subsequent SpMV. For
 * a split NCCL communicator the reduction is enqueued on a dedicated
 * collective stream (after waiting for the producing dot products on the
 * compute stream); for a plain NCCL communicator it is enqueued directly on
 * the compute stream; for MPI a non-blocking MPI_Iallreduce is launched. It
 * is a no-op for a single process or when collectives are disabled.
 */
static int pbicgstab_reduce_begin(
    double *d_buf,
    int count,
    struct acgcomm *comm,
    int commsize,
    int nocomm_allreduce,
    cudaStream_t stream,
    cudaStream_t collective_stream,
    cudaEvent_t dotEvent,
    MPI_Request *request,
    int *errcode)
{
  int err;
  if (!(commsize > 1 && !nocomm_allreduce))
    return ACG_SUCCESS;
  if (comm->type == acgcomm_nccl_split)
  {
    err = cudaEventRecord(dotEvent, stream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaStreamWaitEvent(collective_stream, dotEvent, 0);
    if (err)
      return ACG_ERR_CUDA;
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, collective_stream,
        comm, errcode);
    if (err)
      return err;
  }
  else if (comm->type == acgcomm_nccl)
  {
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, stream,
        comm, errcode);
    if (err)
      return err;
  }
  else if (comm->type == acgcomm_mpi)
  {
    cudaStreamSynchronize(stream);
    err = MPI_Iallreduce(
        MPI_IN_PLACE, d_buf, count, MPI_DOUBLE, MPI_SUM, comm->mpicomm, request);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_MPI;
    }
  }
  return ACG_SUCCESS;
}

/*
 * ‘pbicgstab_reduce_end()’ makes the compute stream wait for the reduction
 * started by ‘pbicgstab_reduce_begin()’ to complete, so the reduced values
 * may be consumed safely.
 */
static int pbicgstab_reduce_end(
    struct acgcomm *comm,
    int commsize,
    int nocomm_allreduce,
    cudaStream_t stream,
    cudaStream_t collective_stream,
    cudaEvent_t redEvent,
    MPI_Request *request)
{
  int err;
  if (!(commsize > 1 && !nocomm_allreduce))
    return ACG_SUCCESS;
  if (comm->type == acgcomm_nccl_split)
  {
    err = cudaEventRecord(redEvent, collective_stream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaStreamWaitEvent(stream, redEvent, 0);
    if (err)
      return ACG_ERR_CUDA;
  }
  else if (comm->type == acgcomm_nccl)
  {
    /* the allreduce was enqueued on the compute stream itself, so it is
     * already ordered before any subsequent work; nothing to wait for. */
  }
  else if (comm->type == acgcomm_mpi)
  {
    err = MPI_Wait(request, MPI_STATUS_IGNORE);
    if (err)
      return ACG_ERR_MPI;
  }
  return ACG_SUCCESS;
}

/*
 * ‘pbicgstab_reduce_ready_stream()’ returns the stream whose completion makes
 * the values reduced by ‘pbicgstab_reduce_begin()’ readable, which is where
 * the reduction was enqueued. It returns ‘NULL’ when the reduction is not
 * carried out by the device at all, so that the values are only in place once
 * ‘pbicgstab_reduce_end()’ has returned.
 */
static cudaStream_t pbicgstab_reduce_ready_stream(
    const struct acgcomm *comm,
    int commsize,
    int nocomm_allreduce,
    cudaStream_t stream,
    cudaStream_t collective_stream)
{
  if (commsize > 1 && !nocomm_allreduce)
  {
    if (comm->type == acgcomm_nccl_split)
      return collective_stream;
    if (comm->type == acgcomm_nccl)
      return stream;
    /* a host-driven MPI_Iallreduce is only complete after its MPI_Wait() */
    return NULL;
  }
  /* nothing was reduced, so the local dot products are the last word */
  return stream;
}

/*
 * ‘pbicgstab_scalars_copy_begin()’ enqueues asynchronous device-to-host copies
 * of the scalars that the host needs for the convergence and breakdown tests:
 * the reduced ‖r‖₂², and, if ‘nscalars’ exceeds one, ω and ρ as well. The
 * copies go on a stream of their own and are ordered after ‘srcstream’, the
 * stream that produced the values, so that they do not depend on -- and the
 * host does not end up waiting for -- the rest of the compute stream.
 *
 * ‘copiedEvent’ is recorded on ‘copystream’ once the copies are enqueued. The
 * caller must make the compute stream wait for it before enqueueing anything
 * that overwrites a source: ρ in particular, which
 * ‘acgsolvercuda_pipelined_bicgstab_scalars()’ updates in place.
 */
static int pbicgstab_scalars_copy_begin(
    double *h_scalars,
    int nscalars,
    const double *d_rr,
    const double *d_omega,
    const double *d_rho,
    cudaStream_t srcstream,
    cudaStream_t copystream,
    cudaEvent_t srcEvent,
    cudaEvent_t copiedEvent)
{
  int err = cudaEventRecord(srcEvent, srcstream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaStreamWaitEvent(copystream, srcEvent, 0);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      &h_scalars[0], d_rr, sizeof(*d_rr), cudaMemcpyDeviceToHost, copystream);
  if (err)
    return ACG_ERR_CUDA;
  if (nscalars > 1)
  {
    err = cudaMemcpyAsync(
        &h_scalars[1], d_omega, sizeof(*d_omega), cudaMemcpyDeviceToHost,
        copystream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMemcpyAsync(
        &h_scalars[2], d_rho, sizeof(*d_rho), cudaMemcpyDeviceToHost,
        copystream);
    if (err)
      return ACG_ERR_CUDA;
  }
  err = cudaEventRecord(copiedEvent, copystream);
  if (err)
    return ACG_ERR_CUDA;
  return ACG_SUCCESS;
}

/**
 * ‘acgsolvercuda_solve_pipelined_bicgstab()’ solves the given linear system,
 * Ax=b, using the communication-hiding pipelined BiCGStab method (Cools &
 * Vanroose, 2017), without a preconditioner. The linear system may be
 * distributed across multiple processes; the two global reductions per
 * iteration are overlapped with the two sparse matrix-vector products.
 */
int acgsolvercuda_solve_pipelined_bicgstab(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

#if defined(ACG_NOCOMM)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE) && defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE)
  int nocomm_allreduce = 1, nocomm_p2p = 0;
#elif defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 0, nocomm_p2p = 1;
#else
  int nocomm_allreduce = 0, nocomm_p2p = 0;
#endif

  /* allocate the extra work vectors (reuses the pipelined-CG storage) */
  struct acgvector **vecptrs[] = {&cg->w, &cg->q, &cg->z, &cg->m, &cg->n, &cg->u, &cg->y};
  double **devptrs[] = {&cg->d_w, &cg->d_q, &cg->d_z, &cg->d_m, &cg->d_n, &cg->d_u, &cg->d_y};
  for (int i = 0; i < 7; i++)
  {
    if (!*vecptrs[i])
    {
      *vecptrs[i] = malloc(sizeof(struct acgvector));
      if (!*vecptrs[i])
        return ACG_ERR_ERRNO;
      err = acgvector_init_copy(*vecptrs[i], x);
      if (err)
        return err;
      err = cudaMalloc((void **)devptrs[i], (*vecptrs[i])->num_nonzeros * sizeof(double));
      if (err)
        return ACG_ERR_CUDA;
    }
  }

  /*
   * Vector role mapping for pipelined BiCGStab:
   *   d_r = r        d_n = r̂0 (shadow, constant)   d_p = p
   *   d_m = s        d_z = z = A·s                  d_t = t = A·w
   *   d_u = v = A·z  d_w = w = A·r                  d_q = q = r − α·s
   *   d_y = y = w − α·z = A·q
   */
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_r = cg->d_r;    /* residual r */
  double *d_rhat = cg->d_n; /* r̂0 */
  double *d_p = cg->d_p;
  double *d_s = cg->d_m;
  double *d_z = cg->d_z;
  double *d_t = cg->d_t;
  double *d_v = cg->d_u;
  double *d_w = cg->d_w;
  double *d_q = cg->d_q;
  double *d_y = cg->d_y;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;
  int nnz_full = x->num_nonzeros;
  int n_owned = cg->r.num_nonzeros - cg->r.num_ghost_nonzeros;

  /* BiCGStab device-side scalars and reduction buffers */
  double *d_alpha, *d_beta, *d_omega, *d_rho, *d_R1, *d_R2;
  err = cudaMalloc((void **)&d_alpha, sizeof(*d_alpha));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_beta, sizeof(*d_beta));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_omega, sizeof(*d_omega));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_rho, sizeof(*d_rho));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_R1, 2 * sizeof(*d_R1));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_R2, 5 * sizeof(*d_R2));
  if (err)
    return ACG_ERR_CUDA;
  double *d_qy = &d_R1[0]; /* (q,y) */
  double *d_yy = &d_R1[1]; /* (y,y) */
  double *d_d1 = &d_R2[0]; /* (r̂0,r) */
  double *d_d2 = &d_R2[1]; /* (r̂0,w) */
  double *d_d3 = &d_R2[2]; /* (r̂0,s) */
  double *d_d4 = &d_R2[3]; /* (r̂0,z) */
  double *d_rr = &d_R2[4]; /* (r,r) for the residual norm */

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  int leastPriority, greatestPriority;
  err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t stream;
  err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t collective_stream;
  err = cudaStreamCreateWithPriority(&collective_stream, cudaStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t commstream;
  err = cudaStreamCreateWithPriority(&commstream, cudaStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_CUDA;
  acgSetStreamName(commstream, "P2P");
  acgSetStreamName(collective_stream, "Allreduce");
  acgSetStreamName(stream, "Compute");

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  cudaEvent_t dotEvent, redEvent, haloReady, haloRecv;
  cudaEventCreateWithFlags(&dotEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&redEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&haloRecv, cudaEventDisableTiming);
  MPI_Request request;

  /* a dedicated stream for fetching the handful of scalars that the host needs
   * for the convergence tests, together with pinned staging memory for ‖r‖₂²,
   * ω and ρ, so that those transfers neither block nor are blocked by the work
   * queued on the compute stream */
  cudaStream_t copystream;
  err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  acgSetStreamName(copystream, "Scalars");
  cudaEvent_t scalarsReduced, scalarsCopied;
  cudaEventCreateWithFlags(&scalarsReduced, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&scalarsCopied, cudaEventDisableTiming);
  double *h_scalars;
  err = cudaMallocHost((void **)&h_scalars, 3 * sizeof(*h_scalars));
  if (err)
    return ACG_ERR_CUDA;

  /* copy right-hand side and initial guess to device */
  double *d_b, *d_x;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  cusparseDnVecDescr_t vecx, vecr, vecw, vect, vecz, vecv;
  cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  cusparseCreateDnVec(&vecw, A->nownedrows, d_w, CUDA_R_64F);
  cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
  cusparseCreateDnVec(&vecz, A->nownedrows, d_z, CUDA_R_64F);
  cusparseCreateDnVec(&vecv, A->nownedrows, d_v, CUDA_R_64F);
  cusparseDnVecDescr_t vecxo, vecro, vecwo, vecto, veczo, vecvo;
  if (commsize > 1)
  {
    int no = A->nborderrows + A->nghostrows;
    cusparseCreateDnVec(&vecxo, no, d_x + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecro, no, d_r + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecwo, no, d_w + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecto, no, d_t + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&veczo, no, d_z + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecvo, no, d_v + A->borderrowoffset, CUDA_R_64F);
  }
  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
    return ACG_ERR_CUSPARSE;
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
    return ACG_ERR_CUSPARSE;
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;
  cusparseSpMatDescr_t matO = NULL;
  void *d_obuffer = NULL;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows,
        A->onpnzs, d_orowptr, d_ocolidx, d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
      return ACG_ERR_CUSPARSE;
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo,
        d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
  }
  acgidx_t nstartrows_v = 0;
  acgidx_t *d_startrows = NULL;
#else
  cusparseSpMatDescr_t matA = NULL, matO = NULL;
  cusparseDnVecDescr_t vecx = NULL, vecr = NULL, vecw = NULL, vect = NULL,
                       vecz = NULL, vecv = NULL;
  cusparseDnVecDescr_t vecxo = NULL, vecro = NULL, vecwo = NULL, vecto = NULL,
                       veczo = NULL, vecvo = NULL;
  void *d_buffer = NULL, *d_obuffer = NULL;
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows_v = ntasks / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = cudaMalloc((void **)&d_startrows, nstartrows_v * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_CUDA;
  err = acgsolvercuda_csrgemv_merge_init(
      (A->nprows - A->nghostrows), d_rowptr, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
  err = cudaStreamSynchronize(stream);
  if (err)
    return ACG_ERR_CUDA;
#endif

  err = cublasSetStream(cublas, stream);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cusparseSetStream(cusparse, stream);
  if (err)
    return ACG_ERR_CUSPARSE;

  /*
   * Zero the work vectors and set the scalars to one, so that the arithmetic
   * of the warmup iterations below is well defined and stays well
   * conditioned: with β = ω = α = ρ = 1, and with d₁…d₄ taken from ‘d_one’
   * rather than from the reductions, the fused updates and the scalar
   * recurrence reproduce their inputs instead of drifting towards zero or
   * infinity. Every buffer touched here is assigned its real initial value
   * after the timing starts, so none of this affects the solve; the
   * right-hand side ‘d_b’ and the initial guess ‘d_x’ are never written.
   */
  double *d_scratchvecs[] = {d_r, d_rhat, d_p, d_s, d_z, d_t, d_v, d_w, d_q, d_y};
  int nscratchvecs = (int)(sizeof(d_scratchvecs) / sizeof(*d_scratchvecs));
  for (int i = 0; i < nscratchvecs; i++)
  {
    err = cudaMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_CUDA;
  }
  double *d_scratchscalars[] = {d_alpha, d_beta, d_omega, d_rho};
  for (int i = 0; i < (int)(sizeof(d_scratchscalars) / sizeof(*d_scratchscalars)); i++)
  {
    err = cudaMemcpyAsync(
        d_scratchscalars[i], d_one, sizeof(**d_scratchscalars),
        cudaMemcpyDeviceToDevice, stream);
    if (err)
      return ACG_ERR_CUDA;
  }

#if defined(ACG_USE_CUSPARSE)
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
  /* let cuSPARSE analyse the two matrices up front, so that the analysis is
   * not repeated on each of the four SpMV calls per iteration */
  err = cusparseSpMV_preprocess(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecz, d_zero,
      vecv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  if (commsize > 1)
  {
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, veczo, d_one,
        vecvo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
#endif
#endif

  /*
   * Warmup iterations.
   *
   * NCCL establishes the point-to-point connections and the collective
   * channels of a communicator lazily, on the first operation that needs
   * them; cuSPARSE selects its SpMV kernel on the first call for a given
   * descriptor; cuBLAS loads its modules on demand; and freshly allocated
   * device memory is faulted in on first touch. Those costs grow with the
   * number of neighbouring processes and with the size of the communicator,
   * so leaving them inside the timed region shows up as a penalty that gets
   * worse the further the solver is scaled out. Run through every operation
   * of the solver loop ‘warmup’ times to pay for them here instead.
   *
   * ‘warmup’ is the same on every process and the collectives are enqueued in
   * the same order as in the solver loop, as NCCL requires. ‘d_rhat’ stands
   * in for the iterate, since it is only assigned once the timing starts.
   */
  for (int i = 0; i < warmup; i++)
  {
    /* the copy and the ‖b‖₂ reduction that precede the loop */
    err = acgsolvercuda_dcopy(n_owned, d_r, d_b, stream);
    if (err)
      return err;
    err = cublasDdot(cublas, n_owned, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* steps 1-5: the p,s,z and q,y recurrences */
    err = acgsolvercuda_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;
    err = acgsolvercuda_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* the first reduction, overlapped with v = A·z */
    err = cublasDdot(cublas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        numSMs, stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
        d_one, d_zero, cusparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
        nstartrows_v, d_startrows);
    if (err)
      return err;
    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11, with ω taken from ‘d_one’ rather than from the reduction */
    err = acgsolvercuda_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_one, d_one, d_alpha, d_p, d_q, d_y, d_t, d_v, d_rhat, d_r, d_w, stream);
    if (err)
      return err;

    /* the second reduction, overlapped with t = A·w */
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R2, 5, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
        d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
        nstartrows_v, d_startrows);
    if (err)
      return err;
    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* the asynchronous fetch of the scalars that the convergence tests read.
     * Sourced from the compute stream, which ‘pbicgstab_reduce_end()’ has just
     * ordered after the reduction, so that this is correct for every
     * communicator type. */
    err = pbicgstab_scalars_copy_begin(
        h_scalars, 3, d_rr, d_omega, d_rho, stream, copystream, scalarsReduced,
        scalarsCopied);
    if (err)
      return err;
    err = cudaStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_CUDA;

    /* steps 14-15, again fed from ‘d_one’ to keep β and α at one */
    err = acgsolvercuda_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_one, d_one, d_one, d_one, stream);
    if (err)
      return err;

    err = cudaStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_CUDA;
  }

  /* discard what the warmup left behind; the scalars and the owned parts of
   * the vectors are assigned again by the initialisation below, and the ghost
   * parts by the halo exchange preceding each SpMV that reads them */
  for (int i = 0; i < nscratchvecs; i++)
  {
    err = cudaMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_CUDA;
  }

  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;

  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  cudaStreamSynchronize(commstream);
  cudaStreamSynchronize(collective_stream);
  cudaStreamSynchronize(copystream);
  gettime(&t0);

  /* ‖b‖₂ */
  err = cublasDdot(cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
  if (err)
    return ACG_ERR_CUBLAS;
  err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  double bnrm2sqr;
  err = cudaMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* r₀ = b − A·x₀ */
  err = acgsolvercuda_dcopy(b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
  if (err)
    return err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = cudaEventRecord(haloReady, stream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaStreamWaitEvent(commstream, haloReady, 0);
    if (err)
      return ACG_ERR_CUDA;
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x,
        ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_CUSPARSE)
  err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
    return ACG_ERR_CUSPARSE;
#else
  err = acgsolvercuda_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1, nstartrows_v, d_startrows, numSMs, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_cuda_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
      err = cudaEventRecord(haloRecv, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, haloRecv, 0);
      if (err)
        return ACG_ERR_CUDA;
    }
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
      return ACG_ERR_CUSPARSE;
#else
    err = csrgemv_host(A->nborderrows, d_r + A->borderrowoffset, d_x + A->borderrowoffset, d_orowptr, d_ocolidx, d_oa, -1.0, numSMs, stream);
    if (err)
      return err;
#endif
  }
  cg->ngemv++;

  /* r̂0 = r₀ */
  err = acgsolvercuda_dcopy(n_owned, d_rhat, d_r, stream);
  if (err)
    return err;

  /* w₀ = A·r₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      numSMs, stream, commstream, nnz_full, d_r, d_w, haloReady, haloRecv,
      d_one, d_zero, cusparse, matA, matO, vecr, vecw, vecro, vecwo, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
      nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* t₀ = A·w₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
      d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
      nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* initial dot products: (r₀,r₀) and (r₀,w₀) */
  err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, &d_R2[0]);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasDdot(cublas, n_owned, d_r, 1, d_w, 1, &d_R2[1]);
  if (err)
    return ACG_ERR_CUBLAS;
  err = bicgstab_allreduce(d_R2, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  double rr0, rw0;
  err = cudaMemcpy(&rr0, &d_R2[0], sizeof(rr0), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(&rw0, &d_R2[1], sizeof(rw0), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->r0nrm2 = cg->rnrm2 = sqrt(rr0);
  if (residualrtol > 0)
    residualrtol *= cg->r0nrm2;

  /*
   * Initial scalars: α₀ = (r,r)/(r,w), ρ₀ = (r,r), β = ω = 0, and p = s = z =
   * v = 0 (the β₋₁ = 0 recurrence then yields p₀=r, s₀=w, z₀=t).
   *
   * All of this must be enqueued on the compute stream. The solver's streams
   * are created with ‘cudaStreamNonBlocking’, so they are not ordered against
   * the legacy default stream that the synchronous ‘cudaMemcpy()’ and
   * ‘cudaMemset()’ would use. Those calls are also asynchronous with respect
   * to the host for device memory, and the four vector fills are large -- a
   * few hundred megabytes on a large system -- so they can still be in flight
   * once the solver loop below has started, and land on top of the p, s, z and
   * v that the first iterations have already updated.
   */
  double alpha0 = rr0 / rw0;
  err = cudaMemcpyAsync(
      d_alpha, &alpha0, sizeof(alpha0), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_rho, &rr0, sizeof(rr0), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_beta, d_zero, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_omega, d_zero, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_p, 0, nnz_full * sizeof(*d_p), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_s, 0, nnz_full * sizeof(*d_s), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_z, 0, nnz_full * sizeof(*d_z), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_v, 0, nnz_full * sizeof(*d_v), stream);
  if (err)
    return ACG_ERR_CUDA;
  /* ‘alpha0’ and ‘rr0’ are read by the asynchronous copies above, so they must
   * stay put until those have run */
  err = cudaStreamSynchronize(stream);
  if (err)
    return ACG_ERR_CUDA;

  /* iterative solver loop */
  bool breakdown = false;
  /* fixed-iteration benchmark mode: no stopping criteria are set, so run
   * exactly ‘maxits’ iterations and disable the breakdown early-out */
  bool fixed_iterations =
      (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0);
  /* the stream that makes a reduction readable, and how many scalars the host
   * needs each iteration: the residual norm always, ω and ρ only when the
   * breakdown test is active */
  cudaStream_t readystream = pbicgstab_reduce_ready_stream(
      comm, commsize, nocomm_allreduce, stream, collective_stream);
  int nscalars = fixed_iterations ? 1 : 3;
  for (int k = 0; k < maxits; k++)
  {
    /* steps 1-3: p,s,z recurrences */
    err = acgsolvercuda_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;

    /*
    begin halo exchange
    */
    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
          ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
    }

    /* steps 4-5: q = r − α·s, y = w − α·z */
    err = acgsolvercuda_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* reduction R1 = {(q,y),(y,y)}, overlapped with v = A·z */
    err = cublasDdot(cublas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;

    // spmv diag

#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecz, d_zero,
        vecv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_v, d_z, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows, d_startrows, numSMs, stream);
    if (err)
      return err;
#endif

    // err = bicgstab_spmv(
    //     A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
    //     numSMs, stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
    //     d_one, d_zero, cusparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
    //     d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
    //     nstartrows_v, d_startrows);
    // if (err)
    //   return err;
    // cg->ngemv++;

    // end halo exchange
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
            ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
        if (err)
          return err;
        err = cudaEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_CUDA;
      }
// spmv off-diagonal
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, veczo, d_one,
          vecvo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(
          A->nborderrows, d_v + A->borderrowoffset, d_z + A->borderrowoffset,
          d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
      if (err)
        return err;
#endif
    }

    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11: ω, then x, r, w updates */
    err = acgsolvercuda_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_qy, d_yy, d_alpha, d_p, d_q, d_y, d_t, d_v, d_x, d_r, d_w, stream);
    if (err)
      return err;

    // begin halo exchange
    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
          ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
    }

    /* reduction R2 = {(r̂0,r),(r̂0,w),(r̂0,s),(r̂0,z),(r,r)}, overlapped with t = A·w */
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R2, 5, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;

    /* Start bringing the reduced residual norm to the host as soon as the
     * reduction itself completes, rather than after the sparse matrix-vector
     * product that is meant to hide the reduction. */
    if (readystream)
    {
      err = pbicgstab_scalars_copy_begin(
          h_scalars, nscalars, d_rr, d_omega, d_rho, readystream, copystream,
          scalarsReduced, scalarsCopied);
      if (err)
        return err;
    }
// err = bicgstab_spmv(
//     A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
//     numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
//     d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
//     d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
//     nstartrows_v, d_startrows);
// if (err)
//   return err;
// cg->ngemv++;

// spmv diag
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecw, d_zero,
        vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_t, d_w, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows, d_startrows, numSMs, stream);
    if (err)
      return err;
#endif

    // end halo exchange
    // spmv off-diagonal
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
            ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
        if (err)
          return err;
        err = cudaEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_CUDA;
      }
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecwo, d_one,
          vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(
          A->nborderrows, d_t + A->borderrowoffset, d_w + A->borderrowoffset,
          d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
      if (err)
        return err;
#endif
    }

    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* the MPI reduction is host-driven, so its result only became readable
     * when ‘pbicgstab_reduce_end()’ completed the request just above */
    if (!readystream)
    {
      err = pbicgstab_scalars_copy_begin(
          h_scalars, nscalars, d_rr, d_omega, d_rho, stream, copystream,
          scalarsReduced, scalarsCopied);
      if (err)
        return err;
    }

    /* the scalar update overwrites ρ in place, so hold it back until the copy
     * above has read the old value */
    err = cudaStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_CUDA;

    /* steps 14-15: β and α for the next iteration. Enqueued before the host
     * blocks on the residual norm below, so that the first dependency of the
     * next iteration is ready as soon as the trailing t = A·w retires. Its
     * result goes unused if one of the tests below leaves the loop. */
    err = acgsolvercuda_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_d1, d_d2, d_d3, d_d4, stream);
    if (err)
      return err;

    /* residual norm for convergence. Only the few bytes staged on
     * ‘copystream’ are waited for, and they were gated on the reduction alone,
     * so the trailing sparse matrix-vector product, the scalar update and the
     * launches of the next iteration all overlap with this. Synchronising on
     * the compute stream instead would drain the whole queue every
     * iteration. */
    err = cudaStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_CUDA;
    cg->rnrm2 = sqrt(h_scalars[0]);
    cg->ntotaliterations++;
    cg->niterations++;
    cg->ndot += 7;

    if ((residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      converged = true;
      break;
    }

    /*
     * Breakdown / stagnation detection. The pipelined BiCGStab recurrences
     * lose accuracy once the residual reaches the attainable precision, and
     * the scalar update ‘acgsolvercuda_pipelined_bicgstab_scalars()’ divides
     * by ω and by ρ = (r̂0,r). If those collapse toward zero (a classic
     * BiCGStab breakdown) or the recurrence residual becomes non-finite, the
     * iterates blow up. Stop here rather than iterating into Inf/NaN — this
     * mirrors PETSc's KSP_DIVERGED_BREAKDOWN / KSP_DIVERGED_NANORINF, so both
     * solvers terminate at the same point instead of diverging.
     *
     * Skipped entirely in fixed-iteration benchmark mode (all tolerances
     * zero): there the caller wants exactly ‘maxits’ iterations for timing,
     * and the per-iteration cost is value-independent, so we let it run to
     * completion without the early break, and ‘nscalars’ leaves ω and ρ on
     * the device.
     */
    if (!fixed_iterations)
    {
      double omega = h_scalars[1], rho_old = h_scalars[2];
      if (!isfinite(cg->rnrm2))
      {
        breakdown = true;
        fprintf(stderr,
                "%s: non-finite residual norm at iteration %d; stopping "
                "(KSP_DIVERGED_NANORINF analogue)\n",
                __func__, cg->niterations);
        break;
      }
      if (omega == 0.0 || !isfinite(omega) || rho_old == 0.0 || !isfinite(rho_old))
      {
        breakdown = true;
        fprintf(stderr,
                "%s: BiCGStab breakdown at iteration %d "
                "(omega=%.*g, rho=%.*g, residual norm=%.*g); stopping "
                "(KSP_DIVERGED_BREAKDOWN analogue)\n",
                __func__, cg->niterations, DBL_DIG, omega, DBL_DIG, rho_old,
                DBL_DIG, cg->rnrm2);
        break;
      }
    }
  }
  /* The loop no longer drains the compute stream every iteration, so the
   * trailing sparse matrix-vector product and scalar update of the last
   * iteration may still be in flight. Drain once here, before stopping the
   * clock, so that the reported time covers all of the work -- and so that the
   * solution copied out below is complete. The streams are non-blocking, so
   * the copy on the default stream would not synchronise with them. */
  cudaStreamSynchronize(stream);
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

  /* copy solution back to host */
  err = cudaMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecw);
  cusparseDestroyDnVec(vect);
  cusparseDestroyDnVec(vecz);
  cusparseDestroyDnVec(vecv);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecwo);
    cusparseDestroyDnVec(vecto);
    cusparseDestroyDnVec(veczo);
    cusparseDestroyDnVec(vecvo);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
#else
  cudaFree(d_startrows);
#endif

  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  cudaFree(d_omega);
  cudaFree(d_rho);
  cudaFree(d_R1);
  cudaFree(d_R2);
  cudaEventDestroy(dotEvent);
  cudaEventDestroy(redEvent);
  cudaEventDestroy(haloReady);
  cudaEventDestroy(haloRecv);
  cudaStreamDestroy(stream);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(collective_stream);
  cudaEventDestroy(scalarsReduced);
  cudaEventDestroy(scalarsCopied);
  cudaStreamDestroy(copystream);
  cudaFreeHost(h_scalars);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  if (converged)
    return ACG_SUCCESS;
  /* a breakdown is an early stop, so report it as not converged even in
   * fixed-iteration benchmark mode (mirrors the PETSc wrapper's handling of
   * a negative KSPConvergedReason) */
  if (breakdown)
    return ACG_ERR_NOT_CONVERGED;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;
  return ACG_ERR_NOT_CONVERGED;
}

#ifndef ACG_BICGSTAB_RR_PERIOD
#define ACG_BICGSTAB_RR_PERIOD 100 /* residual-replacement period (0 = off) */
#endif
#ifndef ACG_BICGSTAB_RR_MAXIT
#define ACG_BICGSTAB_RR_MAXIT 1001 /* stop replacing past this iteration (<=0 = no cap) */
#endif

/**
 * ‘acgsolvercuda_solve_pipelined_bicgstab_rr()’ solves the given linear
 * system, Ax=b, using the communication-hiding pipelined BiCGStab method
 * (Cools & Vanroose, 2017), without a preconditioner, augmented with
 * PETSc-style periodic residual replacement: every ACG_BICGSTAB_RR_PERIOD
 * iterations the recurrence-propagated vectors r,w,t,s,z,v are recomputed
 * from the primary vectors x and p with explicit SpMVs, resetting the
 * accumulated rounding error (mirrors PETSc KSPPIPEBCGS). The linear system
 * may be distributed across multiple processes; the two global reductions
 * per iteration are overlapped with the two sparse matrix-vector products.
 */
int acgsolvercuda_solve_pipelined_bicgstab_rr(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

#if defined(ACG_NOCOMM)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE) && defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 1, nocomm_p2p = 1;
#elif defined(ACG_NOCOMM_ALLREDUCE)
  int nocomm_allreduce = 1, nocomm_p2p = 0;
#elif defined(ACG_NOCOMM_P2P)
  int nocomm_allreduce = 0, nocomm_p2p = 1;
#else
  int nocomm_allreduce = 0, nocomm_p2p = 0;
#endif

  /* allocate the extra work vectors (reuses the pipelined-CG storage) */
  struct acgvector **vecptrs[] = {&cg->w, &cg->q, &cg->z, &cg->m, &cg->n, &cg->u, &cg->y};
  double **devptrs[] = {&cg->d_w, &cg->d_q, &cg->d_z, &cg->d_m, &cg->d_n, &cg->d_u, &cg->d_y};
  for (int i = 0; i < 7; i++)
  {
    if (!*vecptrs[i])
    {
      *vecptrs[i] = malloc(sizeof(struct acgvector));
      if (!*vecptrs[i])
        return ACG_ERR_ERRNO;
      err = acgvector_init_copy(*vecptrs[i], x);
      if (err)
        return err;
      err = cudaMalloc((void **)devptrs[i], (*vecptrs[i])->num_nonzeros * sizeof(double));
      if (err)
        return ACG_ERR_CUDA;
    }
  }

  /*
   * Vector role mapping for pipelined BiCGStab:
   *   d_r = r        d_n = r̂0 (shadow, constant)   d_p = p
   *   d_m = s        d_z = z = A·s                  d_t = t = A·w
   *   d_u = v = A·z  d_w = w = A·r                  d_q = q = r − α·s
   *   d_y = y = w − α·z = A·q
   */
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_r = cg->d_r;    /* residual r */
  double *d_rhat = cg->d_n; /* r̂0 */
  double *d_p = cg->d_p;
  double *d_s = cg->d_m;
  double *d_z = cg->d_z;
  double *d_t = cg->d_t;
  double *d_v = cg->d_u;
  double *d_w = cg->d_w;
  double *d_q = cg->d_q;
  double *d_y = cg->d_y;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;
  int nnz_full = x->num_nonzeros;
  int n_owned = cg->r.num_nonzeros - cg->r.num_ghost_nonzeros;

  /* BiCGStab device-side scalars and reduction buffers */
  double *d_alpha, *d_beta, *d_omega, *d_rho, *d_R1, *d_R2;
  err = cudaMalloc((void **)&d_alpha, sizeof(*d_alpha));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_beta, sizeof(*d_beta));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_omega, sizeof(*d_omega));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_rho, sizeof(*d_rho));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_R1, 2 * sizeof(*d_R1));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_R2, 5 * sizeof(*d_R2));
  if (err)
    return ACG_ERR_CUDA;
  double *d_qy = &d_R1[0]; /* (q,y) */
  double *d_yy = &d_R1[1]; /* (y,y) */
  double *d_d1 = &d_R2[0]; /* (r̂0,r) */
  double *d_d2 = &d_R2[1]; /* (r̂0,w) */
  double *d_d3 = &d_R2[2]; /* (r̂0,s) */
  double *d_d4 = &d_R2[3]; /* (r̂0,z) */
  double *d_rr = &d_R2[4]; /* (r,r) for the residual norm */

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  int leastPriority, greatestPriority;
  err = cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t stream;
  err = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t collective_stream;
  err = cudaStreamCreateWithPriority(&collective_stream, cudaStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t commstream;
  err = cudaStreamCreateWithPriority(&commstream, cudaStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_CUDA;
  acgSetStreamName(commstream, "P2P");
  acgSetStreamName(collective_stream, "Allreduce");
  acgSetStreamName(stream, "Compute");

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  cudaEvent_t dotEvent, redEvent, haloReady, haloRecv;
  cudaEventCreateWithFlags(&dotEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&redEvent, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&haloRecv, cudaEventDisableTiming);
  MPI_Request request;

  /* a dedicated stream for fetching the handful of scalars that the host needs
   * for the convergence tests, together with pinned staging memory for ‖r‖₂²,
   * ω and ρ, so that those transfers neither block nor are blocked by the work
   * queued on the compute stream */
  cudaStream_t copystream;
  err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  acgSetStreamName(copystream, "Scalars");
  cudaEvent_t scalarsReduced, scalarsCopied;
  cudaEventCreateWithFlags(&scalarsReduced, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&scalarsCopied, cudaEventDisableTiming);
  double *h_scalars;
  err = cudaMallocHost((void **)&h_scalars, 3 * sizeof(*h_scalars));
  if (err)
    return ACG_ERR_CUDA;

  /* copy right-hand side and initial guess to device */
  double *d_b, *d_x;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  cusparseDnVecDescr_t vecx, vecr, vecw, vect, vecz, vecv;
  cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  cusparseCreateDnVec(&vecw, A->nownedrows, d_w, CUDA_R_64F);
  cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
  cusparseCreateDnVec(&vecz, A->nownedrows, d_z, CUDA_R_64F);
  cusparseCreateDnVec(&vecv, A->nownedrows, d_v, CUDA_R_64F);
  /* extra descriptors for residual replacement (s = A·p, z = A·s) */
  cusparseDnVecDescr_t vecp, vecs;
  cusparseCreateDnVec(&vecp, A->nownedrows, d_p, CUDA_R_64F);
  cusparseCreateDnVec(&vecs, A->nownedrows, d_s, CUDA_R_64F);
  cusparseDnVecDescr_t vecxo, vecro, vecwo, vecto, veczo, vecvo;
  cusparseDnVecDescr_t vecpo = NULL, vecso = NULL;
  if (commsize > 1)
  {
    int no = A->nborderrows + A->nghostrows;
    cusparseCreateDnVec(&vecxo, no, d_x + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecro, no, d_r + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecwo, no, d_w + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecto, no, d_t + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&veczo, no, d_z + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecvo, no, d_v + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecpo, no, d_p + A->borderrowoffset, CUDA_R_64F);
    cusparseCreateDnVec(&vecso, no, d_s + A->borderrowoffset, CUDA_R_64F);
  }
  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
    return ACG_ERR_CUSPARSE;
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
    return ACG_ERR_CUSPARSE;
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;
  cusparseSpMatDescr_t matO = NULL;
  void *d_obuffer = NULL;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows,
        A->onpnzs, d_orowptr, d_ocolidx, d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err)
      return ACG_ERR_CUSPARSE;
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo,
        d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
      return ACG_ERR_CUSPARSE;
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
  }
  acgidx_t nstartrows_v = 0;
  acgidx_t *d_startrows = NULL;
#else
  cusparseSpMatDescr_t matA = NULL, matO = NULL;
  cusparseDnVecDescr_t vecx = NULL, vecr = NULL, vecw = NULL, vect = NULL,
                       vecz = NULL, vecv = NULL;
  cusparseDnVecDescr_t vecxo = NULL, vecro = NULL, vecwo = NULL, vecto = NULL,
                       veczo = NULL, vecvo = NULL;
  cusparseDnVecDescr_t vecp = NULL, vecs = NULL, vecpo = NULL, vecso = NULL;
  void *d_buffer = NULL, *d_obuffer = NULL;
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows_v = ntasks / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = cudaMalloc((void **)&d_startrows, nstartrows_v * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_CUDA;
  err = acgsolvercuda_csrgemv_merge_init(
      (A->nprows - A->nghostrows), d_rowptr, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
  err = cudaStreamSynchronize(stream);
  if (err)
    return ACG_ERR_CUDA;
#endif

  err = cublasSetStream(cublas, stream);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cusparseSetStream(cusparse, stream);
  if (err)
    return ACG_ERR_CUSPARSE;

  /*
   * Zero the work vectors and set the scalars to one, so that the arithmetic
   * of the warmup iterations below is well defined and stays well
   * conditioned: with β = ω = α = ρ = 1, and with d₁…d₄ taken from ‘d_one’
   * rather than from the reductions, the fused updates and the scalar
   * recurrence reproduce their inputs instead of drifting towards zero or
   * infinity. Every buffer touched here is assigned its real initial value
   * after the timing starts, so none of this affects the solve; the
   * right-hand side ‘d_b’ and the initial guess ‘d_x’ are never written.
   */
  double *d_scratchvecs[] = {d_r, d_rhat, d_p, d_s, d_z, d_t, d_v, d_w, d_q, d_y};
  int nscratchvecs = (int)(sizeof(d_scratchvecs) / sizeof(*d_scratchvecs));
  for (int i = 0; i < nscratchvecs; i++)
  {
    err = cudaMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_CUDA;
  }
  double *d_scratchscalars[] = {d_alpha, d_beta, d_omega, d_rho};
  for (int i = 0; i < (int)(sizeof(d_scratchscalars) / sizeof(*d_scratchscalars)); i++)
  {
    err = cudaMemcpyAsync(
        d_scratchscalars[i], d_one, sizeof(**d_scratchscalars),
        cudaMemcpyDeviceToDevice, stream);
    if (err)
      return ACG_ERR_CUDA;
  }

#if defined(ACG_USE_CUSPARSE)
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
  /* let cuSPARSE analyse the two matrices up front, so that the analysis is
   * not repeated on each of the four SpMV calls per iteration */
  err = cusparseSpMV_preprocess(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecz, d_zero,
      vecv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  if (commsize > 1)
  {
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, veczo, d_one,
        vecvo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
#endif
#endif

  /*
   * Warmup iterations.
   *
   * NCCL establishes the point-to-point connections and the collective
   * channels of a communicator lazily, on the first operation that needs
   * them; cuSPARSE selects its SpMV kernel on the first call for a given
   * descriptor; cuBLAS loads its modules on demand; and freshly allocated
   * device memory is faulted in on first touch. Those costs grow with the
   * number of neighbouring processes and with the size of the communicator,
   * so leaving them inside the timed region shows up as a penalty that gets
   * worse the further the solver is scaled out. Run through every operation
   * of the solver loop ‘warmup’ times to pay for them here instead.
   *
   * ‘warmup’ is the same on every process and the collectives are enqueued in
   * the same order as in the solver loop, as NCCL requires. ‘d_rhat’ stands
   * in for the iterate, since it is only assigned once the timing starts.
   */
  for (int i = 0; i < warmup; i++)
  {
    /* the copy and the ‖b‖₂ reduction that precede the loop */
    err = acgsolvercuda_dcopy(n_owned, d_r, d_b, stream);
    if (err)
      return err;
    err = cublasDdot(cublas, n_owned, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* steps 1-5: the p,s,z and q,y recurrences */
    err = acgsolvercuda_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;
    err = acgsolvercuda_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* the first reduction, overlapped with v = A·z */
    err = cublasDdot(cublas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        numSMs, stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
        d_one, d_zero, cusparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
        nstartrows_v, d_startrows);
    if (err)
      return err;
    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11, with ω taken from ‘d_one’ rather than from the reduction */
    err = acgsolvercuda_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_one, d_one, d_alpha, d_p, d_q, d_y, d_t, d_v, d_rhat, d_r, d_w, stream);
    if (err)
      return err;

    /* the second reduction, overlapped with t = A·w */
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R2, 5, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
        d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
        nstartrows_v, d_startrows);
    if (err)
      return err;
    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* the asynchronous fetch of the scalars that the convergence tests read.
     * Sourced from the compute stream, which ‘pbicgstab_reduce_end()’ has just
     * ordered after the reduction, so that this is correct for every
     * communicator type. */
    err = pbicgstab_scalars_copy_begin(
        h_scalars, 3, d_rr, d_omega, d_rho, stream, copystream, scalarsReduced,
        scalarsCopied);
    if (err)
      return err;
    err = cudaStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_CUDA;

    /* steps 14-15, again fed from ‘d_one’ to keep β and α at one */
    err = acgsolvercuda_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_one, d_one, d_one, d_one, stream);
    if (err)
      return err;

    /* the periodic residual-replacement path. Its s = A·p and z = A·s use
     * descriptors that appear nowhere else, so without this their first
     * cuSPARSE call would land inside the timed region, at iteration
     * ACG_BICGSTAB_RR_PERIOD. */
    if (ACG_BICGSTAB_RR_PERIOD > 0)
    {
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_p, d_s, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecp, vecs, vecpo, vecso, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_s, d_z, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecs, vecz, vecso, veczo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
    }

    err = cudaStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_CUDA;
  }

  /* discard what the warmup left behind; the scalars and the owned parts of
   * the vectors are assigned again by the initialisation below, and the ghost
   * parts by the halo exchange preceding each SpMV that reads them */
  for (int i = 0; i < nscratchvecs; i++)
  {
    err = cudaMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_CUDA;
  }
  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;

  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  cudaStreamSynchronize(commstream);
  cudaStreamSynchronize(collective_stream);
  cudaStreamSynchronize(copystream);
  gettime(&t0);

  /* ‖b‖₂ */
  err = cublasDdot(cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
  if (err)
    return ACG_ERR_CUBLAS;
  err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  double bnrm2sqr;
  err = cudaMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* r₀ = b − A·x₀ */
  err = acgsolvercuda_dcopy(b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
  if (err)
    return err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = cudaEventRecord(haloReady, stream);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaStreamWaitEvent(commstream, haloReady, 0);
    if (err)
      return ACG_ERR_CUDA;
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x,
        ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_CUSPARSE)
  err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
    return ACG_ERR_CUSPARSE;
#else
  err = acgsolvercuda_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1, nstartrows_v, d_startrows, numSMs, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_cuda_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
      err = cudaEventRecord(haloRecv, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, haloRecv, 0);
      if (err)
        return ACG_ERR_CUDA;
    }
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
      return ACG_ERR_CUSPARSE;
#else
    err = csrgemv_host(A->nborderrows, d_r + A->borderrowoffset, d_x + A->borderrowoffset, d_orowptr, d_ocolidx, d_oa, -1.0, numSMs, stream);
    if (err)
      return err;
#endif
  }
  cg->ngemv++;

  /* r̂0 = r₀ */
  err = acgsolvercuda_dcopy(n_owned, d_rhat, d_r, stream);
  if (err)
    return err;

  /* w₀ = A·r₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      numSMs, stream, commstream, nnz_full, d_r, d_w, haloReady, haloRecv,
      d_one, d_zero, cusparse, matA, matO, vecr, vecw, vecro, vecwo, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
      nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* t₀ = A·w₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
      d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
      nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* initial dot products: (r₀,r₀) and (r₀,w₀) */
  err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, &d_R2[0]);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasDdot(cublas, n_owned, d_r, 1, d_w, 1, &d_R2[1]);
  if (err)
    return ACG_ERR_CUBLAS;
  err = bicgstab_allreduce(d_R2, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  double rr0, rw0;
  err = cudaMemcpy(&rr0, &d_R2[0], sizeof(rr0), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(&rw0, &d_R2[1], sizeof(rw0), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->r0nrm2 = cg->rnrm2 = sqrt(rr0);
  if (residualrtol > 0)
    residualrtol *= cg->r0nrm2;

  /*
   * Initial scalars: α₀ = (r,r)/(r,w), ρ₀ = (r,r), β = ω = 0, and p = s = z =
   * v = 0 (the β₋₁ = 0 recurrence then yields p₀=r, s₀=w, z₀=t).
   *
   * All of this must be enqueued on the compute stream. The solver's streams
   * are created with ‘cudaStreamNonBlocking’, so they are not ordered against
   * the legacy default stream that the synchronous ‘cudaMemcpy()’ and
   * ‘cudaMemset()’ would use. Those calls are also asynchronous with respect
   * to the host for device memory, and the four vector fills are large -- a
   * few hundred megabytes on a large system -- so they can still be in flight
   * once the solver loop below has started, and land on top of the p, s, z and
   * v that the first iterations have already updated.
   */
  double alpha0 = rr0 / rw0;
  err = cudaMemcpyAsync(
      d_alpha, &alpha0, sizeof(alpha0), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_rho, &rr0, sizeof(rr0), cudaMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_beta, d_zero, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpyAsync(
      d_omega, d_zero, sizeof(double), cudaMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_p, 0, nnz_full * sizeof(*d_p), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_s, 0, nnz_full * sizeof(*d_s), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_z, 0, nnz_full * sizeof(*d_z), stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemsetAsync(d_v, 0, nnz_full * sizeof(*d_v), stream);
  if (err)
    return ACG_ERR_CUDA;
  /* ‘alpha0’ and ‘rr0’ are read by the asynchronous copies above, so they must
   * stay put until those have run */
  err = cudaStreamSynchronize(stream);
  if (err)
    return ACG_ERR_CUDA;

  /* iterative solver loop */
  bool breakdown = false;
  /* fixed-iteration benchmark mode: no stopping criteria are set, so run
   * exactly ‘maxits’ iterations and disable the breakdown early-out */
  bool fixed_iterations =
      (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0);
  /* residual-replacement schedule (mirrors PETSc KSPPIPEBCGS); a period of
   * 0 disables it, and rr_maxit <= 0 removes the upper iteration cap */
  const int rr_period = ACG_BICGSTAB_RR_PERIOD;
  const int rr_maxit = ACG_BICGSTAB_RR_MAXIT;
  /* the stream that makes a reduction readable, and how many scalars the host
   * needs each iteration: the residual norm always, ω and ρ only when the
   * breakdown test is active */
  cudaStream_t readystream = pbicgstab_reduce_ready_stream(
      comm, commsize, nocomm_allreduce, stream, collective_stream);
  int nscalars = fixed_iterations ? 1 : 3;
  for (int k = 0; k < maxits; k++)
  {
    /* steps 1-3: p,s,z recurrences */
    err = acgsolvercuda_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;

    /*
    begin halo exchange
    */
    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
          ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
    }

    /* steps 4-5: q = r − α·s, y = w − α·z */
    err = acgsolvercuda_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* reduction R1 = {(q,y),(y,y)}, overlapped with v = A·z */
    err = cublasDdot(cublas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;

    // spmv diag

#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecz, d_zero,
        vecv, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_v, d_z, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows, d_startrows, numSMs, stream);
    if (err)
      return err;
#endif

    // err = bicgstab_spmv(
    //     A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
    //     numSMs, stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
    //     d_one, d_zero, cusparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
    //     d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
    //     nstartrows_v, d_startrows);
    // if (err)
    //   return err;
    // cg->ngemv++;

    // end halo exchange
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
            ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
        if (err)
          return err;
        err = cudaEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_CUDA;
      }
// spmv off-diagonal
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, veczo, d_one,
          vecvo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(
          A->nborderrows, d_v + A->borderrowoffset, d_z + A->borderrowoffset,
          d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
      if (err)
        return err;
#endif
    }

    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11: ω, then x, r, w updates */
    err = acgsolvercuda_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_qy, d_yy, d_alpha, d_p, d_q, d_y, d_t, d_v, d_x, d_r, d_w, stream);
    if (err)
      return err;

    // begin halo exchange
    if (commsize > 1 && !nocomm_p2p)
    {
      err = cudaEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
          ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
      if (err)
        return err;
    }

    /* reduction R2 = {(r̂0,r),(r̂0,w),(r̂0,s),(r̂0,z),(r,r)}, overlapped with t = A·w */
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(cublas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = pbicgstab_reduce_begin(d_R2, 5, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;

    /* Start bringing the reduced residual norm to the host as soon as the
     * reduction itself completes, rather than after the sparse matrix-vector
     * product that is meant to hide the reduction. */
    if (readystream)
    {
      err = pbicgstab_scalars_copy_begin(
          h_scalars, nscalars, d_rr, d_omega, d_rho, readystream, copystream,
          scalarsReduced, scalarsCopied);
      if (err)
        return err;
    }
// err = bicgstab_spmv(
//     A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
//     numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
//     d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
//     d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
//     nstartrows_v, d_startrows);
// if (err)
//   return err;
// cg->ngemv++;

// spmv diag
#if defined(ACG_USE_CUSPARSE)
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecw, d_zero,
        vect, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#else
    err = acgsolvercuda_csrgemv_merge(
        (A->nprows - A->nghostrows), d_t, d_w, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows, d_startrows, numSMs, stream);
    if (err)
      return err;
#endif

    // end halo exchange
    // spmv off-diagonal
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
            ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
        if (err)
          return err;
        err = cudaEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_CUDA;
      }
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecwo, d_one,
          vecto, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
#else
      err = csrgemv_host(
          A->nborderrows, d_t + A->borderrowoffset, d_w + A->borderrowoffset,
          d_orowptr, d_ocolidx, d_oa, 1.0, numSMs, stream);
      if (err)
        return err;
#endif
    }

    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* the MPI reduction is host-driven, so its result only became readable
     * when ‘pbicgstab_reduce_end()’ completed the request just above */
    if (!readystream)
    {
      err = pbicgstab_scalars_copy_begin(
          h_scalars, nscalars, d_rr, d_omega, d_rho, stream, copystream,
          scalarsReduced, scalarsCopied);
      if (err)
        return err;
    }

    /* the scalar update overwrites ρ in place, so hold it back until the copy
     * above has read the old value */
    err = cudaStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_CUDA;

    /* steps 14-15: β and α for the next iteration. Enqueued before the host
     * blocks on the residual norm below, so that the first dependency of the
     * next iteration is ready as soon as the trailing t = A·w retires. Its
     * result goes unused if one of the tests below leaves the loop. */
    err = acgsolvercuda_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_d1, d_d2, d_d3, d_d4, stream);
    if (err)
      return err;

    /* residual norm for convergence. Only the few bytes staged on
     * ‘copystream’ are waited for, and they were gated on the reduction alone,
     * so the trailing sparse matrix-vector product, the scalar update and the
     * launches of the next iteration all overlap with this. Synchronising on
     * the compute stream instead would drain the whole queue every
     * iteration. */
    err = cudaStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_CUDA;
    cg->rnrm2 = sqrt(h_scalars[0]);
    cg->ntotaliterations++;
    cg->niterations++;
    cg->ndot += 7;

    if ((residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      converged = true;
      break;
    }

    /*
     * Breakdown / stagnation detection. The pipelined BiCGStab recurrences
     * lose accuracy once the residual reaches the attainable precision, and
     * the scalar update ‘acgsolvercuda_pipelined_bicgstab_scalars()’ divides
     * by ω and by ρ = (r̂0,r). If those collapse toward zero (a classic
     * BiCGStab breakdown) or the recurrence residual becomes non-finite, the
     * iterates blow up. Stop here rather than iterating into Inf/NaN — this
     * mirrors PETSc's KSP_DIVERGED_BREAKDOWN / KSP_DIVERGED_NANORINF, so both
     * solvers terminate at the same point instead of diverging.
     *
     * Skipped entirely in fixed-iteration benchmark mode (all tolerances
     * zero): there the caller wants exactly ‘maxits’ iterations for timing,
     * and the per-iteration cost is value-independent, so we let it run to
     * completion without the early break, and ‘nscalars’ leaves ω and ρ on
     * the device.
     */
    if (!fixed_iterations)
    {
      double omega = h_scalars[1], rho_old = h_scalars[2];
      if (!isfinite(cg->rnrm2))
      {
        breakdown = true;
        fprintf(stderr,
                "%s: non-finite residual norm at iteration %d; stopping "
                "(KSP_DIVERGED_NANORINF analogue)\n",
                __func__, cg->niterations);
        break;
      }
      if (omega == 0.0 || !isfinite(omega) || rho_old == 0.0 || !isfinite(rho_old))
      {
        breakdown = true;
        fprintf(stderr,
                "%s: BiCGStab breakdown at iteration %d "
                "(omega=%.*g, rho=%.*g, residual norm=%.*g); stopping "
                "(KSP_DIVERGED_BREAKDOWN analogue)\n",
                __func__, cg->niterations, DBL_DIG, omega, DBL_DIG, rho_old,
                DBL_DIG, cg->rnrm2);
        break;
      }
    }

    /*
     * Periodic residual replacement (mirrors PETSc KSPPIPEBCGS). Every
     * ‘rr_period’ iterations, recompute the recurrence-propagated vectors
     * from the primary vectors x and p with explicit SpMVs, which resets the
     * accumulated residual gap and keeps the pipelined recurrences accurate:
     *   r = b − A·x,  w = A·r,  t = A·w,  s = A·p,  z = A·s,  v = A·z.
     * (unpreconditioned, so PETSc's PCApply steps collapse to the identity)
     */
    if (rr_period > 0 && k > 0 && (k % rr_period) == 0 &&
        (rr_maxit <= 0 || k < rr_maxit))
    {
      /* r = b − A·x */
      err = acgsolvercuda_dcopy(n_owned, d_r, d_b, stream);
      if (err)
        return err;
      if (commsize > 1 && !nocomm_p2p)
      {
        err = cudaEventRecord(haloReady, stream);
        if (err)
          return ACG_ERR_CUDA;
        err = cudaStreamWaitEvent(commstream, haloReady, 0);
        if (err)
          return ACG_ERR_CUDA;
        err = acghalo_exchange_cuda_begin(
            cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full,
            d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
        if (err)
          return err;
      }
#if defined(ACG_USE_CUSPARSE)
      err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
      if (err)
        return ACG_ERR_CUSPARSE;
#else
      err = acgsolvercuda_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1, nstartrows_v, d_startrows, numSMs, stream);
      if (err)
        return err;
#endif
      if (commsize > 1)
      {
        if (!nocomm_p2p)
        {
          err = acghalo_exchange_cuda_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs, commstream);
          if (err)
            return err;
          err = cudaEventRecord(haloRecv, commstream);
          if (err)
            return ACG_ERR_CUDA;
          err = cudaStreamWaitEvent(stream, haloRecv, 0);
          if (err)
            return ACG_ERR_CUDA;
        }
#if defined(ACG_USE_CUSPARSE)
        err = cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
          return ACG_ERR_CUSPARSE;
#else
        err = csrgemv_host(A->nborderrows, d_r + A->borderrowoffset, d_x + A->borderrowoffset, d_orowptr, d_ocolidx, d_oa, -1.0, numSMs, stream);
        if (err)
          return err;
#endif
      }
      cg->ngemv++;

      /* w = A·r */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_r, d_w, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecr, vecw, vecro, vecwo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* t = A·w */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* s = A·p */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_p, d_s, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecp, vecs, vecpo, vecso, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* z = A·s */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_s, d_z, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecs, vecz, vecso, veczo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* v = A·z */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          numSMs, stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
          d_one, d_zero, cusparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, d_orowptr, d_ocolidx, d_oa,
          nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;
    }
  }
  /* The loop no longer drains the compute stream every iteration, so the
   * trailing work of the last iteration may still be in flight. Drain once
   * here, before stopping the clock, so that the reported time covers all of
   * the work -- and so that the solution copied out below is complete. The
   * streams are non-blocking, so the copy on the default stream would not
   * synchronise with them. */
  cudaStreamSynchronize(stream);
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

  /* copy solution back to host */
  err = cudaMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

#if defined(ACG_USE_CUSPARSE)
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecw);
  cusparseDestroyDnVec(vect);
  cusparseDestroyDnVec(vecz);
  cusparseDestroyDnVec(vecv);
  cusparseDestroyDnVec(vecp);
  cusparseDestroyDnVec(vecs);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecwo);
    cusparseDestroyDnVec(vecto);
    cusparseDestroyDnVec(veczo);
    cusparseDestroyDnVec(vecvo);
    cusparseDestroyDnVec(vecpo);
    cusparseDestroyDnVec(vecso);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
#else
  cudaFree(d_startrows);
#endif

  cudaFree(d_x);
  cudaFree(d_b);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  cudaFree(d_omega);
  cudaFree(d_rho);
  cudaFree(d_R1);
  cudaFree(d_R2);
  cudaEventDestroy(dotEvent);
  cudaEventDestroy(redEvent);
  cudaEventDestroy(haloReady);
  cudaEventDestroy(haloRecv);
  cudaStreamDestroy(stream);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(collective_stream);
  cudaEventDestroy(scalarsReduced);
  cudaEventDestroy(scalarsCopied);
  cudaStreamDestroy(copystream);
  cudaFreeHost(h_scalars);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  if (converged)
    return ACG_SUCCESS;
  /* a breakdown is an early stop, so report it as not converged even in
   * fixed-iteration benchmark mode (mirrors the PETSc wrapper's handling of
   * a negative KSPConvergedReason) */
  if (breakdown)
    return ACG_ERR_NOT_CONVERGED;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;
  return ACG_ERR_NOT_CONVERGED;
}


/**
 * ‘acgsolvercuda_solve_pipelined()’ solves the given linear system,
 * Ax=b, using a pipelined conjugate gradient method. The linear
 * system may be distributed across multiple processes and
 * communication is handled using MPI.
 *
 * The solver must already have been configured with ‘acgsolvercuda_init()’
 * for a linear system Ax=b, and the dimensions of the vectors b and x
 * must match the number of columns and rows of A, respectively.
 *
 * The stopping criterion are:
 *
 *  - ‘maxits’, the maximum number of iterations to perform
 *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ < γₐ
 *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖ <
 * γᵣ
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
int acgsolvercuda_solve_pipelined(
    struct acgsolvercuda *cg,
    const struct acgsymcsrmatrix *A,
    const struct acgvector *b,
    struct acgvector *x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    struct acgcomm *comm,
    int tag,
    int *errcode,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse)
{
  int err;
  if (b->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (x->size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->r.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->p.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;
  if (cg->t.size < A->nrows)
    return ACG_ERR_INDEX_OUT_OF_BOUNDS;

  /* not implemented */
  if (diffatol > 0 || diffrtol > 0)
    return ACG_ERR_NOT_SUPPORTED;

  int commsize, rank;
  acgcomm_size(comm, &commsize);
  acgcomm_rank(comm, &rank);

  /* allocate extra vectors needed for pipelined CG */
  if (!cg->w)
  {
    cg->w = malloc(sizeof(*cg->w));
    if (!cg->w)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->w, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->q)
  {
    cg->q = malloc(sizeof(*cg->q));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->q, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
    if (err)
      return ACG_ERR_CUDA;
  }
  if (!cg->z)
  {
    cg->z = malloc(sizeof(*cg->z));
    if (!cg->z)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->z, x);
    if (err)
      return err;
    err = cudaMalloc(
        (void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
    if (err)
      return ACG_ERR_CUDA;
  }

  /* /\* If the stopping criterion is based on the difference in */
  /*  * solution from one iteration to the next, then allocate */
  /*  * additional storage for storing the difference. *\/ */
  /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
  /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO;
   */
  /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
  /* } */

  cudaStream_t stream = 0;
  const struct acghalo *halo = cg->halo;
  double *d_bnrm2sqr = cg->d_bnrm2sqr;
  double *d_rnrm2sqr = &cg->d_rnrm2sqr[0];
  double *d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
  double *d_delta = &cg->d_rnrm2sqr[1];
  double *d_alpha = cg->d_alpha;
  double *d_minus_alpha = cg->d_minus_alpha;
  double *d_beta = cg->d_beta;
  double *d_one = cg->d_one;
  double *d_minus_one = cg->d_minus_one;
  double *d_zero = cg->d_zero;
  double *d_inf = cg->d_inf;
  double *d_r = cg->d_r;
  double *d_p = cg->d_p;
  double *d_t = cg->d_t;
  double *d_w = cg->d_w;
  double *d_q = cg->d_q;
  double *d_z = cg->d_z;
  acgidx_t *d_rowptr = cg->d_rowptr;
  acgidx_t *d_colidx = cg->d_colidx;
  double *d_a = cg->d_a;
  acgidx_t *d_orowptr = cg->d_orowptr;
  acgidx_t *d_ocolidx = cg->d_ocolidx;
  double *d_oa = cg->d_oa;

  /* get cuda device properties */
  int numSMs;
  err = getNumberOfSMs(&numSMs);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUDA;
  }

  /* configure cublas and cusparse to use device-side pointers */
  cublasPointerMode_t cublaspointermode;
  err = cublasGetPointerMode(cublas, &cublaspointermode);
  if (err)
    return ACG_ERR_CUBLAS;
  err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_CUBLAS;
  cusparsePointerMode_t cusparsepointermode;
  err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }

  double *rnrm2sqr;
  err = cudaMallocHost((void **)&rnrm2sqr, sizeof(*rnrm2sqr));
  if (err)
    return ACG_ERR_CUDA;
  cudaStream_t copystream;
  err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t rnrm2sqrready;
  cudaEventCreateWithFlags(&rnrm2sqrready, cudaEventDisableTiming);

  /* copy right-hand side and initial guess to device */
  double *d_b;
  err = cudaMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;
  double *d_x;
  err = cudaMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyHostToDevice);
  if (err)
    return ACG_ERR_CUDA;

  /* used to overlap P2P communication with SpMV */
  cudaStream_t commstream;
  err = cudaStreamCreateWithFlags(&commstream, cudaStreamNonBlocking);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t xreadytosend, xreceived;
  err = cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(xreadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t rreadytosend, rreceived;
  err = cudaEventCreateWithFlags(&rreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(rreadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&rreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  cudaEvent_t wreadytosend, wreceived;
  err = cudaEventCreateWithFlags(&wreadytosend, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventRecord(wreadytosend, stream);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaEventCreateWithFlags(&wreceived, cudaEventDisableTiming);
  if (err)
    return ACG_ERR_CUDA;

  /* create cusparse matrix and vectors */
  cusparseDnVecDescr_t vecx, vecr, vecw, vecq;
  err = cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecw, A->nownedrows, d_w, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cusparseCreateDnVec(&vecq, A->nownedrows, d_q, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  cusparseDnVecDescr_t vecxo, vecro, vecwo, vecqo;
  if (commsize > 1)
  {
    err = cusparseCreateDnVec(
        &vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cusparseCreateDnVec(
        &vecqo, A->nborderrows + A->nghostrows, d_q + A->borderrowoffset,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }

  cusparseSpMatDescr_t matA;
  err = cusparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  size_t buffersize;
  err = cusparseSpMV_bufferSize(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  void *d_buffer;
  err = cudaMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_CUDA;
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
  err = cusparseSpMV_preprocess(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
#endif

  cusparseSpMatDescr_t matO;
  void *d_obuffer;
  if (commsize > 1)
  {
    err = cusparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows,
        A->nborderrows + A->nghostrows, A->onpnzs, d_orowptr, d_ocolidx,
        d_oa, CUSPARSE_IDX_T, CUSPARSE_IDX_T, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    size_t obuffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_CUDA;
#if ( \
    CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
#endif
  }

  /* create timing events for profiling */
  acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0,
           nallreduce = 0, nhalo = 0;
  cudaEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
  tgemv = malloc(2 * (maxits + 2) * sizeof(*tgemv));
  if (!tgemv)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&tgemv[i]);
  tdot = malloc(2 * maxits * sizeof(*tdot));
  if (!tdot)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * maxits; i++)
    cudaEventCreate(&tdot[i]);
  tnrm2 = malloc(2 * (maxits + 1) * sizeof(*tnrm2));
  if (!tnrm2)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tnrm2[i]);
  taxpy = malloc(2 * maxits * sizeof(*taxpy));
  if (!taxpy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * maxits; i++)
    cudaEventCreate(&taxpy[i]);
  tcopy = malloc(2 * 1 * sizeof(*tcopy));
  if (!tcopy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * 1; i++)
    cudaEventCreate(&tcopy[i]);
  tallreduce = malloc(2 * (maxits + 1) * sizeof(*tallreduce));
  if (!tallreduce)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    cudaEventCreate(&tallreduce[i]);
  thalo = malloc(2 * (maxits + 2) * sizeof(*thalo));
  if (!thalo)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    cudaEventCreate(&thalo[i]);
#endif

  /* warmup iterations for dot/allreduce */
  for (int i = 0; i < warmup; i++)
  {
    cudaMemcpy(
        d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(
        d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_delta, d_zero, sizeof(*d_delta), cudaMemcpyDeviceToDevice);
    err = cublasDdot(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
        d_bnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
      return ACG_ERR_CUBLAS;
    err = cublasDdot(
        cublas, cg->w->num_nonzeros - cg->w->num_ghost_nonzeros, d_w, 1,
        d_r, 1, d_delta);
    if (err)
      return ACG_ERR_CUBLAS;
    if (commsize > 1)
      acgcomm_allreduce(
          ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    cudaStreamSynchronize(stream);
  }
  cudaMemcpy(
      d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
  cudaMemcpy(
      d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
  cudaMemcpy(d_delta, d_zero, sizeof(*d_delta), cudaMemcpyDeviceToDevice);

  /* warmup iterations for halo exchange/SpMV */
  for (int i = 0; i < warmup; i++)
  {
    /* r = b-Ax */
    err = cublasDcopy(
        cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUBLAS;
    }
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, xreadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
        d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
      err = cudaEventRecord(xreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, xreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
          vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
          d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
      err = cudaEventRecord(xreadytosend, stream);
      if (err)
        return ACG_ERR_CUDA;
    }

    /* w = Ar */
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, rreadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->r.num_nonzeros, d_r, ACG_DOUBLE,
          cg->r.num_nonzeros, d_r, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecr,
        d_zero, vecw, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->r.num_nonzeros, d_r, ACG_DOUBLE,
          cg->r.num_nonzeros, d_r, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
          commstream);
      if (err)
        return err;
      err = cudaEventRecord(rreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, rreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecro,
          d_one, vecwo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
    }
    err = cudaEventRecord(wreadytosend, stream);
    if (err)
      return ACG_ERR_CUDA;

    /* q = Aw */
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, wreadytosend, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->w->num_nonzeros, d_w,
          ACG_DOUBLE, cg->w->num_nonzeros, d_w, ACG_DOUBLE, comm, tag,
          errcode, 0, numSMs, commstream);
      if (err)
        return err;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecw,
        d_zero, vecq, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->w->num_nonzeros, d_w,
          ACG_DOUBLE, cg->w->num_nonzeros, d_w, ACG_DOUBLE, comm, tag,
          errcode, 0, numSMs, commstream);
      if (err)
        return err;
      err = cudaEventRecord(wreceived, commstream);
      if (err)
        return ACG_ERR_CUDA;
      err = cudaStreamWaitEvent(stream, wreceived, 0);
      if (err)
        return ACG_ERR_CUDA;
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecwo,
          d_one, vecqo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
    }
  }

  /* warmup iterations for axpy */
  err =
      cudaMemcpy(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev),
      cudaMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_z, 0,
      (cg->z->num_nonzeros - cg->z->num_ghost_nonzeros) * sizeof(*d_z));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_t, 0, (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_p, 0, (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*d_p));
  if (err)
    return ACG_ERR_CUDA;
  for (int i = 0; i < warmup; i++)
  {
    err = cudaMemcpy(
        d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMemcpy(
        d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev),
        cudaMemcpyDeviceToDevice);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMemcpy(
        d_delta, d_inf, sizeof(*d_delta), cudaMemcpyDeviceToDevice);
    if (err)
      return ACG_ERR_CUDA;
    err = cudaMemcpy(
        d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice);
    if (err)
      return ACG_ERR_CUDA;
    err = acgsolvercuda_pipelined_daxpy_fused(
        cg->t.num_nonzeros - cg->t.num_ghost_nonzeros, d_rnrm2sqr,
        d_rnrm2sqr_prev, d_delta, d_q, d_p, d_r, d_t, d_x, d_z, d_w,
        d_alpha, stream);
    if (err)
      return err;
  }
  err = cudaMemset(
      d_r, 0, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_w, 0,
      (cg->w->num_nonzeros - cg->w->num_ghost_nonzeros) * sizeof(*d_w));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_q, 0,
      (cg->q->num_nonzeros - cg->q->num_ghost_nonzeros) * sizeof(*d_q));
  if (err)
    return ACG_ERR_CUDA;

  /* set scalars to infinity (needed to produce correct results on
   * the first call to acgsolvercuda_pipelined_daxpy_fused) */
  err =
      cudaMemcpy(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemcpy(
      d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev),
      cudaMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_CUDA;

  /* set the vectors z, t and and p to zero */
  err = cudaMemset(
      d_z, 0,
      (cg->z->num_nonzeros - cg->z->num_ghost_nonzeros) * sizeof(*d_z));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_t, 0, (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t));
  if (err)
    return ACG_ERR_CUDA;
  err = cudaMemset(
      d_p, 0, (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*d_p));
  if (err)
    return ACG_ERR_CUDA;

  /* set initial state */
  bool converged = false;
  cg->nsolves++;
  cg->niterations = 0;
  cg->bnrm2 = INFINITY;
  cg->r0nrm2 = cg->rnrm2 = INFINITY;
  cg->x0nrm2 = cg->dxnrm2 = INFINITY;
  cg->maxits = maxits;
  cg->diffatol = diffatol;
  cg->diffrtol = diffrtol;
  cg->residualatol = residualatol;
  cg->residualrtol = residualrtol;
  acgtime_t t0, t1;
  err = acgcomm_barrier(stream, comm, errcode);
  if (err)
    return err;
  cudaStreamSynchronize(stream);
  gettime(&t0);

  /* compute right-hand side norm */
  double bnrm2sqr;
  acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
  err = cublasDdot(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
      d_bnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
  nnrm2++;
  cg->nnrm2++;
  cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
  cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
  if (commsize > 1)
  {
    acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    err = acgcomm_allreduce(
        ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
        errcode);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    nallreduce++;
    cg->nallreduce++;
    cg->Ballreduce += sizeof(bnrm2sqr);
  }
  err = cudaMemcpy(
      &bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* /\* compute norm of initial guess *\/ */
  /* if (diffatol > 0 || diffrtol > 0) { */
  /*     gettime(&tnrm20); */
  /*     double x0nrm2sqr; */
  /*     err = acgvector_dnrm2sqr(x, &x0nrm2sqr, &cg->nflops, &cg->Bnrm2); */
  /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
   */
  /*     gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
   */
  /*     gettime(&tallreduce0); */
  /*     err = MPI_Allreduce(MPI_IN_PLACE, &x0nrm2sqr, 1, MPI_DOUBLE, MPI_SUM,
   * comm->mpicomm); */
  /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *errcode =
   * err; return ACG_ERR_MPI; } */
  /*     cg->Ballreduce += sizeof(x0nrm2sqr); */
  /*     gettime(&tallreduce1); cg->nallreduce++; cg->tallreduce +=
   * elapsed(tallreduce0,tallreduce1); */
  /*     cg->x0nrm2 = sqrt(x0nrm2sqr); */
  /*     diffrtol *= cg->x0nrm2; */
  /* } */

  /* compute initial residual, r₀ = b-A*x₀ */
  acgEventRecord(tcopy[2 * ncopy + 0], 0);
  err = cublasDcopy(
      cublas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUBLAS;
  }
  acgEventRecord(tcopy[2 * ncopy + 1], 0);
  ncopy++;
  cg->ncopy++;
  cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

  if (commsize > 1)
  {
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  if (commsize > 1)
  {
    acgEventRecord(thalo[2 * nhalo + 0], 0);
    err = acghalo_exchange_cuda_end(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 1], 0);
    nhalo++;
    cg->nhalo++;
    cg->Bhalo += cg->halo->sendsize * sizeof(*x->x);
    cg->nhalomsgs += cg->halo->nrecipients;
    err = cudaEventRecord(xreceived, commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(stream, xreceived, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    err = cudaEventRecord(rreadytosend, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 1], 0);
  ngemv++;
  cg->ngemv++;
  cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
  cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

  /* compute w = Ar */
  if (commsize > 1)
  {
    err = cudaStreamWaitEvent(commstream, rreadytosend, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = acghalo_exchange_cuda_begin(
        cg->halo, cg->haloexchange, cg->r.num_nonzeros, d_r, ACG_DOUBLE,
        cg->r.num_nonzeros, d_r, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
  err = cusparseSpMV(
      cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecr, d_zero,
      vecw, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  if (commsize > 1)
  {
    acgEventRecord(thalo[2 * nhalo + 0], 0);
    err = acghalo_exchange_cuda_end(
        cg->halo, cg->haloexchange, cg->r.num_nonzeros, d_r, ACG_DOUBLE,
        cg->r.num_nonzeros, d_r, ACG_DOUBLE, comm, tag, errcode, 0, numSMs,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(thalo[2 * nhalo + 1], 0);
    nhalo++;
    cg->nhalo++;
    cg->Bhalo += cg->halo->sendsize * sizeof(*cg->r.x);
    cg->nhalomsgs += cg->halo->nrecipients;
    err = cudaEventRecord(rreceived, commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(stream, rreceived, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecro,
        d_one, vecwo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 1], 0);
  ngemv++;
  cg->ngemv++;
  cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
  cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->w->x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->r.num_nonzeros * sizeof(*cg->r.x);
  err = cudaEventRecord(wreadytosend, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_CUDA;
  }

  /* iterative solver loop */
  for (int k = 0; k < maxits; k++)
  {

    /* compute residual norm (r,r) */
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = cublasDdot(
        cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
    cg->Bnrm2 +=
        (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);

    /* compute (w,r) */
    acgEventRecord(tdot[2 * ndot + 0], 0);
    err = cublasDdot(
        cublas, cg->w->num_nonzeros - cg->w->num_ghost_nonzeros, d_w, 1,
        d_r, 1, d_delta);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUBLAS;
    }
    acgEventRecord(tdot[2 * ndot + 1], 0);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->w->num_nonzeros - cg->w->num_ghost_nonzeros);
    cg->Bdot += (cg->w->num_nonzeros - cg->w->num_ghost_nonzeros) * (sizeof(*cg->w->x) + sizeof(*cg->r.x));

    /* perform a single reduction for the two dot products */
    if (commsize > 1)
    {
      acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
      err = acgcomm_allreduce(
          ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm,
          errcode);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
      nallreduce++;
      cg->nallreduce++;
      cg->Ballreduce += 2 * sizeof(*d_rnrm2sqr);
    }

    /* start copying residual norm from device to host,
     * overlapping it with the matrix-vector product */
    err = cudaEventRecord(rnrm2sqrready, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    err = cudaMemcpyAsync(
        rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost,
        copystream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }

    /* compute q = Aw */
    if (commsize > 1)
    {
      err = cudaStreamWaitEvent(commstream, wreadytosend, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = acghalo_exchange_cuda_begin(
          cg->halo, cg->haloexchange, cg->w->num_nonzeros, d_w,
          ACG_DOUBLE, cg->w->num_nonzeros, d_w, ACG_DOUBLE, comm, tag,
          errcode, 0, numSMs, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecw,
        d_zero, vecq, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_CUSPARSE;
    }
    if (commsize > 1)
    {
      acgEventRecord(thalo[2 * nhalo + 0], 0);
      err = acghalo_exchange_cuda_end(
          cg->halo, cg->haloexchange, cg->w->num_nonzeros, d_w,
          ACG_DOUBLE, cg->w->num_nonzeros, d_w, ACG_DOUBLE, comm, tag,
          errcode, 0, numSMs, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
      acgEventRecord(thalo[2 * nhalo + 1], 0);
      nhalo++;
      cg->nhalo++;
      cg->Bhalo += cg->halo->sendsize * sizeof(*cg->w->x);
      cg->nhalomsgs += cg->halo->nrecipients;
      err = cudaEventRecord(wreceived, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cudaStreamWaitEvent(stream, wreceived, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_CUDA;
      }
      err = cusparseSpMV(
          cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecwo,
          d_one, vecqo, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
          *errcode = err;
        return ACG_ERR_CUSPARSE;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->q->x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->w->num_nonzeros * sizeof(*cg->w->x);

    /* wait for host to receive updated residual norm */
    cudaStreamSynchronize(copystream);
    cg->rnrm2 = sqrt(*rnrm2sqr);
    if (k == 0)
    {
      cg->r0nrm2 = cg->rnrm2;
      residualrtol *= cg->r0nrm2;
    }

    /* convergence tests */
    if ((diffatol > 0 && cg->dxnrm2 < diffatol) || (diffrtol > 0 && cg->dxnrm2 < diffrtol) || (residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      cudaStreamSynchronize(stream);
      converged = true;
      break;
    }

    /* update vectors */
    acgEventRecord(taxpy[2 * naxpy + 0], 0);
    err = acgsolvercuda_pipelined_daxpy_fused(
        cg->t.num_nonzeros - cg->t.num_ghost_nonzeros, d_rnrm2sqr,
        d_rnrm2sqr_prev, d_delta, d_q, d_p, d_r, d_t, d_x, d_z, d_w,
        d_alpha, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
    acgEventRecord(taxpy[2 * naxpy + 1], 0);
    naxpy++;
    cg->naxpy++;
    cg->nflops += 12 * (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros);
    cg->Baxpy += 7 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*cg->p.x);
    err = cudaEventRecord(wreadytosend, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_CUDA;
    }
    cg->ntotaliterations++;
    cg->niterations++;
  }
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

#if defined(ACG_ENABLE_PROFILING)
  /* record profiling information */
  float t;
  for (acgidx_t i = 0; i < ngemv; i++)
  {
    cudaEventSynchronize(tgemv[2 * i + 1]);
    cudaEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
    cg->tgemv += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ndot; i++)
  {
    cudaEventSynchronize(tdot[2 * i + 1]);
    cudaEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
    cg->tdot += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nnrm2; i++)
  {
    cudaEventSynchronize(tnrm2[2 * i + 1]);
    cudaEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
    cg->tnrm2 += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < naxpy; i++)
  {
    cudaEventSynchronize(taxpy[2 * i + 1]);
    cudaEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
    cg->taxpy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ncopy; i++)
  {
    cudaEventSynchronize(tcopy[2 * i + 1]);
    cudaEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
    cg->tcopy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nallreduce; i++)
  {
    cudaEventSynchronize(tallreduce[2 * i + 1]);
    cudaEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
    cg->tallreduce += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nhalo; i++)
  {
    cudaEventSynchronize(thalo[2 * i + 1]);
    cudaEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
    cg->thalo += 1.0e-3 * t;
  }
#endif

  /* copy solution back to host */
  err = cudaMemcpy(
      x->x, d_x, x->num_nonzeros * sizeof(*d_x), cudaMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_CUDA;

  /* free cusparse matrix and vectors */
  cusparseDestroyDnVec(vecx);
  cusparseDestroyDnVec(vecr);
  cusparseDestroyDnVec(vecw);
  cusparseDestroyDnVec(vecq);
  if (commsize > 1)
  {
    cusparseDestroyDnVec(vecxo);
    cusparseDestroyDnVec(vecro);
    cusparseDestroyDnVec(vecwo);
    cusparseDestroyDnVec(vecqo);
  }
  cusparseDestroySpMat(matA);
  cudaFree(d_buffer);
  if (commsize > 1)
  {
    cusparseDestroySpMat(matO);
    cudaFree(d_obuffer);
  }
  cudaFree(d_x);
  cudaFree(d_b);
  cudaFreeHost(rnrm2sqr);
  cudaStreamDestroy(commstream);
  cudaStreamDestroy(copystream);

  /* reset cusparse and cublas pointer modes */
  err = cusparseSetPointerMode(cusparse, cusparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUSPARSE;
  }
  err = cublasSetPointerMode(cublas, cublaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_CUBLAS;
  }

  /* check for CUDA errors */
  if (cudaGetLastError() != cudaSuccess)
    return ACG_ERR_CUDA;

  /* if the solver converged or the only stopping criteria is a
   * maximum number of iterations, then the solver succeeded */
  if (converged)
    return ACG_SUCCESS;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;

  /* otherwise, the solver failed to converge with the given number
   * of maximum iterations */
  return ACG_ERR_NOT_CONVERGED;
}
// #endif

/*
 * output solver info
 */

static void findent(FILE *f, int indent)
{
  fprintf(f, "%*c", indent, ' ');
}

/**
 * ‘acgsolvercuda_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolvercuda_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolvercuda_fwrite(FILE *f, const struct acgsolvercuda *cg, int indent)
{
  double tother = cg->tsolve - (cg->tgemv + cg->tdot + cg->tnrm2 + cg->taxpy + cg->tcopy + cg->tallreduce + cg->thalo);
  findent(f, indent);
  fprintf(f, "unknowns: %'" PRIdx "\n", cg->p.size);
  findent(f, indent);
  fprintf(f, "solves: %'d\n", cg->nsolves);
  findent(f, indent);
  fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
  findent(f, indent);
  fprintf(f, "total flops: %'.3f Gflop\n", 1.0e-9 * cg->nflops);
  findent(f, indent);
  fprintf(
      f, "total flop rate: %'.3f Gflop/s\n",
      cg->tsolve > 0 ? 1.0e-9 * cg->nflops / cg->tsolve : 0);
  findent(f, indent);
  fprintf(f, "total solver time: %'.6f seconds\n", cg->tsolve);
  findent(f, indent);
  fprintf(f, "performance breakdown:\n");
  findent(f, indent);
  fprintf(
      f,
      "  gemv: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
      cg->tgemv, cg->ngemv, cg->Bgemv,
      cg->tgemv > 0 ? 1.0e-9 * cg->Bgemv / cg->tgemv : 0.0);
  findent(f, indent);
  fprintf(
      f,
      "  dot: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
      cg->tdot, cg->ndot, cg->Bdot,
      cg->tdot > 0 ? 1.0e-9 * cg->Bdot / cg->tdot : 0.0);
  findent(f, indent);
  fprintf(
      f,
      "  nrm2: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
      cg->tnrm2, cg->nnrm2, cg->Bnrm2,
      cg->tnrm2 > 0 ? 1.0e-9 * cg->Bnrm2 / cg->tnrm2 : 0.0);
  findent(f, indent);
  fprintf(
      f,
      "  axpy: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
      cg->taxpy, cg->naxpy, cg->Baxpy,
      cg->taxpy > 0 ? 1.0e-9 * cg->Baxpy / cg->taxpy : 0.0);
  findent(f, indent);
  fprintf(
      f,
      "  copy: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
      cg->tcopy, cg->ncopy, cg->Bcopy,
      cg->tcopy > 0 ? 1.0e-9 * cg->Bcopy / cg->tcopy : 0.0);
  findent(f, indent);
  fprintf(
      f,
      "  MPI_Allreduce: %'.6f seconds %'" PRId64 " times %'" PRId64
      " B %'.3f GB/s\n",
      cg->tallreduce, cg->nallreduce, cg->Ballreduce,
      cg->tallreduce > 0.0 ? 1.0e-9 * cg->Ballreduce / cg->tallreduce : 0.0);
  findent(f, indent);
  fprintf(
      f,
      "  MPI_HaloExchange: %'.6f seconds %'" PRId64 " times %'" PRId64
      " B %'.3f GB/s\n",
      cg->thalo, cg->nhalo, cg->Bhalo,
      cg->thalo > 0.0 ? 1.0e-9 * cg->Bhalo / cg->thalo : 0.0);
  findent(f, indent);
  fprintf(f, "  other: %'.6f seconds\n", tother);
  findent(f, indent);
  fprintf(f, "last solve:\n");
  findent(f, indent);
  fprintf(f, "  stopping criterion:\n");
  findent(f, indent);
  fprintf(f, "    maximum iterations: %'d\n", cg->maxits);
  findent(f, indent);
  fprintf(f, "    tolerance for residual: %.*g\n", DBL_DIG, cg->residualatol);
  findent(f, indent);
  fprintf(
      f, "    tolerance for relative residual: %.*g\n", DBL_DIG,
      cg->residualrtol);
  findent(f, indent);
  fprintf(
      f, "    tolerance for difference in solution iterates: %.*g\n", DBL_DIG,
      cg->diffatol);
  findent(f, indent);
  fprintf(
      f, "    tolerance for relative difference in solution iterates: %.*g\n",
      DBL_DIG, cg->diffrtol);
  findent(f, indent);
  fprintf(f, "  iterations: %'d\n", cg->niterations);
  findent(f, indent);
  fprintf(f, "  right-hand side 2-norm: %.*g\n", DBL_DIG, cg->bnrm2);
  findent(f, indent);
  fprintf(f, "  initial guess 2-norm: %.*g\n", DBL_DIG, cg->x0nrm2);
  findent(f, indent);
  fprintf(f, "  initial residual 2-norm: %.*g\n", DBL_DIG, cg->r0nrm2);
  findent(f, indent);
  fprintf(f, "  residual 2-norm: %.*g\n", DBL_DIG, cg->rnrm2);
  findent(f, indent);
  fprintf(
      f, "  difference in solution iterates 2-norm: %.*g\n", DBL_DIG,
      cg->dxnrm2);
  findent(f, indent);
  fprintf(
      f, "  floating-point exceptions: %s\n",
      acgerrcodestr(ACG_ERR_FEXCEPT, 0));
  return ACG_SUCCESS;
}

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolvercuda_fwritempi()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolvercuda_solvempi()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolvercuda_fwritempi(
    FILE *f,
    const struct acgsolvercuda *cg,
    int indent,
    int verbose,
    MPI_Comm comm,
    int root)
{
  int commsize, rank;
  MPI_Comm_size(comm, &commsize);
  MPI_Comm_rank(comm, &rank);
  int64_t nflops = cg->nflops;
  double tsolve = cg->tsolve;
  double tgemv = cg->tgemv;
  double tgemv_o = cg->tgemv_o;
  double tdot = cg->tdot;
  double tnrm2 = cg->tnrm2;
  double taxpy = cg->taxpy;
  double tcopy = cg->tcopy;
  double tallreduce = cg->tallreduce;
  double thalo = cg->thalo;
  double tprecond = cg->tprecond;
  double tother =
      tsolve - (tgemv + tgemv_o + tdot + tnrm2 + taxpy + tcopy + tallreduce + thalo + tprecond);
  int64_t ngemv = cg->ngemv, Bgemv = cg->Bgemv;
  int64_t ngemv_o = cg->ngemv_o;
  int64_t ndot = cg->ndot, Bdot = cg->Bdot;
  int64_t nnrm2 = cg->nnrm2, Bnrm2 = cg->Bnrm2;
  int64_t naxpy = cg->naxpy, Baxpy = cg->Baxpy;
  int64_t ncopy = cg->ncopy, Bcopy = cg->Bcopy;
  int64_t nallreduce = cg->nallreduce, Ballreduce = cg->Ballreduce;
  int64_t nprecond = cg->nprecond;
  int64_t nhalo = cg->nhalo, Bhalo = cg->Bhalo;
  int64_t nhalopack = cg->halo->npack, Bhalopack = cg->halo->Bpack;
  int64_t nhalounpack = cg->halo->nunpack, Bhalounpack = cg->halo->Bunpack;
  int64_t nhalompiirecv = cg->halo->nmpiirecv,
          Bhalompiirecv = cg->halo->Bmpiirecv;
  int64_t nhalompisend = cg->halo->nmpisend,
          Bhalompisend = cg->halo->Bmpisend;
  int64_t nhalomsgs = cg->nhalomsgs;
  MPI_Reduce(&cg->nflops, &nflops, 1, MPI_INT64_T, MPI_SUM, root, comm);
  MPI_Reduce(&cg->tsolve, &tsolve, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  MPI_Reduce(&cg->tgemv, &tgemv, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tgemv /= commsize;
  MPI_Reduce(&cg->tgemv_o, &tgemv_o, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tgemv_o /= commsize;
  MPI_Reduce(&cg->tdot, &tdot, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tdot /= commsize;
  MPI_Reduce(&cg->tnrm2, &tnrm2, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tnrm2 /= commsize;
  MPI_Reduce(&cg->taxpy, &taxpy, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  taxpy /= commsize;
  MPI_Reduce(&cg->tcopy, &tcopy, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tcopy /= commsize;
  MPI_Reduce(
      &cg->tallreduce, &tallreduce, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tallreduce /= commsize;
  MPI_Reduce(
      &cg->tprecond, &tprecond, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  tprecond /= commsize;
  MPI_Reduce(&cg->thalo, &thalo, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  thalo /= commsize;
  MPI_Reduce(
      rank == root ? MPI_IN_PLACE : &tother, &tother, 1, MPI_DOUBLE, MPI_SUM,
      root, comm);
  tother /= commsize;
  MPI_Reduce(&cg->ngemv, &ngemv, 1, MPI_INT64_T, MPI_SUM, root, comm);
  ngemv /= commsize;
  MPI_Reduce(&cg->ngemv_o, &ngemv_o, 1, MPI_INT64_T, MPI_SUM, root, comm);
  ngemv_o /= commsize;
  MPI_Reduce(&cg->ndot, &ndot, 1, MPI_INT64_T, MPI_SUM, root, comm);
  ndot /= commsize;
  MPI_Reduce(&cg->nnrm2, &nnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm);
  nnrm2 /= commsize;
  MPI_Reduce(&cg->naxpy, &naxpy, 1, MPI_INT64_T, MPI_SUM, root, comm);
  naxpy /= commsize;
  MPI_Reduce(&cg->ncopy, &ncopy, 1, MPI_INT64_T, MPI_SUM, root, comm);
  ncopy /= commsize;
  MPI_Reduce(
      &cg->nallreduce, &nallreduce, 1, MPI_INT64_T, MPI_SUM, root, comm);
  nallreduce /= commsize;
  MPI_Reduce(
      &cg->nprecond, &nprecond, 1, MPI_INT64_T, MPI_SUM, root, comm);
  nprecond /= commsize;
  MPI_Reduce(
      &cg->halo->npack, &nhalopack, 1, MPI_INT64_T, MPI_SUM, root, comm);
  nhalopack /= commsize;
  MPI_Reduce(
      &cg->halo->nunpack, &nhalounpack, 1, MPI_INT64_T, MPI_SUM, root, comm);
  nhalounpack /= commsize;
  MPI_Reduce(
      &cg->halo->nmpiirecv, &nhalompiirecv, 1, MPI_INT64_T, MPI_SUM, root,
      comm);
  MPI_Reduce(
      &cg->halo->nmpisend, &nhalompisend, 1, MPI_INT64_T, MPI_SUM, root,
      comm);
  MPI_Reduce(&cg->Bgemv, &Bgemv, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Bgemv /= commsize;
  MPI_Reduce(&cg->Bdot, &Bdot, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Bdot /= commsize;
  MPI_Reduce(&cg->Bnrm2, &Bnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Bnrm2 /= commsize;
  MPI_Reduce(&cg->Baxpy, &Baxpy, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Baxpy /= commsize;
  MPI_Reduce(&cg->Bcopy, &Bcopy, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Bcopy /= commsize;
  MPI_Reduce(
      &cg->Ballreduce, &Ballreduce, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Ballreduce /= commsize;
  MPI_Reduce(&cg->Bhalo, &Bhalo, 1, MPI_INT64_T, MPI_SUM, root, comm);
  Bhalo /= commsize;
  MPI_Reduce(&cg->nhalomsgs, &nhalomsgs, 1, MPI_INT64_T, MPI_SUM, root, comm);
  if (rank == root)
  {
    findent(f, indent);
    fprintf(f, "unknowns: %'" PRIdx "\n", cg->p.size);
    findent(f, indent);
    fprintf(f, "solves: %'d\n", cg->nsolves);
    findent(f, indent);
    fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
    findent(f, indent);
    fprintf(f, "total flops: %'.3f Gflop\n", 1.0e-9 * nflops);
    findent(f, indent);
    fprintf(
        f, "total flop rate: %'.3f Gflop/s\n",
        tsolve > 0 ? 1.0e-9 * nflops / tsolve : 0);
    findent(f, indent);
    fprintf(f, "total solver time: %'.6f seconds\n", tsolve);
    findent(f, indent);
    fprintf(f, "performance breakdown:\n");
    findent(f, indent);
    fprintf(
        f,
        "  gemv: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc\n",
        tgemv, ngemv, Bgemv, tgemv > 0.0 ? 1.0e-9 * Bgemv / tgemv : 0.0);
    findent(f, indent);
    fprintf(
        f,
        "  gemv off-diagonal: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc\n",
        tgemv_o, ngemv_o, Bgemv, tgemv_o > 0.0 ? 1.0e-9 * Bgemv / tgemv_o : 0.0);
    findent(f, indent);
    fprintf(
        f,
        "  dot: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc\n",
        tdot, ndot, Bdot, tdot > 0.0 ? 1.0e-9 * Bdot / tdot : 0.0);
    findent(f, indent);
    fprintf(
        f,
        "  nrm2: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc\n",
        tnrm2, nnrm2, Bnrm2, tnrm2 > 0.0 ? 1.0e-9 * Bnrm2 / tnrm2 : 0.0);
    findent(f, indent);
    fprintf(
        f,
        "  axpy: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc\n",
        taxpy, naxpy, Baxpy, taxpy > 0.0 ? 1.0e-9 * Baxpy / taxpy : 0.0);
    fprintf(
        f,
        "  preconditioner: %'.6f seconds/proc %'" PRId64 " times/proc\n",
        tprecond, nprecond);
    findent(f, indent);
    fprintf(
        f,
        "  copy: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc\n",
        tcopy, ncopy, Bcopy, tcopy > 0.0 ? 1.0e-9 * Bcopy / tcopy : 0.0);
    findent(f, indent);
    fprintf(
        f,
        "  allreduce: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc %'.3f us/op/proc\n",
        tallreduce, nallreduce, Ballreduce,
        tallreduce > 0.0 ? 1.0e-9 * Ballreduce / tallreduce : 0.0,
        nallreduce > 0.0 ? 1.0e6 * tallreduce / nallreduce : 0.0);
    findent(f, indent);
    fprintf(
        f,
        "  haloexchange: %'.6f seconds/proc %'" PRId64
        " times/proc %'" PRId64
        " B/proc %'.3f GB/s/proc %'.1f msg/proc %'.3f us/msg/proc\n",
        thalo, nhalo, Bhalo, thalo > 0.0 ? 1.0e-9 * Bhalo / thalo : 0.0,
        ((double)nhalomsgs) / commsize,
        nhalomsgs > 0.0 ? 1.0e6 * thalo / nhalomsgs / commsize : 0.0);
  }

  int *pnrecipients =
      rank == root ? malloc(commsize * sizeof(*pnrecipients)) : NULL;
  MPI_Gather(
      &cg->halo->nrecipients, 1, MPI_INT, pnrecipients, 1, MPI_INT, root,
      comm);
  int *pnsenders =
      rank == root ? malloc(commsize * sizeof(*pnsenders)) : NULL;
  MPI_Gather(
      &cg->halo->nsenders, 1, MPI_INT, pnsenders, 1, MPI_INT, root, comm);
  int *psendsize =
      rank == root ? malloc(commsize * sizeof(*psendsize)) : NULL;
  MPI_Gather(
      &cg->halo->sendsize, 1, MPI_INT, psendsize, 1, MPI_INT, root, comm);
  int *precvsize =
      rank == root ? malloc(commsize * sizeof(*precvsize)) : NULL;
  MPI_Gather(
      &cg->halo->recvsize, 1, MPI_INT, precvsize, 1, MPI_INT, root, comm);
  int *pmaxsendcount =
      rank == root ? malloc(commsize * sizeof(*pmaxsendcount)) : NULL;
  int maxsendcount = 0;
  for (int q = 0; q < cg->halo->nrecipients; q++)
    maxsendcount = maxsendcount > cg->halo->sendcounts[q]
                       ? maxsendcount
                       : cg->halo->sendcounts[q];
  MPI_Gather(
      &maxsendcount, 1, MPI_INT, pmaxsendcount, 1, MPI_INT, root, comm);
  int *pmaxrecvcount =
      rank == root ? malloc(commsize * sizeof(*pmaxrecvcount)) : NULL;
  int maxrecvcount = 0;
  for (int q = 0; q < cg->halo->nsenders; q++)
    maxrecvcount = maxrecvcount > cg->halo->recvcounts[q]
                       ? maxrecvcount
                       : cg->halo->recvcounts[q];
  MPI_Gather(
      &maxrecvcount, 1, MPI_INT, pmaxrecvcount, 1, MPI_INT, root, comm);
  if (rank == root)
  {
    for (int p = 0; p < commsize; p++)
    {
      findent(f, indent);
      fprintf(
          f,
          "    rank %'2d sends %'" PRId64 " B %'zu B/it in %'" PRId64
          " msg %'d msg/it max %'zu B/msg\n",
          p, (int64_t)nhalo * psendsize[p] * sizeof(double),
          psendsize[p] * sizeof(double), nhalo * pnrecipients[p],
          pnrecipients[p], pmaxsendcount[p] * sizeof(double));
      findent(f, indent);
      fprintf(
          f,
          "    rank %'2d receives %'" PRId64 " B %'zu B/it in %'" PRId64
          " msg %'d msg/it max %'zu B/msg\n",
          p, (int64_t)nhalo * precvsize[p] * sizeof(double),
          precvsize[p] * sizeof(double), nhalo * pnrecipients[p],
          pnrecipients[p], pmaxrecvcount[p] * sizeof(double));
    }
  }

  const struct acghaloexchange *haloexchange = cg->haloexchange;
  int maxevents = haloexchange->maxevents, nevents = 0;
  double *texchange = malloc(maxevents * sizeof(*texchange));
  double *tpack = malloc(maxevents * sizeof(*tpack));
  double *tsendrecv = malloc(maxevents * sizeof(*tsendrecv));
  double *tunpack = malloc(maxevents * sizeof(*tunpack));
  int err = acghaloexchange_profile(
      haloexchange, maxevents, &nevents, texchange, tpack, tsendrecv,
      tunpack);
  if (err)
    return err;
  double *ptexchange =
      rank == root ? malloc(commsize * sizeof(*ptexchange)) : NULL;
  double *ptpack = rank == root ? malloc(commsize * sizeof(*ptpack)) : NULL;
  double *ptsendrecv =
      rank == root ? malloc(commsize * sizeof(*ptsendrecv)) : NULL;
  double *ptunpack =
      rank == root ? malloc(commsize * sizeof(*ptunpack)) : NULL;

  /* sum and mean over all iterations per rank */
  double texchangesum = 0.0, tpacksum = 0.0, tsendrecvsum = 0.0,
         tunpacksum = 0.0;
  for (int i = 0; i < nevents; i++)
  {
    texchangesum += texchange[i];
    tpacksum += tpack[i];
    tsendrecvsum += tsendrecv[i];
    tunpacksum += tunpack[i];
  }
  MPI_Gather(
      &texchangesum, 1, MPI_DOUBLE, ptexchange, 1, MPI_DOUBLE, root, comm);
  MPI_Gather(&tpacksum, 1, MPI_DOUBLE, ptpack, 1, MPI_DOUBLE, root, comm);
  MPI_Gather(
      &tsendrecvsum, 1, MPI_DOUBLE, ptsendrecv, 1, MPI_DOUBLE, root, comm);
  MPI_Gather(&tunpacksum, 1, MPI_DOUBLE, ptunpack, 1, MPI_DOUBLE, root, comm);

  int sendsizeavg = 0, recvsizeavg = 0;
  if (rank == root)
  {
    for (int p = 0; p < commsize; p++)
    {
      sendsizeavg += psendsize[p];
      recvsizeavg += precvsize[p];
    }
    sendsizeavg /= commsize;
    recvsizeavg /= commsize;
    double texchangeavg = 0.0, tpackavg = 0.0, tsendrecvavg = 0.0,
           tunpackavg = 0.0;
    for (int p = 0; p < commsize; p++)
    {
      texchangeavg += ptexchange[p];
      tpackavg += ptpack[p];
      tsendrecvavg += ptsendrecv[p];
      tunpackavg += ptunpack[p];
    }
    texchangeavg /= (double)commsize;
    tpackavg /= (double)commsize;
    tsendrecvavg /= (double)commsize;
    tunpackavg /= (double)commsize;

    findent(f, indent);
    fprintf(
        f, "    summary of %'d most recent iterations per rank:\n",
        nevents);
    if (nevents > 0)
    {
      fprintf(
          f,
          "      mean of %'d ranks:"
          " %'.6f s %'.6f s/it total"
          " %'.6f s %'.6f s/it %'5.2f GB/s send %'5.2f GB/s recv"
          " %'.6f s %'.6f s/it %'5.2f GB/s pack"
          " %'.6f s %'.6f s/it %'5.2f GB/s unpack\n",
          commsize, texchangeavg, texchangeavg / (double)nevents,
          tsendrecvavg, tsendrecvavg / (double)nevents,
          nevents * sendsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
          nevents * recvsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
          tpackavg, tpackavg / (double)nevents,
          nevents * sendsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tpackavg,
          tunpackavg, tunpackavg / (double)nevents,
          nevents * recvsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tunpackavg);
      for (int p = 0; p < commsize; p++)
      {
        findent(f, indent);
        fprintf(
            f,
            "      rank %'2d:"
            " %'.6f s %'.6f s/it total"
            " %'.6f s %'.6f s/it %'5.2f GB/s send %'5.2f GB/s recv"
            " %'.6f s %'.6f s/it %'5.2f GB/s pack"
            " %'.6f s %'.6f s/it %'5.2f GB/s unpack\n",
            p, ptexchange[p], ptexchange[p] / (double)nevents,
            ptsendrecv[p], ptsendrecv[p] / (double)nevents,
            nevents * psendsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
            nevents * precvsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
            ptpack[p], ptpack[p] / (double)nevents,
            nevents * psendsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptpack[p],
            ptunpack[p], ptunpack[p] / (double)nevents,
            nevents * precvsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptunpack[p]);
      }
    }
  }

  if (verbose > 0)
  {
    double critpath = 0.0;
    for (int i = 0; i < nevents; i++)
    {
      MPI_Gather(
          &texchange[i], 1, MPI_DOUBLE, ptexchange, 1, MPI_DOUBLE, root,
          comm);
      MPI_Gather(
          &tpack[i], 1, MPI_DOUBLE, ptpack, 1, MPI_DOUBLE, root, comm);
      MPI_Gather(
          &tsendrecv[i], 1, MPI_DOUBLE, ptsendrecv, 1, MPI_DOUBLE, root,
          comm);
      MPI_Gather(
          &tunpack[i], 1, MPI_DOUBLE, ptunpack, 1, MPI_DOUBLE, root,
          comm);
      if (rank == root)
      {
        double texchangeavg = 0.0, tpackavg = 0.0, tsendrecvavg = 0.0,
               tunpackavg = 0.0;
        double texchangemax = 0.0;
        for (int p = 0; p < commsize; p++)
        {
          texchangeavg += ptexchange[p];
          tpackavg += ptpack[p];
          tsendrecvavg += ptsendrecv[p];
          tunpackavg += ptunpack[p];
          texchangemax = texchangemax > ptexchange[p] ? texchangemax
                                                      : ptexchange[p];
        }
        texchangeavg /= (double)commsize;
        tpackavg /= (double)commsize;
        tsendrecvavg /= (double)commsize;
        tunpackavg /= (double)commsize;
        critpath += texchangemax;

        findent(f, indent);
        fprintf(
            f, "    iteration %'4d:\n", cg->halo->nexchanges - i - 1);
        findent(f, indent);
        fprintf(
            f,
            "      mean of %'2d ranks: %'.6f s total %'.6f s %'5.2f "
            "GB/s "
            "send %'5.2f GB/s recv %'.6f s %'5.2f GB/s pack %'.6f s "
            "%'5.2f "
            "GB/s unpack\n",
            commsize, texchangeavg, tsendrecvavg,
            sendsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
            recvsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
            tpackavg,
            sendsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tpackavg,
            tunpackavg,
            recvsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tunpackavg);
        for (int p = 0; p < commsize; p++)
        {
          findent(f, indent);
          fprintf(
              f,
              "      rank %'2d: %'.6f s total %'.6f s %'5.2f GB/s "
              "send %'5.2f "
              "GB/s recv %'.6f s %'5.2f GB/s pack %'.6f s %'5.2f "
              "GB/s unpack\n",
              p, ptexchange[p], ptsendrecv[p],
              psendsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
              precvsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
              ptpack[p],
              psendsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptpack[p],
              ptunpack[p],
              precvsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptunpack[p]);
        }
      }
    }
    if (rank == root)
      fprintf(f, "      critical path: %'.6f s\n", critpath);
  }
  free(ptunpack);
  free(ptsendrecv);
  free(ptpack);
  free(ptexchange);
  free(tunpack);
  free(tsendrecv);
  free(tpack);
  free(texchange);
  if (rank == root)
  {
    free(pmaxsendcount);
    free(pmaxrecvcount);
    free(pnrecipients);
    free(pnsenders);
    free(precvsize);
    free(psendsize);
  }

  if (rank == root)
  {
    findent(f, indent);
    fprintf(f, "  other: %'.6f seconds\n", tother);
    findent(f, indent);
    fprintf(f, "last solve:\n");
    findent(f, indent);
    fprintf(f, "  stopping criterion:\n");
    findent(f, indent);
    fprintf(f, "    maximum iterations: %'d\n", cg->maxits);
    findent(f, indent);
    fprintf(
        f, "    tolerance for residual: %.*g\n", DBL_DIG, cg->residualatol);
    findent(f, indent);
    fprintf(
        f, "    tolerance for relative residual: %.*g\n", DBL_DIG,
        cg->residualrtol);
    findent(f, indent);
    fprintf(
        f, "    tolerance for difference in solution iterates: %.*g\n",
        DBL_DIG, cg->diffatol);
    findent(f, indent);
    fprintf(
        f,
        "    tolerance for relative difference in solution iterates: "
        "%.*g\n",
        DBL_DIG, cg->diffrtol);
    findent(f, indent);
    fprintf(f, "  iterations: %'d\n", cg->niterations);
    findent(f, indent);
    fprintf(f, "  right-hand side 2-norm: %.*g\n", DBL_DIG, cg->bnrm2);
    findent(f, indent);
    fprintf(f, "  initial guess 2-norm: %.*g\n", DBL_DIG, cg->x0nrm2);
    findent(f, indent);
    fprintf(f, "  initial residual 2-norm: %.*g\n", DBL_DIG, cg->r0nrm2);
    findent(f, indent);
    fprintf(f, "  residual 2-norm: %.*g\n", DBL_DIG, cg->rnrm2);
    findent(f, indent);
    fprintf(
        f, "  difference in solution iterates 2-norm: %.*g\n", DBL_DIG,
        cg->dxnrm2);
    findent(f, indent);
    fprintf(
        f, "  floating-point exceptions: %s\n",
        acgerrcodestr(ACG_ERR_FEXCEPT, 0));
  }
  return ACG_SUCCESS;
}
#endif