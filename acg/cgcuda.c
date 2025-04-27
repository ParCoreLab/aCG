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

#include "acg/config.h"
#include "acg/cgcuda.h"
#include "acg/cg-kernels-cuda.h"
#include "acg/comm.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/error.h"
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

/*
 * profiling
 */

/* #define ACG_ENABLE_PROFILING 1 */

#ifdef ACG_ENABLE_PROFILING
#define acgEventRecord(event, stream) cudaEventRecord((event), (stream))
#else
#define acgEventRecord(event, stream)
#endif

/*
 * memory management
 */

/**
 * ‘acgsolvercuda_free()’ frees storage allocated for a solver.
 */
void acgsolvercuda_free(
    struct acgsolvercuda * cg)
{
    acgvector_free(&cg->r);
    acgvector_free(&cg->p);
    acgvector_free(&cg->t);
    if (cg->w) { acgvector_free(cg->w); } free(cg->w);
    if (cg->q) { acgvector_free(cg->q); } free(cg->q);
    if (cg->z) { acgvector_free(cg->z); } free(cg->z);
    if (cg->dx) { acgvector_free(cg->dx); } free(cg->dx);
    if (cg->halo) acghalo_free(cg->halo);
    free(cg->halo);
    if (cg->haloexchange) acghaloexchange_free(cg->haloexchange);
    free(cg->haloexchange);
    if (!cg->use_nvshmem) {
        cudaFree(cg->d_bnrm2sqr);
        cudaFree(cg->d_r0nrm2sqr);
        cudaFree(cg->d_rnrm2sqr);
        cudaFree(cg->d_pdott);
    } else {
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
    if (cg->d_w) cudaFree(cg->d_w);
    if (cg->d_q) cudaFree(cg->d_q);
    if (cg->d_z) cudaFree(cg->d_z);
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
    struct acgsolvercuda * cg,
    const struct acgsymcsrmatrix * A,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    const struct acgcomm * comm)
{
    int err = acgsymcsrmatrix_vector(A, &cg->r);
    if (err) return err;
    acgvector_setzero(&cg->r);
    err = acgsymcsrmatrix_vector(A, &cg->p);
    if (err) { acgvector_free(&cg->r); return err; }
    acgvector_setzero(&cg->p);
    err = acgsymcsrmatrix_vector(A, &cg->t);
    if (err) { acgvector_free(&cg->p); acgvector_free(&cg->r); return err; }
    acgvector_setzero(&cg->t);
    cg->w = cg->q = cg->z = NULL;
    cg->dx = NULL;
    cg->halo = malloc(sizeof(*cg->halo));
    if (!cg->halo) {
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return ACG_ERR_ERRNO;
    }
    err = acgsymcsrmatrix_halo(A, cg->halo);
    if (err) {
        free(cg->haloexchange); free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return err;
    }
    cg->haloexchange = malloc(sizeof(*cg->haloexchange));
    if (!cg->haloexchange) {
        acghalo_free(cg->halo); free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return ACG_ERR_ERRNO;
    }
    cudaStream_t stream = 0;
    err = acghaloexchange_init_cuda(
        cg->haloexchange, cg->halo,
        ACG_DOUBLE, ACG_DOUBLE, comm, stream);
    if (err) {
        free(cg->haloexchange);
        acghalo_free(cg->halo); free(cg->halo);
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
    err = cudaMalloc((void **) &cg->d_one, sizeof(*cg->d_one));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_one, &one, sizeof(*cg->d_one), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_minus_one, sizeof(*cg->d_minus_one));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_minus_one, &minus_one, sizeof(*cg->d_minus_one), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_zero, sizeof(*cg->d_zero));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_zero, &zero, sizeof(*cg->d_zero), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_inf, sizeof(*cg->d_inf));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_inf, &inf, sizeof(*cg->d_inf), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;

    cg->use_nvshmem = comm->type == acgcomm_nvshmem;
    if (!cg->use_nvshmem) {
        err = cudaMalloc((void **) &cg->d_bnrm2sqr, sizeof(*cg->d_bnrm2sqr));
        if (err) return ACG_ERR_CUDA;
        err = cudaMalloc((void **) &cg->d_r0nrm2sqr, sizeof(*cg->d_r0nrm2sqr));
        if (err) return ACG_ERR_CUDA;
        err = cudaMalloc((void **) &cg->d_rnrm2sqr, 2*sizeof(*cg->d_rnrm2sqr));
        if (err) return ACG_ERR_CUDA;
        err = cudaMalloc((void **) &cg->d_pdott, sizeof(*cg->d_pdott));
        if (err) return ACG_ERR_CUDA;
    } else {
        int errcode;
        err = acgcomm_nvshmem_malloc((void **) &cg->d_bnrm2sqr, sizeof(*cg->d_bnrm2sqr), &errcode);
        if (err) return err;
        err = acgcomm_nvshmem_malloc((void **) &cg->d_r0nrm2sqr, sizeof(*cg->d_r0nrm2sqr), &errcode);
        if (err) return err;
        err = acgcomm_nvshmem_malloc((void **) &cg->d_rnrm2sqr, 2*sizeof(*cg->d_rnrm2sqr), &errcode);
        if (err) return err;
        err = acgcomm_nvshmem_malloc((void **) &cg->d_pdott, sizeof(*cg->d_pdott), &errcode);
        if (err) return err;
    }
    err = cudaMalloc((void **) &cg->d_rnrm2sqr_prev, sizeof(*cg->d_rnrm2sqr_prev));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_niterations, sizeof(*cg->d_niterations));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_converged, sizeof(*cg->d_converged));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_alpha, sizeof(*cg->d_alpha));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_minus_alpha, sizeof(*cg->d_minus_alpha));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_beta, sizeof(*cg->d_beta));
    if (err) return ACG_ERR_CUDA;

    /* allocate storage for auxiliary vectors on device */
    err = cudaMalloc((void **) &cg->d_r, cg->r.num_nonzeros*sizeof(*cg->d_r));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_p, cg->p.num_nonzeros*sizeof(*cg->d_p));
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_t, cg->t.num_nonzeros*sizeof(*cg->d_t));
    if (err) return ACG_ERR_CUDA;
    cg->d_w = cg->d_q = cg->d_z = NULL;

    /* copy sparse matrix to device */
    err = cudaMalloc((void **) &cg->d_rowptr, (A->nprows+1)*sizeof(*cg->d_rowptr));
    if (err) return ACG_ERR_CUDA;
    if (sizeof(*cg->d_rowptr) == sizeof(*A->frowptr)) {
        err = cudaMemcpy(cg->d_rowptr, A->frowptr, (A->nprows+1)*sizeof(*cg->d_rowptr), cudaMemcpyHostToDevice);
        if (err) return ACG_ERR_CUDA;
    } else {
        acgidx_t * tmprowptr = malloc((A->nprows+1)*sizeof(*tmprowptr));
        if (!tmprowptr) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i <= A->nprows; i++) {
            if (A->frowptr[i] > ACGIDX_T_MAX) { return ACG_ERR_INDEX_OUT_OF_BOUNDS; }
            tmprowptr[i] = A->frowptr[i];
        }
        err = cudaMemcpy(cg->d_rowptr, tmprowptr, (A->nprows+1)*sizeof(*cg->d_rowptr), cudaMemcpyHostToDevice);
        if (err) return ACG_ERR_CUDA;
        free(tmprowptr);
    }
    err = cudaMalloc((void **) &cg->d_colidx, A->fnpnzs*sizeof(*cg->d_colidx));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_colidx, A->fcolidx, A->fnpnzs*sizeof(*cg->d_colidx), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_a, A->fnpnzs*sizeof(*cg->d_a));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_a, A->fa, A->fnpnzs*sizeof(*cg->d_a), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;

    /* copy sparse matrix to device */
    err = cudaMalloc((void **) &cg->d_orowptr, (A->nborderrows+A->nghostrows+1)*sizeof(*cg->d_orowptr));
    if (err) return ACG_ERR_CUDA;
    if (sizeof(*cg->d_orowptr) == sizeof(*A->orowptr)) {
        err = cudaMemcpy(cg->d_orowptr, A->orowptr, (A->nborderrows+A->nghostrows+1)*sizeof(*cg->d_orowptr), cudaMemcpyHostToDevice);
        if (err) return ACG_ERR_CUDA;
    } else {
        acgidx_t * tmprowptr = malloc((A->nborderrows+A->nghostrows+1)*sizeof(*tmprowptr));
        if (!tmprowptr) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i <= A->nborderrows+A->nghostrows; i++) {
            if (A->orowptr[i] > ACGIDX_T_MAX) { return ACG_ERR_INDEX_OUT_OF_BOUNDS; }
            tmprowptr[i] = A->orowptr[i];
        }
        err = cudaMemcpy(cg->d_orowptr, tmprowptr, (A->nborderrows+A->nghostrows+1)*sizeof(*cg->d_orowptr), cudaMemcpyHostToDevice);
        if (err) return ACG_ERR_CUDA;
        free(tmprowptr);
    }
    err = cudaMalloc((void **) &cg->d_ocolidx, A->onpnzs*sizeof(*cg->d_ocolidx));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_ocolidx, A->ocolidx, A->onpnzs*sizeof(*cg->d_ocolidx), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    err = cudaMalloc((void **) &cg->d_oa, A->onpnzs*sizeof(*cg->d_oa));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(cg->d_oa, A->oa, A->onpnzs*sizeof(*cg->d_oa), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;

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
    cg->tgemv = cg->tdot = cg->tnrm2 = cg->taxpy = cg->tcopy = cg->tallreduce = cg->thalo = 0;
    cg->ngemv = cg->ndot = cg->nnrm2 = cg->naxpy = cg->ncopy = cg->nallreduce = cg->nhalo = 0;
    cg->Bgemv = cg->Bdot = cg->Bnrm2 = cg->Baxpy = cg->Bcopy = cg->Ballreduce = cg->Bhalo = 0;
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
int acgsolvercuda_solve(
    struct acgsolvercuda * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup);

/*
 * iterative solution procedure in distributed memory using MPI
 */

#if defined(ACG_HAVE_MPI) && defined(ACG_HAVE_CUBLAS) && defined(ACG_HAVE_CUSPARSE)
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
int acgsolvercuda_solvempi(
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
    int tag,
    int * errcode,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    cusparseSpMVAlg_t cusparse_spmv_alg)
{
    int err;
    if (b->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->r.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->p.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->t.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    /* not implemented */
    if (diffatol > 0 || diffrtol > 0) return ACG_ERR_NOT_SUPPORTED;

    int commsize, rank;
    acgcomm_size(comm, &commsize);
    acgcomm_rank(comm, &rank);

    /* /\* If the stopping criterion is based on the difference in */
    /*  * solution from one iteration to the next, then allocate */
    /*  * additional storage for storing the difference. *\/ */
    /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
    /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO; */
    /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
    /* } */

    cudaStream_t stream = 0;
    const struct acghalo * halo = cg->halo;
    double * d_bnrm2sqr = cg->d_bnrm2sqr;
    double * d_rnrm2sqr = cg->d_rnrm2sqr;
    double * d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
    double * d_pdott = cg->d_pdott;
    double * d_alpha = cg->d_alpha;
    double * d_minus_alpha = cg->d_minus_alpha;
    double * d_beta = cg->d_beta;
    double * d_one = cg->d_one;
    double * d_minus_one = cg->d_minus_one;
    double * d_zero = cg->d_zero;
    double * d_r = cg->d_r;
    double * d_p = cg->d_p;
    double * d_t = cg->d_t;
    acgidx_t * d_rowptr = cg->d_rowptr;
    acgidx_t * d_colidx = cg->d_colidx;
    double * d_a = cg->d_a;
    acgidx_t * d_orowptr = cg->d_orowptr;
    acgidx_t * d_ocolidx = cg->d_ocolidx;
    double * d_oa = cg->d_oa;

    /* configure cublas and cusparse to use device-side pointers */
    cublasPointerMode_t cublaspointermode;
    err = cublasGetPointerMode(cublas, &cublaspointermode);
    if (err) return ACG_ERR_CUBLAS;
    err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
    if (err) return ACG_ERR_CUBLAS;
    cusparsePointerMode_t cusparsepointermode;
    err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }

    double * rnrm2sqr;
    err = cudaMallocHost((void **) &rnrm2sqr, sizeof(*rnrm2sqr));
    if (err) return ACG_ERR_CUDA;
    cudaStream_t copystream;
    err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
    if (err) return ACG_ERR_CUDA;
    cudaEvent_t rnrm2sqrready;
    cudaEventCreateWithFlags(&rnrm2sqrready, cudaEventDisableTiming);
    /* cudaEvent_t rnrm2sqrreceived; */
    /* err = cudaEventCreateWithFlags(&rnrm2sqrreceived, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA; */

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

    /* used to overlap P2P communication with SpMV */
    cudaStream_t commstream;
    err = cudaStreamCreateWithFlags(&commstream, cudaStreamNonBlocking);
    if (err) return ACG_ERR_CUDA;
    cudaEvent_t xreadytosend, xreceived;
    err = cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    err = cudaEventRecord(xreadytosend, stream); if (err) return ACG_ERR_CUDA;
    err = cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    cudaEvent_t preadytosend, preceived;
    err = cudaEventCreateWithFlags(&preadytosend, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    err = cudaEventRecord(preadytosend, stream); if (err) return ACG_ERR_CUDA;
    err = cudaEventCreateWithFlags(&preceived, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;

    /* create cusparse matrix and vectors */
    cusparseDnVecDescr_t vecx, vecr, vecp, vect;
    err = cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseCreateDnVec(&vecp, A->nownedrows, d_p, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseCreateDnVec(&vect, A->nownedrows, d_t, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    cusparseDnVecDescr_t vecxo, vecro, vecpo, vecto;
    if (commsize > 1) {
        err = cusparseCreateDnVec(&vecxo, A->nborderrows+A->nghostrows, d_x+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cusparseCreateDnVec(&vecro, A->nborderrows+A->nghostrows, d_r+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cusparseCreateDnVec(&vecpo, A->nborderrows+A->nghostrows, d_p+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cusparseCreateDnVec(&vecto, A->nborderrows+A->nghostrows, d_t+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    }

    cusparseSpMatDescr_t matA;
    err = cusparseCreateCsr(
        &matA, A->nownedrows, A->nownedrows, A->fnpnzs,
        d_rowptr, d_colidx, d_a,
        CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    size_t buffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
        cusparse_spmv_alg, &buffersize);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    void * d_buffer;
    err = cudaMalloc(&d_buffer, buffersize);
    if (err) return ACG_ERR_CUDA;
#if (CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
        cusparse_spmv_alg, d_buffer);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
#endif

    cusparseSpMatDescr_t matO;
    void * d_obuffer;
    if (commsize > 1) {
        err = cusparseCreateCsr(
            &matO, A->nborderrows+A->nghostrows, A->nborderrows+A->nghostrows, A->onpnzs,
            d_orowptr, d_ocolidx, d_oa,
            CUSPARSE_IDX_T, CUSPARSE_IDX_T,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        size_t obuffersize;
        err = cusparseSpMV_bufferSize(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
            cusparse_spmv_alg, &obuffersize);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cudaMalloc(&d_obuffer, obuffersize);
        if (err) return ACG_ERR_CUDA;
#if (CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
        err = cusparseSpMV_preprocess(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
            cusparse_spmv_alg, d_obuffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
#endif
    }

    /* create timing events for profiling */
    acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0, nallreduce = 0, nhalo = 0;
    cudaEvent_t * tgemv, * tdot, * tnrm2, * taxpy, * tcopy, * tallreduce, * thalo;
#if defined(ACG_ENABLE_PROFILING)
    tgemv = malloc(2*(maxits+1)*sizeof(*tgemv));
    if (!tgemv) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+1); i++) cudaEventCreate(&tgemv[i]);
    tdot = malloc(2*maxits*sizeof(*tdot));
    if (!tdot) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*maxits; i++) cudaEventCreate(&tdot[i]);
    tnrm2 = malloc(2*(maxits+2)*sizeof(*tnrm2));
    if (!tnrm2) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+2); i++) cudaEventCreate(&tnrm2[i]);
    taxpy = malloc(2*(3*maxits)*sizeof(*taxpy));
    if (!taxpy) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(3*maxits); i++) cudaEventCreate(&taxpy[i]);
    tcopy = malloc(2*2*sizeof(*tcopy));
    if (!tcopy) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*2; i++) cudaEventCreate(&tcopy[i]);
    tallreduce = malloc(2*(2*maxits+2)*sizeof(*tallreduce));
    if (!tallreduce) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(2*maxits+2); i++) cudaEventCreate(&tallreduce[i]);
    thalo = malloc(2*(maxits+1)*sizeof(*thalo));
    if (!thalo) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+1); i++) cudaEventCreate(&thalo[i]);
#endif

    /* warmup iterations for dot/allreduce */
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice);
        err = cublasDdot(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
        if (err) return ACG_ERR_CUBLAS;
        if (commsize > 1) acgcomm_allreduce(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = cublasDdot(cublas, cg->r.num_nonzeros-cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err) return ACG_ERR_CUBLAS;
        if (commsize > 1) acgcomm_allreduce(ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = cublasDdot(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_p, 1, d_t, 1, d_pdott);
        if (err) return ACG_ERR_CUBLAS;
        if (commsize > 1) acgcomm_allreduce(ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
    }
    cudaMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_pdott, d_zero, sizeof(*d_pdott), cudaMemcpyDeviceToDevice);

    /* warmup iterations for halo exchange/SpMV */
    for (int i = 0; i < warmup; i++) {
        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, xreadytosend, 0); if (err) return ACG_ERR_CUDA;
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
        }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
            err = cudaEventRecord(xreceived, commstream); if (err) return ACG_ERR_CUDA;
            err = cudaStreamWaitEvent(stream, xreceived, 0); if (err) return ACG_ERR_CUDA;
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
            err = cudaEventRecord(xreadytosend, stream); if (err) return ACG_ERR_CUDA;
        }

        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, preadytosend, 0); if (err) return ACG_ERR_CUDA;
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
        }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecp, d_zero, vect, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
            err = cudaEventRecord(preceived, commstream); if (err) return ACG_ERR_CUDA;
            err = cudaStreamWaitEvent(stream, preceived, 0); if (err) return ACG_ERR_CUDA;
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecpo, d_one, vecto, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
            err = cudaEventRecord(preadytosend, stream); if (err) return ACG_ERR_CUDA;
        }
    }

    /* warmup iterations for axpy */
    for (int i = 0; i < warmup; i++) {
        err = acgsolvercuda_daxpy_alpha(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_zero, d_one, d_p, d_x);
        if (err) return err;
        err = acgsolvercuda_daypx_beta(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_one, d_one, d_p, d_r);
        if (err) return err;
    }

    /* warmup iterations for copy */
    for (int i = 0; i < warmup; i++) {
        err = cublasDcopy(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_r, 1);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUBLAS; }
        err = cublasDcopy(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_r, 1, d_p, 1);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUBLAS; }
    }

    /* set initial state */
    bool converged = false;
    cg->nsolves++; cg->niterations = 0;
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
    if (err) return err;
    cudaStreamSynchronize(stream);
    gettime(&t0);

    /* compute right-hand side norm */
    double bnrm2sqr;
    acgEventRecord(tnrm2[2*nnrm2+0], 0);
    err = cublasDdot(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
    acgEventRecord(tnrm2[2*nnrm2+1], 0); nnrm2++; cg->nnrm2++;
    cg->nflops += 2*(b->num_nonzeros-b->num_ghost_nonzeros);
    cg->Bnrm2 += (b->num_nonzeros-b->num_ghost_nonzeros)*sizeof(*b->x);
    if (commsize > 1) {
        acgEventRecord(tallreduce[2*nallreduce+0], 0);
        err = acgcomm_allreduce(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
        cg->Ballreduce += sizeof(bnrm2sqr);
    }
    err = cudaMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->bnrm2 = sqrt(bnrm2sqr);

    /* /\* compute norm of initial guess *\/ */
    /* if (diffatol > 0 || diffrtol > 0) { */
    /*     gettime(&tnrm20); */
    /*     double x0nrm2sqr; */
    /*     err = acgvector_dnrm2sqr(x, &x0nrm2sqr, &cg->nflops, &cg->Bnrm2); */
    /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; } */
    /*     gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21); */
    /*     gettime(&tallreduce0); */
    /*     err = MPI_Allreduce(MPI_IN_PLACE, &x0nrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm); */
    /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *errcode = err; return ACG_ERR_MPI; } */
    /*     cg->Ballreduce += sizeof(x0nrm2sqr); */
    /*     gettime(&tallreduce1); cg->nallreduce++; cg->tallreduce += elapsed(tallreduce0,tallreduce1); */
    /*     cg->x0nrm2 = sqrt(x0nrm2sqr); */
    /*     diffrtol *= cg->x0nrm2; */
    /* } */

    /* compute initial residual, r₀ = b-A*x₀ */
    acgEventRecord(tcopy[2*ncopy+0], 0);
    err = cublasDcopy(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
    acgEventRecord(tcopy[2*ncopy+1], 0); ncopy++; cg->ncopy++;
    cg->Bcopy += (b->num_nonzeros-b->num_ghost_nonzeros)*(sizeof(*cg->r.x)+sizeof(*b->x));

    if (commsize > 1) {
        err = acghalo_exchange_cuda_begin(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    }
    acgEventRecord(tgemv[2*ngemv+0], 0);
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    if (commsize > 1) {
        acgEventRecord(thalo[2*nhalo+0], 0);
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(thalo[2*nhalo+1], 0); nhalo++; cg->nhalo++;
        cg->Bhalo += cg->halo->sendsize*sizeof(*x->x);
        cg->nhalomsgs += cg->halo->nrecipients;
        err = cudaEventRecord(xreceived, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaStreamWaitEvent(stream, xreceived, 0);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    }
    acgEventRecord(tgemv[2*ngemv+1], 0); ngemv++; cg->ngemv++;
    cg->nflops += 3*(int64_t)(A->fnpnzs+A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs+A->onpnzs)*(sizeof(*A->fa)+sizeof(*A->fcolidx))
        + A->nownedrows*(sizeof(*A->frowptr)+sizeof(*cg->r.x))
        + (A->nborderrows+A->nghostrows)*sizeof(*A->orowptr)
        + x->num_nonzeros*sizeof(*x->x);

    /* compute initial search direction: p = r₀ */
    acgEventRecord(tcopy[2*ncopy+0], 0);
    err = cublasDcopy(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_r, 1, d_p, 1);
    if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
    acgEventRecord(tcopy[2*ncopy+1], 0); ncopy++; cg->ncopy++;
    cg->Bcopy += (cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*(sizeof(*cg->p.x)+sizeof(*cg->r.x));
    err = cudaEventRecord(preadytosend, stream); if (err) return ACG_ERR_CUDA;

    /* compute initial residual norm */
    acgEventRecord(tnrm2[2*nnrm2+0], 0);
    err = cublasDdot(cublas, cg->r.num_nonzeros-cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
    if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
    acgEventRecord(tnrm2[2*nnrm2+1], 0); nnrm2++; cg->nnrm2++;
    cg->nflops += 2*(cg->r.num_nonzeros-cg->r.num_ghost_nonzeros);
    cg->Bnrm2 += (cg->r.num_nonzeros-cg->r.num_ghost_nonzeros)*sizeof(*cg->r.x);
    if (commsize > 1) {
        acgEventRecord(tallreduce[2*nallreduce+0], 0);
        err = acgcomm_allreduce(ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
        cg->Ballreduce += sizeof(*rnrm2sqr);
    }
    err = cudaMemcpy(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->rnrm2 = cg->r0nrm2 = sqrt(*rnrm2sqr);
    residualrtol *= cg->r0nrm2;

    /* initial convergence test */
    if ((residualatol > 0 && cg->rnrm2 < residualatol) ||
        (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
        gettime(&t1); cg->tsolve += elapsed(t0,t1);
        return ACG_SUCCESS;
    }

    /* iterative solver loop */
    for (int k = 0; k < maxits; k++) {
        /* compute t = Ap */
        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, preadytosend, 0);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        }
        acgEventRecord(tgemv[2*ngemv+0], 0);
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecp, d_zero, vect, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            acgEventRecord(thalo[2*nhalo+0], 0);
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            acgEventRecord(thalo[2*nhalo+1], 0); nhalo++; cg->nhalo++;
            cg->Bhalo += cg->halo->sendsize*sizeof(*cg->p.x);
            cg->nhalomsgs += cg->halo->nrecipients;
            err = cudaEventRecord(preceived, commstream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = cudaStreamWaitEvent(stream, preceived, 0);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecpo, d_one, vecto, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        }
        acgEventRecord(tgemv[2*ngemv+1], 0); ngemv++; cg->ngemv++;
        cg->nflops += 3*(int64_t)(A->fnpnzs+A->onpnzs);
        cg->Bgemv +=
            (int64_t)(A->fnpnzs+A->onpnzs)*(sizeof(*A->fa)+sizeof(*A->fcolidx))
            + A->nownedrows*(sizeof(*A->frowptr)+sizeof(*cg->t.x))
            + (A->nborderrows+A->nghostrows)*sizeof(*A->orowptr)
            + cg->p.num_nonzeros*sizeof(*cg->p.x);

        /* compute (p,Ap) */
        acgEventRecord(tdot[2*ndot+0], 0);
        err = cublasDdot(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_p, 1, d_t, 1, d_pdott);
        if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
        acgEventRecord(tdot[2*ndot+1], 0); ndot++; cg->ndot++;
        cg->nflops += 2*(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros);
        cg->Bdot += (cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*(sizeof(*cg->p.x)+sizeof(*cg->t.x));
        if (commsize > 1) {
            acgEventRecord(tallreduce[2*nallreduce+0], 0);
#ifndef HOST_ALLREDUCE
            err = acgcomm_allreduce(ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
#else
            double pdott;
            cudaMemcpy(&pdott, d_pdott, sizeof(*d_pdott), cudaMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, &pdott, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
            cudaMemcpy(d_pdott, &pdott, sizeof(*d_pdott), cudaMemcpyHostToDevice);
#endif
            acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
            cg->Ballreduce += sizeof(*d_pdott);
        }
        err = cudaMemcpyAsync(d_rnrm2sqr_prev, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }

        /* update residual, r = -αt + r */
        acgEventRecord(taxpy[2*naxpy+0], 0);
#ifndef NO_FUSED_KERNELS
        err = acgsolvercuda_daxpy_minus_alpha(cg->t.num_nonzeros-cg->t.num_ghost_nonzeros, d_rnrm2sqr, d_pdott, d_t, d_r);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
#else
        err = acgsolvercuda_alpha(d_alpha, d_minus_alpha, d_rnrm2sqr, d_pdott);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        err = cublasDaxpy(cublas, cg->t.num_nonzeros-cg->t.num_ghost_nonzeros, d_minus_alpha, d_t, 1, d_r, 1);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
#endif
        acgEventRecord(taxpy[2*naxpy+1], 0); naxpy++; cg->naxpy++;
        cg->nflops += 2*(cg->t.num_nonzeros-cg->t.num_ghost_nonzeros);
        cg->Baxpy += (cg->t.num_nonzeros-cg->t.num_ghost_nonzeros)*(sizeof(*cg->t.x)+sizeof(*cg->r.x));

        /* compute residual norm */
        acgEventRecord(tnrm2[2*nnrm2+0], 0);
        err = cublasDdot(cublas, cg->r.num_nonzeros-cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
        acgEventRecord(tnrm2[2*nnrm2+1], 0); nnrm2++; cg->nnrm2++;
        cg->nflops += 2*(cg->r.num_nonzeros-cg->r.num_ghost_nonzeros);
        cg->Bnrm2 += (cg->r.num_nonzeros-cg->r.num_ghost_nonzeros)*sizeof(*cg->r.x);
#ifndef HOST_ALLREDUCE
        if (commsize > 1) {
            acgEventRecord(tallreduce[2*nallreduce+0], 0);
            err = acgcomm_allreduce(ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
            cg->Ballreduce += sizeof(*rnrm2sqr);
        }
        err = cudaEventRecord(rnrm2sqrready, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost, copystream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        /* err = cudaEventRecord(rnrm2sqrreceived, copystream); */
        /* if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; } */
#else
        if (commsize > 1) {
            acgEventRecord(tallreduce[2*nallreduce+0], 0);
            cudaMemcpy(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, rnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
            err = cudaMemcpyAsync(d_rnrm2sqr, rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyHostToDevice, copystream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
            cg->Ballreduce += sizeof(*rnrm2sqr);
        } else {
            err = cudaEventRecord(rnrm2sqrready, stream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = cudaMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost, copystream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        }
#endif

        /* update solution, x = αp + x, where α = (r,r)/(p,t) */
        acgEventRecord(taxpy[2*naxpy+0], 0);
#ifndef NO_FUSED_KERNELS
        err = acgsolvercuda_daxpy_alpha(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_rnrm2sqr_prev, d_pdott, d_p, d_x);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
#else
        err = cublasDaxpy(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_alpha, d_p, 1, d_x, 1);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
#endif
        acgEventRecord(taxpy[2*naxpy+1], 0); naxpy++; cg->naxpy++;
        cg->nflops += 2*(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros);
        cg->Baxpy += (cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*(sizeof(*cg->p.x)+sizeof(*x->x));

        /* update search direction, p = βp + r, where β = (rₖ,rₖ)/(rₖ₋₁,rₖₖ₋₁) */
        acgEventRecord(taxpy[2*naxpy+0], 0);
#ifndef NO_FUSED_KERNELS
        err = acgsolvercuda_daypx_beta(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_rnrm2sqr, d_rnrm2sqr_prev, d_p, d_r);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
#else
        err = acgsolvercuda_beta(d_beta, d_rnrm2sqr, d_rnrm2sqr_prev);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        err = cublasDscal(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_beta, d_p, 1);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
        err = cublasDaxpy(cublas, cg->p.num_nonzeros-cg->p.num_ghost_nonzeros, d_one, d_r, 1, d_p, 1);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
#endif
        acgEventRecord(taxpy[2*naxpy+1], 0); naxpy++; cg->naxpy++;
        cg->nflops += 2*(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros);
        cg->Baxpy += (cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*(sizeof(*cg->p.x)+sizeof(*cg->r.x));
        err = cudaEventRecord(preadytosend, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }

        /* convergence tests */
        /* cudaEventSynchronize(rnrm2sqrreceived); */
        cudaStreamSynchronize(copystream);
        cg->rnrm2 = sqrt(*rnrm2sqr);
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) ||
            (diffrtol > 0 && cg->dxnrm2 < diffrtol) ||
            (residualatol > 0 && cg->rnrm2 < residualatol) ||
            (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            cudaStreamSynchronize(stream);
            cg->ntotaliterations++; cg->niterations++;
            converged = true;
            break;
        }
        cg->ntotaliterations++; cg->niterations++;
    }
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

#if defined(ACG_ENABLE_PROFILING)
    /* record profiling information */
    float t;
    for (acgidx_t i = 0; i < ngemv; i++) {
        cudaEventSynchronize(tgemv[2*i+1]);
        cudaEventElapsedTime(&t, tgemv[2*i+0], tgemv[2*i+1]);
        cg->tgemv += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < ndot; i++) {
        cudaEventSynchronize(tdot[2*i+1]);
        cudaEventElapsedTime(&t, tdot[2*i+0], tdot[2*i+1]);
        cg->tdot += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < nnrm2; i++) {
        cudaEventSynchronize(tnrm2[2*i+1]);
        cudaEventElapsedTime(&t, tnrm2[2*i+0], tnrm2[2*i+1]);
        cg->tnrm2 += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < naxpy; i++) {
        cudaEventSynchronize(taxpy[2*i+1]);
        cudaEventElapsedTime(&t, taxpy[2*i+0], taxpy[2*i+1]);
        cg->taxpy += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < ncopy; i++) {
        cudaEventSynchronize(tcopy[2*i+1]);
        cudaEventElapsedTime(&t, tcopy[2*i+0], tcopy[2*i+1]);
        cg->tcopy += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < nallreduce; i++) {
        cudaEventSynchronize(tallreduce[2*i+1]);
        cudaEventElapsedTime(&t, tallreduce[2*i+0], tallreduce[2*i+1]);
        cg->tallreduce += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < nhalo; i++) {
        cudaEventSynchronize(thalo[2*i+1]);
        cudaEventElapsedTime(&t, thalo[2*i+0], thalo[2*i+1]);
        cg->thalo += 1.0e-3*t;
    }
#endif

    /* copy solution back to host */
    err = cudaMemcpy(x->x, d_x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;

    /* free cusparse matrix and vectors */
    cusparseDestroyDnVec(vecx);
    cusparseDestroyDnVec(vecr);
    cusparseDestroyDnVec(vecp);
    cusparseDestroyDnVec(vect);
    if (commsize > 1) {
        cusparseDestroyDnVec(vecxo);
        cusparseDestroyDnVec(vecro);
        cusparseDestroyDnVec(vecpo);
        cusparseDestroyDnVec(vecto);
    }
    cusparseDestroySpMat(matA);
    cudaFree(d_buffer);
    if (commsize > 1) {
        cusparseDestroySpMat(matO);
        cudaFree(d_obuffer);
    }
    cudaFree(d_x); cudaFree(d_b);
    cudaFreeHost(rnrm2sqr);
    cudaStreamDestroy(commstream);
    cudaStreamDestroy(copystream);

    /* reset cusparse and cublas pointer modes */
    err = cusparseSetPointerMode(cusparse, cusparsepointermode);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cublasSetPointerMode(cublas, cublaspointermode);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUBLAS; }

    /* check for CUDA errors */
    if (cudaGetLastError() != cudaSuccess)
        return ACG_ERR_CUDA;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged) return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
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
int acgsolvercuda_solve_pipelined(
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
    int tag,
    int * errcode,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse)
{
    int err;
    if (b->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->r.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->p.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->t.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    /* not implemented */
    if (diffatol > 0 || diffrtol > 0) return ACG_ERR_NOT_SUPPORTED;

    int commsize, rank;
    acgcomm_size(comm, &commsize);
    acgcomm_rank(comm, &rank);

    /* allocate extra vectors needed for pipelined CG */
    if (!cg->w) {
        cg->w = malloc(sizeof(*cg->w)); if (!cg->w) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->w, x); if (err) return err;
        err = cudaMalloc((void **) &cg->d_w, cg->w->num_nonzeros*sizeof(*cg->d_w));
        if (err) return ACG_ERR_CUDA;
    }
    if (!cg->q) {
        cg->q = malloc(sizeof(*cg->q)); if (!cg->q) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->q, x); if (err) return err;
        err = cudaMalloc((void **) &cg->d_q, cg->q->num_nonzeros*sizeof(*cg->d_q));
        if (err) return ACG_ERR_CUDA;
    }
    if (!cg->z) {
        cg->z = malloc(sizeof(*cg->z)); if (!cg->z) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->z, x); if (err) return err;
        err = cudaMalloc((void **) &cg->d_z, cg->z->num_nonzeros*sizeof(*cg->d_z));
        if (err) return ACG_ERR_CUDA;
    }

    /* /\* If the stopping criterion is based on the difference in */
    /*  * solution from one iteration to the next, then allocate */
    /*  * additional storage for storing the difference. *\/ */
    /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
    /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO; */
    /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
    /* } */

    cudaStream_t stream = 0;
    const struct acghalo * halo = cg->halo;
    double * d_bnrm2sqr = cg->d_bnrm2sqr;
    double * d_rnrm2sqr = &cg->d_rnrm2sqr[0];
    double * d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
    double * d_delta = &cg->d_rnrm2sqr[1];
    double * d_alpha = cg->d_alpha;
    double * d_minus_alpha = cg->d_minus_alpha;
    double * d_beta = cg->d_beta;
    double * d_one = cg->d_one;
    double * d_minus_one = cg->d_minus_one;
    double * d_zero = cg->d_zero;
    double * d_inf = cg->d_inf;
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

    /* configure cublas and cusparse to use device-side pointers */
    cublasPointerMode_t cublaspointermode;
    err = cublasGetPointerMode(cublas, &cublaspointermode);
    if (err) return ACG_ERR_CUBLAS;
    err = cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
    if (err) return ACG_ERR_CUBLAS;
    cusparsePointerMode_t cusparsepointermode;
    err = cusparseGetPointerMode(cusparse, &cusparsepointermode);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseSetPointerMode(cusparse, CUSPARSE_POINTER_MODE_DEVICE);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }

    double * rnrm2sqr;
    err = cudaMallocHost((void **) &rnrm2sqr,  sizeof(*rnrm2sqr));
    if (err) return ACG_ERR_CUDA;
    cudaStream_t copystream;
    err = cudaStreamCreateWithFlags(&copystream, cudaStreamNonBlocking);
    if (err) return ACG_ERR_CUDA;
    cudaEvent_t rnrm2sqrready;
    cudaEventCreateWithFlags(&rnrm2sqrready, cudaEventDisableTiming);

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

    /* used to overlap P2P communication with SpMV */
    cudaStream_t commstream;
    err = cudaStreamCreateWithFlags(&commstream, cudaStreamNonBlocking);
    if (err) return ACG_ERR_CUDA;
    cudaEvent_t xreadytosend, xreceived;
    err = cudaEventCreateWithFlags(&xreadytosend, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    err = cudaEventRecord(xreadytosend, stream); if (err) return ACG_ERR_CUDA;
    err = cudaEventCreateWithFlags(&xreceived, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    cudaEvent_t rreadytosend, rreceived;
    err = cudaEventCreateWithFlags(&rreadytosend, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    err = cudaEventRecord(rreadytosend, stream); if (err) return ACG_ERR_CUDA;
    err = cudaEventCreateWithFlags(&rreceived, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    cudaEvent_t wreadytosend, wreceived;
    err = cudaEventCreateWithFlags(&wreadytosend, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;
    err = cudaEventRecord(wreadytosend, stream); if (err) return ACG_ERR_CUDA;
    err = cudaEventCreateWithFlags(&wreceived, cudaEventDisableTiming); if (err) return ACG_ERR_CUDA;

    /* create cusparse matrix and vectors */
    cusparseDnVecDescr_t vecx, vecr, vecw, vecq;
    err = cusparseCreateDnVec(&vecx, A->nownedrows, d_x, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseCreateDnVec(&vecr, A->nownedrows, d_r, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseCreateDnVec(&vecw, A->nownedrows, d_w, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cusparseCreateDnVec(&vecq, A->nownedrows, d_q, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    cusparseDnVecDescr_t vecxo, vecro, vecwo, vecqo;
    if (commsize > 1) {
        err = cusparseCreateDnVec(&vecxo, A->nborderrows+A->nghostrows, d_x+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cusparseCreateDnVec(&vecro, A->nborderrows+A->nghostrows, d_r+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cusparseCreateDnVec(&vecwo, A->nborderrows+A->nghostrows, d_w+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cusparseCreateDnVec(&vecqo, A->nborderrows+A->nghostrows, d_q+A->borderrowoffset, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    }

    cusparseSpMatDescr_t matA;
    err = cusparseCreateCsr(
        &matA, A->nownedrows, A->nownedrows, A->fnpnzs,
        d_rowptr, d_colidx, d_a,
        CUSPARSE_IDX_T, CUSPARSE_IDX_T,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    size_t buffersize;
    err = cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, &buffersize);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    void * d_buffer;
    err = cudaMalloc(&d_buffer, buffersize);
    if (err) return ACG_ERR_CUDA;
#if (CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
    err = cusparseSpMV_preprocess(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
#endif

    cusparseSpMatDescr_t matO;
    void * d_obuffer;
    if (commsize > 1) {
        err = cusparseCreateCsr(
            &matO, A->nborderrows+A->nghostrows, A->nborderrows+A->nghostrows, A->onpnzs,
            d_orowptr, d_ocolidx, d_oa,
            CUSPARSE_IDX_T, CUSPARSE_IDX_T,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        size_t obuffersize;
        err = cusparseSpMV_bufferSize(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cudaMalloc(&d_obuffer, obuffersize);
        if (err) return ACG_ERR_CUDA;
#if (CUSPARSE_VER_MAJOR > 12 || CUSPARSE_VER_MAJOR == 12 && CUSPARSE_VER_MINOR >= 4)
        err = cusparseSpMV_preprocess(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
#endif
    }

    /* create timing events for profiling */
    acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0, nallreduce = 0, nhalo = 0;
    cudaEvent_t * tgemv, * tdot, * tnrm2, * taxpy, * tcopy, * tallreduce, * thalo;
#if defined(ACG_ENABLE_PROFILING)
    tgemv = malloc(2*(maxits+2)*sizeof(*tgemv));
    if (!tgemv) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+2); i++) cudaEventCreate(&tgemv[i]);
    tdot = malloc(2*maxits*sizeof(*tdot));
    if (!tdot) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*maxits; i++) cudaEventCreate(&tdot[i]);
    tnrm2 = malloc(2*(maxits+1)*sizeof(*tnrm2));
    if (!tnrm2) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+1); i++) cudaEventCreate(&tnrm2[i]);
    taxpy = malloc(2*maxits*sizeof(*taxpy));
    if (!taxpy) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*maxits; i++) cudaEventCreate(&taxpy[i]);
    tcopy = malloc(2*1*sizeof(*tcopy));
    if (!tcopy) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*1; i++) cudaEventCreate(&tcopy[i]);
    tallreduce = malloc(2*(maxits+1)*sizeof(*tallreduce));
    if (!tallreduce) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+1); i++) cudaEventCreate(&tallreduce[i]);
    thalo = malloc(2*(maxits+2)*sizeof(*thalo));
    if (!thalo) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*(maxits+2); i++) cudaEventCreate(&thalo[i]);
#endif

    /* warmup iterations for dot/allreduce */
    for (int i = 0; i < warmup; i++) {
        cudaMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_delta, d_zero, sizeof(*d_delta), cudaMemcpyDeviceToDevice);
        err = cublasDdot(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
        if (err) return ACG_ERR_CUBLAS;
        if (commsize > 1) acgcomm_allreduce(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = cublasDdot(cublas, cg->r.num_nonzeros-cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err) return ACG_ERR_CUBLAS;
        err = cublasDdot(cublas, cg->w->num_nonzeros-cg->w->num_ghost_nonzeros, d_w, 1, d_r, 1, d_delta);
        if (err) return ACG_ERR_CUBLAS;
        if (commsize > 1) acgcomm_allreduce(ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        cudaStreamSynchronize(stream);
    }
    cudaMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_delta, d_zero, sizeof(*d_delta), cudaMemcpyDeviceToDevice);

    /* warmup iterations for halo exchange/SpMV */
    for (int i = 0; i < warmup; i++) {
        /* r = b-Ax */
        err = cublasDcopy(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_r, 1);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUBLAS; }
        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, xreadytosend, 0); if (err) return ACG_ERR_CUDA;
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
        }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
            err = cudaEventRecord(xreceived, commstream); if (err) return ACG_ERR_CUDA;
            err = cudaStreamWaitEvent(stream, xreceived, 0); if (err) return ACG_ERR_CUDA;
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
            err = cudaEventRecord(xreadytosend, stream); if (err) return ACG_ERR_CUDA;
        }

        /* w = Ar */
        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, rreadytosend, 0); if (err) return ACG_ERR_CUDA;
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
        }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecr, d_zero, vecw, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
            err = cudaEventRecord(rreceived, commstream); if (err) return ACG_ERR_CUDA;
            err = cudaStreamWaitEvent(stream, rreceived, 0); if (err) return ACG_ERR_CUDA;
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecro, d_one, vecwo, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        }
        err = cudaEventRecord(wreadytosend, stream); if (err) return ACG_ERR_CUDA;

        /* q = Aw */
        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, wreadytosend, 0);
            if (err) return ACG_ERR_CUDA;
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
        }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecw, d_zero, vecq, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) return err;
            err = cudaEventRecord(wreceived, commstream);
            if (err) return ACG_ERR_CUDA;
            err = cudaStreamWaitEvent(stream, wreceived, 0);
            if (err) return ACG_ERR_CUDA;
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecwo, d_one, vecqo, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        }
    }

    /* warmup iterations for axpy */
    err = cudaMemcpy(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_z, 0, (cg->z->num_nonzeros-cg->z->num_ghost_nonzeros)*sizeof(*d_z)); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_t, 0, (cg->t.num_nonzeros-cg->t.num_ghost_nonzeros)*sizeof(*d_t)); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_p, 0, (cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*sizeof(*d_p)); if (err) return ACG_ERR_CUDA;
    for (int i = 0; i < warmup; i++) {
        err = cudaMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
        err = cudaMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
        err = cudaMemcpy(d_delta, d_inf, sizeof(*d_delta), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
        err = cudaMemcpy(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
        err = acgsolvercuda_pipelined_daxpy_fused(
            cg->t.num_nonzeros-cg->t.num_ghost_nonzeros,
            d_rnrm2sqr, d_rnrm2sqr_prev, d_delta,
            d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha, stream);
        if (err) return err;
    }
    err = cudaMemset(d_r, 0, (cg->r.num_nonzeros-cg->r.num_ghost_nonzeros)*sizeof(*d_r)); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_w, 0, (cg->w->num_nonzeros-cg->w->num_ghost_nonzeros)*sizeof(*d_w)); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_q, 0, (cg->q->num_nonzeros-cg->q->num_ghost_nonzeros)*sizeof(*d_q)); if (err) return ACG_ERR_CUDA;

    /* set scalars to infinity (needed to produce correct results on
     * the first call to acgsolvercuda_pipelined_daxpy_fused) */
    err = cudaMemcpy(d_alpha, d_inf, sizeof(*d_alpha), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), cudaMemcpyDeviceToDevice); if (err) return ACG_ERR_CUDA;

    /* set the vectors z, t and and p to zero */
    err = cudaMemset(d_z, 0, (cg->z->num_nonzeros-cg->z->num_ghost_nonzeros)*sizeof(*d_z)); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_t, 0, (cg->t.num_nonzeros-cg->t.num_ghost_nonzeros)*sizeof(*d_t)); if (err) return ACG_ERR_CUDA;
    err = cudaMemset(d_p, 0, (cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*sizeof(*d_p)); if (err) return ACG_ERR_CUDA;

    /* set initial state */
    bool converged = false;
    cg->nsolves++; cg->niterations = 0;
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
    if (err) return err;
    cudaStreamSynchronize(stream);
    gettime(&t0);

    /* compute right-hand side norm */
    double bnrm2sqr;
    acgEventRecord(tnrm2[2*nnrm2+0], 0);
    err = cublasDdot(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
    acgEventRecord(tnrm2[2*nnrm2+1], 0); nnrm2++; cg->nnrm2++;
    cg->nflops += 2*(b->num_nonzeros-b->num_ghost_nonzeros);
    cg->Bnrm2 += (b->num_nonzeros-b->num_ghost_nonzeros)*sizeof(*b->x);
    if (commsize > 1) {
        acgEventRecord(tallreduce[2*nallreduce+0], 0);
        err = acgcomm_allreduce(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
        cg->Ballreduce += sizeof(bnrm2sqr);
    }
    err = cudaMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;
    cg->bnrm2 = sqrt(bnrm2sqr);

    /* /\* compute norm of initial guess *\/ */
    /* if (diffatol > 0 || diffrtol > 0) { */
    /*     gettime(&tnrm20); */
    /*     double x0nrm2sqr; */
    /*     err = acgvector_dnrm2sqr(x, &x0nrm2sqr, &cg->nflops, &cg->Bnrm2); */
    /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; } */
    /*     gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21); */
    /*     gettime(&tallreduce0); */
    /*     err = MPI_Allreduce(MPI_IN_PLACE, &x0nrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm); */
    /*     if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *errcode = err; return ACG_ERR_MPI; } */
    /*     cg->Ballreduce += sizeof(x0nrm2sqr); */
    /*     gettime(&tallreduce1); cg->nallreduce++; cg->tallreduce += elapsed(tallreduce0,tallreduce1); */
    /*     cg->x0nrm2 = sqrt(x0nrm2sqr); */
    /*     diffrtol *= cg->x0nrm2; */
    /* } */

    /* compute initial residual, r₀ = b-A*x₀ */
    acgEventRecord(tcopy[2*ncopy+0], 0);
    err = cublasDcopy(cublas, b->num_nonzeros-b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
    acgEventRecord(tcopy[2*ncopy+1], 0); ncopy++; cg->ncopy++;
    cg->Bcopy += (b->num_nonzeros-b->num_ghost_nonzeros)*(sizeof(*cg->r.x)+sizeof(*b->x));

    if (commsize > 1) {
        err = acghalo_exchange_cuda_begin(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    }
    acgEventRecord(tgemv[2*ngemv+0], 0);
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    if (commsize > 1) {
        acgEventRecord(thalo[2*nhalo+0], 0);
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(thalo[2*nhalo+1], 0); nhalo++; cg->nhalo++;
        cg->Bhalo += cg->halo->sendsize*sizeof(*x->x);
        cg->nhalomsgs += cg->halo->nrecipients;
        err = cudaEventRecord(xreceived, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaStreamWaitEvent(stream, xreceived, 0);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        err = cudaEventRecord(rreadytosend, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
    }
    acgEventRecord(tgemv[2*ngemv+1], 0); ngemv++; cg->ngemv++;
    cg->nflops += 3*(int64_t)(A->fnpnzs+A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs+A->onpnzs)*(sizeof(*A->fa)+sizeof(*A->fcolidx))
        + A->nownedrows*(sizeof(*A->frowptr)+sizeof(*cg->r.x))
        + (A->nborderrows+A->nghostrows)*sizeof(*A->orowptr)
        + x->num_nonzeros*sizeof(*x->x);

    /* compute w = Ar */
    if (commsize > 1) {
        err = cudaStreamWaitEvent(commstream, rreadytosend, 0);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = acghalo_exchange_cuda_begin(
            cg->halo, cg->haloexchange,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    }
    acgEventRecord(tgemv[2*ngemv+0], 0);
    err = cusparseSpMV(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_one, matA, vecr, d_zero, vecw, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    if (commsize > 1) {
        acgEventRecord(thalo[2*nhalo+0], 0);
        err = acghalo_exchange_cuda_end(
            cg->halo, cg->haloexchange,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(thalo[2*nhalo+1], 0); nhalo++; cg->nhalo++;
        cg->Bhalo += cg->halo->sendsize*sizeof(*cg->r.x);
        cg->nhalomsgs += cg->halo->nrecipients;
        err = cudaEventRecord(rreceived, commstream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaStreamWaitEvent(stream, rreceived, 0);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matO, vecro, d_one, vecwo, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    }
    acgEventRecord(tgemv[2*ngemv+1], 0); ngemv++; cg->ngemv++;
    cg->nflops += 3*(int64_t)(A->fnpnzs+A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs+A->onpnzs)*(sizeof(*A->fa)+sizeof(*A->fcolidx))
        + A->nownedrows*(sizeof(*A->frowptr)+sizeof(*cg->w->x))
        + (A->nborderrows+A->nghostrows)*sizeof(*A->orowptr)
        + cg->r.num_nonzeros*sizeof(*cg->r.x);
    err = cudaEventRecord(wreadytosend, stream);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }

    /* iterative solver loop */
    for (int k = 0; k < maxits; k++) {

        /* compute residual norm (r,r) */
        acgEventRecord(tnrm2[2*nnrm2+0], 0);
        err = cublasDdot(cublas, cg->r.num_nonzeros-cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
        acgEventRecord(tnrm2[2*nnrm2+1], 0); nnrm2++; cg->nnrm2++;
        cg->nflops += 2*(cg->r.num_nonzeros-cg->r.num_ghost_nonzeros);
        cg->Bnrm2 += (cg->r.num_nonzeros-cg->r.num_ghost_nonzeros)*sizeof(*cg->r.x);

        /* compute (w,r) */
        acgEventRecord(tdot[2*ndot+0], 0);
        err = cublasDdot(cublas, cg->w->num_nonzeros-cg->w->num_ghost_nonzeros, d_w, 1, d_r, 1, d_delta);
        if (err) { if (errcode) *errcode = err; gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUBLAS; }
        acgEventRecord(tdot[2*ndot+1], 0); ndot++; cg->ndot++;
        cg->nflops += 2*(cg->w->num_nonzeros-cg->w->num_ghost_nonzeros);
        cg->Bdot += (cg->w->num_nonzeros-cg->w->num_ghost_nonzeros)*(sizeof(*cg->w->x)+sizeof(*cg->r.x));

        /* perform a single reduction for the two dot products */
        if (commsize > 1) {
            acgEventRecord(tallreduce[2*nallreduce+0], 0);
            err = acgcomm_allreduce(ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            acgEventRecord(tallreduce[2*nallreduce+1], 0); nallreduce++; cg->nallreduce++;
            cg->Ballreduce += 2*sizeof(*d_rnrm2sqr);
        }

        /* start copying residual norm from device to host,
         * overlapping it with the matrix-vector product */
        err = cudaEventRecord(rnrm2sqrready, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaStreamWaitEvent(copystream, rnrm2sqrready, 0);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        err = cudaMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost, copystream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }

        /* compute q = Aw */
        if (commsize > 1) {
            err = cudaStreamWaitEvent(commstream, wreadytosend, 0);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = acghalo_exchange_cuda_begin(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        }
        acgEventRecord(tgemv[2*ngemv+0], 0);
        err = cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecw, d_zero, vecq, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        if (commsize > 1) {
            acgEventRecord(thalo[2*nhalo+0], 0);
            err = acghalo_exchange_cuda_end(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            acgEventRecord(thalo[2*nhalo+1], 0); nhalo++; cg->nhalo++;
            cg->Bhalo += cg->halo->sendsize*sizeof(*cg->w->x);
            cg->nhalomsgs += cg->halo->nrecipients;
            err = cudaEventRecord(wreceived, commstream);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = cudaStreamWaitEvent(stream, wreceived, 0);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
            err = cusparseSpMV(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecwo, d_one, vecqo, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
        }
        acgEventRecord(tgemv[2*ngemv+1], 0); ngemv++; cg->ngemv++;
        cg->nflops += 3*(int64_t)(A->fnpnzs+A->onpnzs);
        cg->Bgemv +=
            (int64_t)(A->fnpnzs+A->onpnzs)*(sizeof(*A->fa)+sizeof(*A->fcolidx))
            + A->nownedrows*(sizeof(*A->frowptr)+sizeof(*cg->q->x))
            + (A->nborderrows+A->nghostrows)*sizeof(*A->orowptr)
            + cg->w->num_nonzeros*sizeof(*cg->w->x);

        /* wait for host to receive updated residual norm */
        cudaStreamSynchronize(copystream);
        cg->rnrm2 = sqrt(*rnrm2sqr);
        if (k == 0) { cg->r0nrm2 = cg->rnrm2; residualrtol *= cg->r0nrm2; }

        /* convergence tests */
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) ||
            (diffrtol > 0 && cg->dxnrm2 < diffrtol) ||
            (residualatol > 0 && cg->rnrm2 < residualatol) ||
            (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            cudaStreamSynchronize(stream);
            converged = true;
            break;
        }

        /* update vectors */
        acgEventRecord(taxpy[2*naxpy+0], 0);
        err = acgsolvercuda_pipelined_daxpy_fused(
            cg->t.num_nonzeros-cg->t.num_ghost_nonzeros,
            d_rnrm2sqr, d_rnrm2sqr_prev, d_delta,
            d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        acgEventRecord(taxpy[2*naxpy+1], 0);
        naxpy++; cg->naxpy++;
        cg->nflops += 12*(cg->t.num_nonzeros-cg->t.num_ghost_nonzeros);
        cg->Baxpy += 7*(cg->p.num_nonzeros-cg->p.num_ghost_nonzeros)*sizeof(*cg->p.x);
        err = cudaEventRecord(wreadytosend, stream);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_CUDA; }
        cg->ntotaliterations++; cg->niterations++;
    }
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

#if defined(ACG_ENABLE_PROFILING)
    /* record profiling information */
    float t;
    for (acgidx_t i = 0; i < ngemv; i++) {
        cudaEventSynchronize(tgemv[2*i+1]);
        cudaEventElapsedTime(&t, tgemv[2*i+0], tgemv[2*i+1]);
        cg->tgemv += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < ndot; i++) {
        cudaEventSynchronize(tdot[2*i+1]);
        cudaEventElapsedTime(&t, tdot[2*i+0], tdot[2*i+1]);
        cg->tdot += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < nnrm2; i++) {
        cudaEventSynchronize(tnrm2[2*i+1]);
        cudaEventElapsedTime(&t, tnrm2[2*i+0], tnrm2[2*i+1]);
        cg->tnrm2 += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < naxpy; i++) {
        cudaEventSynchronize(taxpy[2*i+1]);
        cudaEventElapsedTime(&t, taxpy[2*i+0], taxpy[2*i+1]);
        cg->taxpy += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < ncopy; i++) {
        cudaEventSynchronize(tcopy[2*i+1]);
        cudaEventElapsedTime(&t, tcopy[2*i+0], tcopy[2*i+1]);
        cg->tcopy += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < nallreduce; i++) {
        cudaEventSynchronize(tallreduce[2*i+1]);
        cudaEventElapsedTime(&t, tallreduce[2*i+0], tallreduce[2*i+1]);
        cg->tallreduce += 1.0e-3*t;
    }
    for (acgidx_t i = 0; i < nhalo; i++) {
        cudaEventSynchronize(thalo[2*i+1]);
        cudaEventElapsedTime(&t, thalo[2*i+0], thalo[2*i+1]);
        cg->thalo += 1.0e-3*t;
    }
#endif

    /* copy solution back to host */
    err = cudaMemcpy(x->x, d_x, x->num_nonzeros*sizeof(*d_x), cudaMemcpyDeviceToHost);
    if (err) return ACG_ERR_CUDA;

    /* free cusparse matrix and vectors */
    cusparseDestroyDnVec(vecx);
    cusparseDestroyDnVec(vecr);
    cusparseDestroyDnVec(vecw);
    cusparseDestroyDnVec(vecq);
    if (commsize > 1) {
        cusparseDestroyDnVec(vecxo);
        cusparseDestroyDnVec(vecro);
        cusparseDestroyDnVec(vecwo);
        cusparseDestroyDnVec(vecqo);
    }
    cusparseDestroySpMat(matA);
    cudaFree(d_buffer);
    if (commsize > 1) {
        cusparseDestroySpMat(matO);
        cudaFree(d_obuffer);
    }
    cudaFree(d_x); cudaFree(d_b);
    cudaFreeHost(rnrm2sqr);
    cudaStreamDestroy(commstream);
    cudaStreamDestroy(copystream);

    /* reset cusparse and cublas pointer modes */
    err = cusparseSetPointerMode(cusparse, cusparsepointermode);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUSPARSE; }
    err = cublasSetPointerMode(cublas, cublaspointermode);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_CUBLAS; }

    /* check for CUDA errors */
    if (cudaGetLastError() != cudaSuccess)
        return ACG_ERR_CUDA;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged) return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
}
#endif

/*
 * output solver info
 */

static void findent(FILE * f, int indent) { fprintf(f, "%*c", indent, ' '); }

/**
 * ‘acgsolvercuda_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolvercuda_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolvercuda_fwrite(
    FILE * f,
    const struct acgsolvercuda * cg,
    int indent)
{
    double tother = cg->tsolve -
        (cg->tgemv+cg->tdot+cg->tnrm2+cg->taxpy+cg->tcopy
         +cg->tallreduce+cg->thalo);
    findent(f,indent); fprintf(f, "unknowns: %'"PRIdx"\n", cg->p.size);
    findent(f,indent); fprintf(f, "solves: %'d\n", cg->nsolves);
    findent(f,indent); fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
    findent(f,indent); fprintf(f, "total flops: %'.3f Gflop\n", 1.0e-9*cg->nflops);
    findent(f,indent); fprintf(f, "total flop rate: %'.3f Gflop/s\n", cg->tsolve > 0 ? 1.0e-9*cg->nflops/cg->tsolve : 0);
    findent(f,indent); fprintf(f, "total solver time: %'.6f seconds\n", cg->tsolve);
    findent(f,indent); fprintf(f, "performance breakdown:\n");
    findent(f,indent); fprintf(f, "  gemv: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->tgemv, cg->ngemv, cg->Bgemv, cg->tgemv>0 ? 1.0e-9*cg->Bgemv/cg->tgemv : 0.0);
    findent(f,indent); fprintf(f, "  dot: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->tdot, cg->ndot, cg->Bdot, cg->tdot>0 ? 1.0e-9*cg->Bdot/cg->tdot : 0.0);
    findent(f,indent); fprintf(f, "  nrm2: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->tnrm2, cg->nnrm2, cg->Bnrm2, cg->tnrm2>0 ? 1.0e-9*cg->Bnrm2/cg->tnrm2 : 0.0);
    findent(f,indent); fprintf(f, "  axpy: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->taxpy, cg->naxpy, cg->Baxpy, cg->taxpy>0 ? 1.0e-9*cg->Baxpy/cg->taxpy : 0.0);
    findent(f,indent); fprintf(f, "  copy: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->tcopy, cg->ncopy, cg->Bcopy, cg->tcopy>0 ? 1.0e-9*cg->Bcopy/cg->tcopy : 0.0);
    findent(f,indent); fprintf(f, "  MPI_Allreduce: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->tallreduce, cg->nallreduce, cg->Ballreduce, cg->tallreduce>0.0 ? 1.0e-9*cg->Ballreduce/cg->tallreduce : 0.0);
    findent(f,indent); fprintf(f, "  MPI_HaloExchange: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->thalo, cg->nhalo, cg->Bhalo, cg->thalo>0.0 ? 1.0e-9*cg->Bhalo/cg->thalo : 0.0);
    findent(f,indent); fprintf(f, "  other: %'.6f seconds\n", tother);
    findent(f,indent); fprintf(f, "last solve:\n");
    findent(f,indent); fprintf(f, "  stopping criterion:\n");
    findent(f,indent); fprintf(f, "    maximum iterations: %'d\n", cg->maxits);
    findent(f,indent); fprintf(f, "    tolerance for residual: %.*g\n", DBL_DIG, cg->residualatol);
    findent(f,indent); fprintf(f, "    tolerance for relative residual: %.*g\n", DBL_DIG, cg->residualrtol);
    findent(f,indent); fprintf(f, "    tolerance for difference in solution iterates: %.*g\n", DBL_DIG, cg->diffatol);
    findent(f,indent); fprintf(f, "    tolerance for relative difference in solution iterates: %.*g\n", DBL_DIG, cg->diffrtol);
    findent(f,indent); fprintf(f, "  iterations: %'d\n", cg->niterations);
    findent(f,indent); fprintf(f, "  right-hand side 2-norm: %.*g\n", DBL_DIG, cg->bnrm2);
    findent(f,indent); fprintf(f, "  initial guess 2-norm: %.*g\n", DBL_DIG, cg->x0nrm2);
    findent(f,indent); fprintf(f, "  initial residual 2-norm: %.*g\n", DBL_DIG, cg->r0nrm2);
    findent(f,indent); fprintf(f, "  residual 2-norm: %.*g\n", DBL_DIG, cg->rnrm2);
    findent(f,indent); fprintf(f, "  difference in solution iterates 2-norm: %.*g\n", DBL_DIG, cg->dxnrm2);
    findent(f,indent); fprintf(f, "  floating-point exceptions: %s\n", acgerrcodestr(ACG_ERR_FEXCEPT, 0));
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
    FILE * f,
    const struct acgsolvercuda * cg,
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
    double tdot = cg->tdot;
    double tnrm2 = cg->tnrm2;
    double taxpy = cg->taxpy;
    double tcopy = cg->tcopy;
    double tallreduce = cg->tallreduce;
    double thalo = cg->thalo;
    double tother = tsolve-(tgemv+tdot+tnrm2+taxpy+tcopy+tallreduce+thalo);
    int64_t ngemv = cg->ngemv, Bgemv = cg->Bgemv;
    int64_t ndot = cg->ndot, Bdot = cg->Bdot;
    int64_t nnrm2 = cg->nnrm2, Bnrm2 = cg->Bnrm2;
    int64_t naxpy = cg->naxpy, Baxpy = cg->Baxpy;
    int64_t ncopy = cg->ncopy, Bcopy = cg->Bcopy;
    int64_t nallreduce = cg->nallreduce, Ballreduce = cg->Ballreduce;
    int64_t nhalo = cg->nhalo, Bhalo = cg->Bhalo;
    int64_t nhalopack = cg->halo->npack, Bhalopack = cg->halo->Bpack;
    int64_t nhalounpack = cg->halo->nunpack, Bhalounpack = cg->halo->Bunpack;
    int64_t nhalompiirecv = cg->halo->nmpiirecv, Bhalompiirecv = cg->halo->Bmpiirecv;
    int64_t nhalompisend = cg->halo->nmpisend, Bhalompisend = cg->halo->Bmpisend;
    int64_t nhalomsgs = cg->nhalomsgs;
    MPI_Reduce(&cg->nflops, &nflops, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->tsolve, &tsolve, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(&cg->tgemv, &tgemv, 1, MPI_DOUBLE, MPI_SUM, root, comm); tgemv /= commsize;
    MPI_Reduce(&cg->tdot, &tdot, 1, MPI_DOUBLE, MPI_SUM, root, comm); tdot /= commsize;
    MPI_Reduce(&cg->tnrm2, &tnrm2, 1, MPI_DOUBLE, MPI_SUM, root, comm); tnrm2 /= commsize;
    MPI_Reduce(&cg->taxpy, &taxpy, 1, MPI_DOUBLE, MPI_SUM, root, comm); taxpy /= commsize;
    MPI_Reduce(&cg->tcopy, &tcopy, 1, MPI_DOUBLE, MPI_SUM, root, comm); tcopy /= commsize;
    MPI_Reduce(&cg->tallreduce, &tallreduce, 1, MPI_DOUBLE, MPI_SUM, root, comm); tallreduce /= commsize;
    MPI_Reduce(&cg->thalo, &thalo, 1, MPI_DOUBLE, MPI_SUM, root, comm); thalo /= commsize;
    MPI_Reduce(rank == root ? MPI_IN_PLACE : &tother, &tother, 1, MPI_DOUBLE, MPI_SUM, root, comm); tother /= commsize;
    MPI_Reduce(&cg->ngemv, &ngemv, 1, MPI_INT64_T, MPI_SUM, root, comm); ngemv /= commsize;
    MPI_Reduce(&cg->ndot, &ndot, 1, MPI_INT64_T, MPI_SUM, root, comm); ndot /= commsize;
    MPI_Reduce(&cg->nnrm2, &nnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm); nnrm2 /= commsize;
    MPI_Reduce(&cg->naxpy, &naxpy, 1, MPI_INT64_T, MPI_SUM, root, comm); naxpy /= commsize;
    MPI_Reduce(&cg->ncopy, &ncopy, 1, MPI_INT64_T, MPI_SUM, root, comm); ncopy /= commsize;
    MPI_Reduce(&cg->nallreduce, &nallreduce, 1, MPI_INT64_T, MPI_SUM, root, comm); nallreduce /= commsize;
    MPI_Reduce(&cg->halo->npack, &nhalopack, 1, MPI_INT64_T, MPI_SUM, root, comm); nhalopack /= commsize;
    MPI_Reduce(&cg->halo->nunpack, &nhalounpack, 1, MPI_INT64_T, MPI_SUM, root, comm); nhalounpack /= commsize;
    MPI_Reduce(&cg->halo->nmpiirecv, &nhalompiirecv, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->halo->nmpisend, &nhalompisend, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->Bgemv, &Bgemv, 1, MPI_INT64_T, MPI_SUM, root, comm); Bgemv /= commsize;
    MPI_Reduce(&cg->Bdot, &Bdot, 1, MPI_INT64_T, MPI_SUM, root, comm); Bdot /= commsize;
    MPI_Reduce(&cg->Bnrm2, &Bnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm); Bnrm2 /= commsize;
    MPI_Reduce(&cg->Baxpy, &Baxpy, 1, MPI_INT64_T, MPI_SUM, root, comm); Baxpy /= commsize;
    MPI_Reduce(&cg->Bcopy, &Bcopy, 1, MPI_INT64_T, MPI_SUM, root, comm); Bcopy /= commsize;
    MPI_Reduce(&cg->Ballreduce, &Ballreduce, 1, MPI_INT64_T, MPI_SUM, root, comm); Ballreduce /= commsize;
    MPI_Reduce(&cg->Bhalo, &Bhalo, 1, MPI_INT64_T, MPI_SUM, root, comm); Bhalo /= commsize;
    MPI_Reduce(&cg->nhalomsgs, &nhalomsgs, 1, MPI_INT64_T, MPI_SUM, root, comm);
    if (rank == root) {
        findent(f,indent); fprintf(f, "unknowns: %'"PRIdx"\n", cg->p.size);
        findent(f,indent); fprintf(f, "solves: %'d\n", cg->nsolves);
        findent(f,indent); fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
        findent(f,indent); fprintf(f, "total flops: %'.3f Gflop\n", 1.0e-9*nflops);
        findent(f,indent); fprintf(f, "total flop rate: %'.3f Gflop/s\n", tsolve > 0 ? 1.0e-9*nflops/tsolve : 0);
        findent(f,indent); fprintf(f, "total solver time: %'.6f seconds\n", tsolve);
        findent(f,indent); fprintf(f, "performance breakdown:\n");
        findent(f,indent); fprintf(f, "  gemv: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tgemv, ngemv, Bgemv, tgemv>0.0 ? 1.0e-9*Bgemv/tgemv : 0.0);
        findent(f,indent); fprintf(f, "  dot: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tdot, ndot, Bdot, tdot>0.0 ? 1.0e-9*Bdot/tdot : 0.0);
        findent(f,indent); fprintf(f, "  nrm2: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tnrm2, nnrm2, Bnrm2, tnrm2>0.0 ? 1.0e-9*Bnrm2/tnrm2 : 0.0);
        findent(f,indent); fprintf(f, "  axpy: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   taxpy, naxpy, Baxpy, taxpy>0.0 ? 1.0e-9*Baxpy/taxpy : 0.0);
        findent(f,indent); fprintf(f, "  copy: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tcopy, ncopy, Bcopy, tcopy>0.0 ? 1.0e-9*Bcopy/tcopy : 0.0);
        findent(f,indent); fprintf(f, "  allreduce: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc %'.3f us/op/proc\n",
                                   tallreduce, nallreduce, Ballreduce, tallreduce>0.0 ? 1.0e-9*Ballreduce/tallreduce : 0.0,
                                   nallreduce>0.0 ? 1.0e6*tallreduce/nallreduce : 0.0);
        findent(f,indent); fprintf(f, "  haloexchange: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc %'.1f msg/proc %'.3f us/msg/proc\n",
                                   thalo, nhalo, Bhalo, thalo>0.0 ? 1.0e-9*Bhalo/thalo : 0.0,
                                   ((double) nhalomsgs)/commsize, nhalomsgs>0.0 ? 1.0e6*thalo/nhalomsgs/commsize : 0.0);
    }

    int * pnrecipients = rank == root ? malloc(commsize*sizeof(*pnrecipients)) : NULL;
    MPI_Gather(&cg->halo->nrecipients, 1, MPI_INT, pnrecipients, 1, MPI_INT, root, comm);
    int * pnsenders = rank == root ? malloc(commsize*sizeof(*pnsenders)) : NULL;
    MPI_Gather(&cg->halo->nsenders, 1, MPI_INT, pnsenders, 1, MPI_INT, root, comm);
    int * psendsize = rank == root ? malloc(commsize*sizeof(*psendsize)) : NULL;
    MPI_Gather(&cg->halo->sendsize, 1, MPI_INT, psendsize, 1, MPI_INT, root, comm);
    int * precvsize = rank == root ? malloc(commsize*sizeof(*precvsize)) : NULL;
    MPI_Gather(&cg->halo->recvsize, 1, MPI_INT, precvsize, 1, MPI_INT, root, comm);
    int * pmaxsendcount = rank == root ? malloc(commsize*sizeof(*pmaxsendcount)) : NULL;
    int maxsendcount = 0;
    for (int q = 0; q < cg->halo->nrecipients; q++) maxsendcount = maxsendcount > cg->halo->sendcounts[q] ? maxsendcount : cg->halo->sendcounts[q];
    MPI_Gather(&maxsendcount, 1, MPI_INT, pmaxsendcount, 1, MPI_INT, root, comm);
    int * pmaxrecvcount = rank == root ? malloc(commsize*sizeof(*pmaxrecvcount)) : NULL;
    int maxrecvcount = 0;
    for (int q = 0; q < cg->halo->nsenders; q++) maxrecvcount = maxrecvcount > cg->halo->recvcounts[q] ? maxrecvcount : cg->halo->recvcounts[q];
    MPI_Gather(&maxrecvcount, 1, MPI_INT, pmaxrecvcount, 1, MPI_INT, root, comm);
    if (rank == root) {
        for (int p = 0; p < commsize; p++) {
            findent(f,indent);
            fprintf(f, "    rank %'2d sends %'"PRId64" B %'zu B/it in %'"PRId64" msg %'d msg/it max %'zu B/msg\n",
                    p, (int64_t)nhalo*psendsize[p]*sizeof(double),
                    psendsize[p]*sizeof(double),
                    nhalo*pnrecipients[p], pnrecipients[p],
                    pmaxsendcount[p]*sizeof(double));
            findent(f,indent);
            fprintf(f, "    rank %'2d receives %'"PRId64" B %'zu B/it in %'"PRId64" msg %'d msg/it max %'zu B/msg\n",
                    p, (int64_t)nhalo*precvsize[p]*sizeof(double),
                    precvsize[p]*sizeof(double),
                    nhalo*pnrecipients[p], pnrecipients[p],
                    pmaxrecvcount[p]*sizeof(double));
        }
    }

    const struct acghaloexchange * haloexchange = cg->haloexchange;
    int maxevents = haloexchange->maxevents, nevents = 0;
    double * texchange = malloc(maxevents*sizeof(*texchange));
    double * tpack = malloc(maxevents*sizeof(*tpack));
    double * tsendrecv = malloc(maxevents*sizeof(*tsendrecv));
    double * tunpack = malloc(maxevents*sizeof(*tunpack));
    int err = acghaloexchange_profile(
        haloexchange, maxevents, &nevents,
        texchange, tpack, tsendrecv, tunpack);
    if (err) return err;
    double * ptexchange = rank == root ? malloc(commsize*sizeof(*ptexchange)) : NULL;
    double * ptpack = rank == root ? malloc(commsize*sizeof(*ptpack)) : NULL;
    double * ptsendrecv = rank == root ? malloc(commsize*sizeof(*ptsendrecv)) : NULL;
    double * ptunpack = rank == root ? malloc(commsize*sizeof(*ptunpack)) : NULL;

    /* sum and mean over all iterations per rank */
    double texchangesum = 0.0, tpacksum = 0.0, tsendrecvsum = 0.0, tunpacksum = 0.0;
    for (int i = 0; i < nevents; i++) {
        texchangesum += texchange[i];
        tpacksum += tpack[i];
        tsendrecvsum += tsendrecv[i];
        tunpacksum += tunpack[i];
    }
    MPI_Gather(&texchangesum, 1, MPI_DOUBLE, ptexchange, 1, MPI_DOUBLE, root, comm);
    MPI_Gather(&tpacksum, 1, MPI_DOUBLE, ptpack, 1, MPI_DOUBLE, root, comm);
    MPI_Gather(&tsendrecvsum, 1, MPI_DOUBLE, ptsendrecv, 1, MPI_DOUBLE, root, comm);
    MPI_Gather(&tunpacksum, 1, MPI_DOUBLE, ptunpack, 1, MPI_DOUBLE, root, comm);

    int sendsizeavg = 0, recvsizeavg = 0;
    if (rank == root) {
        for (int p = 0; p < commsize; p++) {
            sendsizeavg += psendsize[p];
            recvsizeavg += precvsize[p];
        }
        sendsizeavg /= commsize; recvsizeavg /= commsize;
        double texchangeavg = 0.0, tpackavg = 0.0, tsendrecvavg = 0.0, tunpackavg = 0.0;
        for (int p = 0; p < commsize; p++) {
            texchangeavg += ptexchange[p];
            tpackavg += ptpack[p];
            tsendrecvavg += ptsendrecv[p];
            tunpackavg += ptunpack[p];
        }
        texchangeavg /= (double) commsize;
        tpackavg /= (double) commsize;
        tsendrecvavg /= (double) commsize;
        tunpackavg /= (double) commsize;

        findent(f,indent); fprintf(f, "    summary of %'d most recent iterations per rank:\n", nevents);
        if (nevents > 0) {
            fprintf(f, "      mean of %'d ranks:"
                    " %'.6f s %'.6f s/it total"
                    " %'.6f s %'.6f s/it %'5.2f GB/s send %'5.2f GB/s recv"
                    " %'.6f s %'.6f s/it %'5.2f GB/s pack"
                    " %'.6f s %'.6f s/it %'5.2f GB/s unpack\n",
                    commsize, texchangeavg, texchangeavg/(double)nevents,
                    tsendrecvavg, tsendrecvavg/(double)nevents,
                    nevents*sendsizeavg*sizeof(double)*1.0e-9/tsendrecvavg,
                    nevents*recvsizeavg*sizeof(double)*1.0e-9/tsendrecvavg,
                    tpackavg, tpackavg/(double)nevents, nevents*sendsizeavg*(2*sizeof(double)+sizeof(int))*1.0e-9/tpackavg,
                    tunpackavg, tunpackavg/(double)nevents, nevents*recvsizeavg*(2*sizeof(double)+sizeof(int))*1.0e-9/tunpackavg);
            for (int p = 0; p < commsize; p++) {
                findent(f,indent);
                fprintf(f, "      rank %'2d:"
                        " %'.6f s %'.6f s/it total"
                        " %'.6f s %'.6f s/it %'5.2f GB/s send %'5.2f GB/s recv"
                        " %'.6f s %'.6f s/it %'5.2f GB/s pack"
                        " %'.6f s %'.6f s/it %'5.2f GB/s unpack\n",
                        p, ptexchange[p], ptexchange[p]/(double)nevents,
                        ptsendrecv[p], ptsendrecv[p]/(double)nevents,
                        nevents*psendsize[p]*sizeof(double)*1.0e-9/ptsendrecv[p],
                        nevents*precvsize[p]*sizeof(double)*1.0e-9/ptsendrecv[p],
                        ptpack[p], ptpack[p]/(double)nevents, nevents*psendsize[p]*(2*sizeof(double)+sizeof(int))*1.0e-9/ptpack[p],
                        ptunpack[p], ptunpack[p]/(double)nevents, nevents*precvsize[p]*(2*sizeof(double)+sizeof(int))*1.0e-9/ptunpack[p]);
            }
        }
    }

    if (verbose > 0) {
        double critpath = 0.0;
        for (int i = 0; i < nevents; i++) {
            MPI_Gather(&texchange[i], 1, MPI_DOUBLE, ptexchange, 1, MPI_DOUBLE, root, comm);
            MPI_Gather(&tpack[i], 1, MPI_DOUBLE, ptpack, 1, MPI_DOUBLE, root, comm);
            MPI_Gather(&tsendrecv[i], 1, MPI_DOUBLE, ptsendrecv, 1, MPI_DOUBLE, root, comm);
            MPI_Gather(&tunpack[i], 1, MPI_DOUBLE, ptunpack, 1, MPI_DOUBLE, root, comm);
            if (rank == root) {
                double texchangeavg = 0.0, tpackavg = 0.0, tsendrecvavg = 0.0, tunpackavg = 0.0;
                double texchangemax = 0.0;
                for (int p = 0; p < commsize; p++) {
                    texchangeavg += ptexchange[p];
                    tpackavg += ptpack[p];
                    tsendrecvavg += ptsendrecv[p];
                    tunpackavg += ptunpack[p];
                    texchangemax = texchangemax > ptexchange[p] ? texchangemax : ptexchange[p];
                }
                texchangeavg /= (double) commsize;
                tpackavg /= (double) commsize;
                tsendrecvavg /= (double) commsize;
                tunpackavg /= (double) commsize;
                critpath += texchangemax;

                findent(f,indent); fprintf(f, "    iteration %'4d:\n", cg->halo->nexchanges-i-1);
                findent(f,indent); fprintf(
                    f, "      mean of %'2d ranks: %'.6f s total %'.6f s %'5.2f GB/s send %'5.2f GB/s recv %'.6f s %'5.2f GB/s pack %'.6f s %'5.2f GB/s unpack\n",
                    commsize, texchangeavg, tsendrecvavg,
                    sendsizeavg*sizeof(double)*1.0e-9/tsendrecvavg,
                    recvsizeavg*sizeof(double)*1.0e-9/tsendrecvavg,
                    tpackavg, sendsizeavg*(2*sizeof(double)+sizeof(int))*1.0e-9/tpackavg,
                    tunpackavg, recvsizeavg*(2*sizeof(double)+sizeof(int))*1.0e-9/tunpackavg);
                for (int p = 0; p < commsize; p++) {
                    findent(f,indent);
                    fprintf(f, "      rank %'2d: %'.6f s total %'.6f s %'5.2f GB/s send %'5.2f GB/s recv %'.6f s %'5.2f GB/s pack %'.6f s %'5.2f GB/s unpack\n",
                            p, ptexchange[p], ptsendrecv[p],
                            psendsize[p]*sizeof(double)*1.0e-9/ptsendrecv[p],
                            precvsize[p]*sizeof(double)*1.0e-9/ptsendrecv[p],
                            ptpack[p], psendsize[p]*(2*sizeof(double)+sizeof(int))*1.0e-9/ptpack[p],
                            ptunpack[p], precvsize[p]*(2*sizeof(double)+sizeof(int))*1.0e-9/ptunpack[p]);
                }
            }
        }
        if (rank == root) fprintf(f, "      critical path: %'.6f s\n", critpath);
    }
    free(ptunpack); free(ptsendrecv); free(ptpack); free(ptexchange);
    free(tunpack); free(tsendrecv); free(tpack); free(texchange);
    if (rank == root) {
        free(pmaxsendcount); free(pmaxrecvcount);
        free(pnrecipients); free(pnsenders);
        free(precvsize); free(psendsize);
    }

    if (rank == root) {
        findent(f,indent); fprintf(f, "  other: %'.6f seconds\n", tother);
        findent(f,indent); fprintf(f, "last solve:\n");
        findent(f,indent); fprintf(f, "  stopping criterion:\n");
        findent(f,indent); fprintf(f, "    maximum iterations: %'d\n", cg->maxits);
        findent(f,indent); fprintf(f, "    tolerance for residual: %.*g\n", DBL_DIG, cg->residualatol);
        findent(f,indent); fprintf(f, "    tolerance for relative residual: %.*g\n", DBL_DIG, cg->residualrtol);
        findent(f,indent); fprintf(f, "    tolerance for difference in solution iterates: %.*g\n", DBL_DIG, cg->diffatol);
        findent(f,indent); fprintf(f, "    tolerance for relative difference in solution iterates: %.*g\n", DBL_DIG, cg->diffrtol);
        findent(f,indent); fprintf(f, "  iterations: %'d\n", cg->niterations);
        findent(f,indent); fprintf(f, "  right-hand side 2-norm: %.*g\n", DBL_DIG, cg->bnrm2);
        findent(f,indent); fprintf(f, "  initial guess 2-norm: %.*g\n", DBL_DIG, cg->x0nrm2);
        findent(f,indent); fprintf(f, "  initial residual 2-norm: %.*g\n", DBL_DIG, cg->r0nrm2);
        findent(f,indent); fprintf(f, "  residual 2-norm: %.*g\n", DBL_DIG, cg->rnrm2);
        findent(f,indent); fprintf(f, "  difference in solution iterates 2-norm: %.*g\n", DBL_DIG, cg->dxnrm2);
        findent(f,indent); fprintf(f, "  floating-point exceptions: %s\n", acgerrcodestr(ACG_ERR_FEXCEPT, 0));
    }
    return ACG_SUCCESS;
}
#endif
