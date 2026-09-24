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
 * conjugate gradient (CG) solver using HIP
 */

#include "acg/config.h"
#include "acg/cghip.h"
#include "acg/cg-kernels-hip.h"
#include "acg/comm.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/error.h"
#include "acg/time.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif
#ifdef ACG_HAVE_HIPBLAS
#include <hipblas/hipblas.h>
#endif
#ifdef ACG_HAVE_HIPSPARSE
#include <hipsparse/hipsparse.h>
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

// #define ACG_USE_HIPSPARSE

/* #define ACG_ENABLE_PROFILING 1 */

#ifdef ACG_ENABLE_PROFILING
#define acgEventRecord(event, stream) hipEventRecord((event), (stream))
#else
#define acgEventRecord(event, stream)
#endif

/*
 * memory management
 */

/**
 * ‘acgsolverhip_free()’ frees storage allocated for a solver.
 */
void acgsolverhip_free(
    struct acgsolverhip *cg)
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
    if (!cg->use_rocshmem)
    {
        hipFree(cg->d_bnrm2sqr);
        hipFree(cg->d_r0nrm2sqr);
        hipFree(cg->d_rnrm2sqr);
        hipFree(cg->d_pdott);
    }
    else
    {
        acgcomm_rocshmem_free(cg->d_bnrm2sqr);
        acgcomm_rocshmem_free(cg->d_r0nrm2sqr);
        acgcomm_rocshmem_free(cg->d_rnrm2sqr);
        acgcomm_rocshmem_free(cg->d_pdott);
    }
    hipFree(cg->d_rnrm2sqr_prev);
    hipFree(cg->d_alpha);
    hipFree(cg->d_minus_alpha);
    hipFree(cg->d_beta);
    hipFree(cg->d_niterations);
    hipFree(cg->d_converged);
    hipFree(cg->d_r);
    hipFree(cg->d_p);
    hipFree(cg->d_t);
    if (cg->d_w)
        hipFree(cg->d_w);
    if (cg->d_q)
        hipFree(cg->d_q);
    if (cg->d_z)
        hipFree(cg->d_z);
    hipFree(cg->d_rowptr);
    hipFree(cg->d_colidx);
    hipFree(cg->d_a);
    hipFree(cg->d_orowptr);
    hipFree(cg->d_ocolidx);
    hipFree(cg->d_oa);
}

/*
 * initialise a solver
 */

#if defined(ACG_HAVE_HIPBLAS) && defined(ACG_HAVE_HIPSPARSE)
/**
 * ‘acgsolverhip_init()’ sets up a conjugate gradient solver for a given
 * sparse matrix in CSR format.
 *
 * The matrix may be partitioned and distributed.
 */
int acgsolverhip_init(
    struct acgsolverhip *cg,
    const struct acgsymcsrmatrix *A,
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse,
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
    cg->w = cg->q = cg->z = NULL;
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
    hipStream_t stream = 0;
    err = acghaloexchange_init_hip(
        cg->haloexchange, cg->halo,
        ACG_DOUBLE, ACG_DOUBLE, comm, stream);
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
    /* err = acgsolverhip_init_constants( */
    /*     &cg->d_minus_one, &cg->d_one, &cg->d_zero); */
    /* if (err) return err; */

    double one = 1.0, minus_one = -1.0, zero = 0.0, inf = INFINITY;
    err = hipMalloc((void **)&cg->d_one, sizeof(*cg->d_one));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_one, &one, sizeof(*cg->d_one), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_minus_one, sizeof(*cg->d_minus_one));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_minus_one, &minus_one, sizeof(*cg->d_minus_one), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_zero, sizeof(*cg->d_zero));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_zero, &zero, sizeof(*cg->d_zero), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_inf, sizeof(*cg->d_inf));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_inf, &inf, sizeof(*cg->d_inf), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;

    cg->use_rocshmem = comm->type == acgcomm_rocshmem;
    cg->use_rccl_split = comm->type == acgcomm_rccl_split;
    if (!cg->use_rocshmem)
    {
        err = hipMalloc((void **)&cg->d_bnrm2sqr, sizeof(*cg->d_bnrm2sqr));
        if (err)
            return ACG_ERR_HIP;
        err = hipMalloc((void **)&cg->d_r0nrm2sqr, sizeof(*cg->d_r0nrm2sqr));
        if (err)
            return ACG_ERR_HIP;
        err = hipMalloc((void **)&cg->d_rnrm2sqr, 2 * sizeof(*cg->d_rnrm2sqr));
        if (err)
            return ACG_ERR_HIP;
        err = hipMalloc((void **)&cg->d_pdott, sizeof(*cg->d_pdott));
        if (err)
            return ACG_ERR_HIP;
    }
    else
    {
        int errcode;
        err = acgcomm_rocshmem_malloc((void **)&cg->d_bnrm2sqr, sizeof(*cg->d_bnrm2sqr), &errcode);
        if (err)
            return err;
        err = acgcomm_rocshmem_malloc((void **)&cg->d_r0nrm2sqr, sizeof(*cg->d_r0nrm2sqr), &errcode);
        if (err)
            return err;
        err = acgcomm_rocshmem_malloc((void **)&cg->d_rnrm2sqr, 2 * sizeof(*cg->d_rnrm2sqr), &errcode);
        if (err)
            return err;
        err = acgcomm_rocshmem_malloc((void **)&cg->d_pdott, sizeof(*cg->d_pdott), &errcode);
        if (err)
            return err;
    }
    err = hipMalloc((void **)&cg->d_rnrm2sqr_prev, sizeof(*cg->d_rnrm2sqr_prev));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_niterations, sizeof(*cg->d_niterations));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_converged, sizeof(*cg->d_converged));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_alpha, sizeof(*cg->d_alpha));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_minus_alpha, sizeof(*cg->d_minus_alpha));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_beta, sizeof(*cg->d_beta));
    if (err)
        return ACG_ERR_HIP;

    /* allocate storage for auxiliary vectors on device */
    err = hipMalloc((void **)&cg->d_r, cg->r.num_nonzeros * sizeof(*cg->d_r));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_p, cg->p.num_nonzeros * sizeof(*cg->d_p));
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_t, cg->t.num_nonzeros * sizeof(*cg->d_t));
    if (err)
        return ACG_ERR_HIP;
    cg->d_w = cg->d_q = cg->d_z = NULL;

    /* copy sparse matrix to device */
    err = hipMalloc((void **)&cg->d_rowptr, (A->nprows + 1) * sizeof(*cg->d_rowptr));
    if (err)
        return ACG_ERR_HIP;
    if (sizeof(*cg->d_rowptr) == sizeof(*A->frowptr))
    {
        err = hipMemcpy(cg->d_rowptr, A->frowptr, (A->nprows + 1) * sizeof(*cg->d_rowptr), hipMemcpyHostToDevice);
        if (err)
            return ACG_ERR_HIP;
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
        err = hipMemcpy(cg->d_rowptr, tmprowptr, (A->nprows + 1) * sizeof(*cg->d_rowptr), hipMemcpyHostToDevice);
        if (err)
            return ACG_ERR_HIP;
        free(tmprowptr);
    }
    err = hipMalloc((void **)&cg->d_colidx, A->fnpnzs * sizeof(*cg->d_colidx));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_colidx, A->fcolidx, A->fnpnzs * sizeof(*cg->d_colidx), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_a, A->fnpnzs * sizeof(*cg->d_a));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_a, A->fa, A->fnpnzs * sizeof(*cg->d_a), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;

    /* copy sparse matrix to device */
    err = hipMalloc((void **)&cg->d_orowptr, (A->nborderrows + A->nghostrows + 1) * sizeof(*cg->d_orowptr));
    if (err)
        return ACG_ERR_HIP;
    if (sizeof(*cg->d_orowptr) == sizeof(*A->orowptr))
    {
        err = hipMemcpy(cg->d_orowptr, A->orowptr, (A->nborderrows + A->nghostrows + 1) * sizeof(*cg->d_orowptr), hipMemcpyHostToDevice);
        if (err)
            return ACG_ERR_HIP;
    }
    else
    {
        acgidx_t *tmprowptr = malloc((A->nborderrows + A->nghostrows + 1) * sizeof(*tmprowptr));
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
        err = hipMemcpy(cg->d_orowptr, tmprowptr, (A->nborderrows + A->nghostrows + 1) * sizeof(*cg->d_orowptr), hipMemcpyHostToDevice);
        if (err)
            return ACG_ERR_HIP;
        free(tmprowptr);
    }
    err = hipMalloc((void **)&cg->d_ocolidx, A->onpnzs * sizeof(*cg->d_ocolidx));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_ocolidx, A->ocolidx, A->onpnzs * sizeof(*cg->d_ocolidx), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMalloc((void **)&cg->d_oa, A->onpnzs * sizeof(*cg->d_oa));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(cg->d_oa, A->oa, A->onpnzs * sizeof(*cg->d_oa), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;

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
 * ‘acgsolverhip_solve()’ solves the given linear system, Ax=b, using the
 * conjugate gradient method.
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
int acgsolverhip_solve(
    struct acgsolverhip *cg,
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

// #define USE_MERGE_BASED_SPMV

#if defined(ACG_HAVE_MPI) && defined(ACG_HAVE_HIPBLAS) && defined(ACG_HAVE_HIPSPARSE)
/**
 * ‘acgsolverhip_solvempi()’ solves the given linear system, Ax=b, using
 * the conjugate gradient method. The linear system may be distributed
 * across multiple processes and communication is handled using MPI.
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
int acgsolverhip_solvempi(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
    /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO; */
    /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
    /* } */

    hipStream_t stream = 0;
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

    /* configure hipblas and hipsparse to use device-side pointers */
    hipblasPointerMode_t hipblaspointermode;
    err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
    if (err)
        return ACG_ERR_HIPBLAS;
    err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
    if (err)
        return ACG_ERR_HIPBLAS;
    hipsparsePointerMode_t hipsparsepointermode;
    err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    double *rnrm2sqr;
    err = hipHostMalloc((void **)&rnrm2sqr, sizeof(*rnrm2sqr), hipHostMallocNumaUser);
    if (err)
        return ACG_ERR_HIP;
    hipStream_t copystream;
    err = hipStreamCreateWithFlags(&copystream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t rnrm2sqrready;
    err = hipEventCreateWithFlags(&rnrm2sqrready, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    /* hipEvent_t rnrm2sqrreceived; */
    /* err = hipEventCreateWithFlags(&rnrm2sqrreceived, hipEventDisableTiming); if (err) return ACG_ERR_HIP; */

    /* copy right-hand side and initial guess to device */
    double *d_b;
    err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    double *d_x;
    err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;

    /* used to overlap P2P communication with SpMV */
    hipStream_t commstream;
    err = hipStreamCreateWithFlags(&commstream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t xreadytosend, xreceived;
    err = hipEventCreateWithFlags(&xreadytosend, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(xreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&xreceived, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t preadytosend, preceived;
    err = hipEventCreateWithFlags(&preadytosend, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(preadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&preceived, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;

    /* create hipsparse matrix and vectors */
    hipsparseDnVecDescr_t vecx, vecr, vecp, vect;
    err = hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecp, A->nownedrows, d_p, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vect, A->nownedrows, d_t, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    hipsparseDnVecDescr_t vecxo, vecro, vecpo, vecto;
    if (commsize > 1)
    {
        err = hipsparseCreateDnVec(&vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecpo, A->nborderrows + A->nghostrows, d_p + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecto, A->nborderrows + A->nghostrows, d_t + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }

    hipsparseSpMatDescr_t matA;
    err = hipsparseCreateCsr(
        &matA, A->nownedrows, A->nownedrows, A->fnpnzs,
        d_rowptr, d_colidx, d_a,
        HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    size_t buffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    void *d_buffer;
    err = hipMalloc(&d_buffer, buffersize);
    if (err)
        return ACG_ERR_HIP;
    /* Note: Disable hipsparseSpMV_preprocess, because it degrades
     * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
    err = hipsparseSpMV_preprocess(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_HIPSPARSE; }
#endif

    hipsparseSpMatDescr_t matO;
    void *d_obuffer;
    if (commsize > 1)
    {
        err = hipsparseCreateCsr(
            &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows, A->onpnzs,
            d_orowptr, d_ocolidx, d_oa,
            HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
            HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        size_t obuffersize;
        err = hipsparseSpMV_bufferSize(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMalloc(&d_obuffer, obuffersize);
        if (err)
            return ACG_ERR_HIP;
        /* Note: Disable hipsparseSpMV_preprocess, because it degrades
         * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
        err = hipsparseSpMV_preprocess(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_HIPSPARSE; }
#endif
    }

#ifdef USE_MERGE_BASED_SPMV
    /* prepare merge-based SpMV */
    const int TASKS_PER_THREAD = 10;
    acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
    acgidx_t nstartrows = (ntasks + TASKS_PER_THREAD - 1) / TASKS_PER_THREAD;
    acgidx_t *d_startrows;
    err = hipMalloc((void **)&d_startrows, nstartrows * sizeof(*d_startrows));
    if (err)
        return ACG_ERR_HIP;
    acgsolverhip_csrgemv_merge_startrows(
        A->nprows - A->nghostrows, d_rowptr, nstartrows, d_startrows, stream);
    if (hipPeekAtLastError())
        return ACG_ERR_HIP;
    hipStreamSynchronize(stream);
#endif

    /* create timing events for profiling */
    acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0, nallreduce = 0, nhalo = 0;
    hipEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
    tgemv = malloc(2 * (maxits + 1) * sizeof(*tgemv));
    if (!tgemv)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 1); i++)
        hipEventCreate(&tgemv[i]);
    tdot = malloc(2 * maxits * sizeof(*tdot));
    if (!tdot)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * maxits; i++)
        hipEventCreate(&tdot[i]);
    tnrm2 = malloc(2 * (maxits + 2) * sizeof(*tnrm2));
    if (!tnrm2)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 2); i++)
        hipEventCreate(&tnrm2[i]);
    taxpy = malloc(2 * (3 * maxits) * sizeof(*taxpy));
    if (!taxpy)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (3 * maxits); i++)
        hipEventCreate(&taxpy[i]);
    tcopy = malloc(2 * 2 * sizeof(*tcopy));
    if (!tcopy)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * 2; i++)
        hipEventCreate(&tcopy[i]);
    tallreduce = malloc(2 * (2 * maxits + 2) * sizeof(*tallreduce));
    if (!tallreduce)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (2 * maxits + 2); i++)
        hipEventCreate(&tallreduce[i]);
    thalo = malloc(2 * (maxits + 1) * sizeof(*thalo));
    if (!thalo)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 1); i++)
        hipEventCreate(&thalo[i]);
#endif

    /* warmup iterations for dot/allreduce */
    for (int i = 0; i < warmup; i++)
    {
        hipMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
        hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
        hipMemcpy(d_pdott, d_zero, sizeof(*d_pdott), hipMemcpyDeviceToDevice);
        err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = hipblasDdot(hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = hipblasDdot(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t, 1, d_pdott);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
    }
    hipMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_pdott, d_zero, sizeof(*d_pdott), hipMemcpyDeviceToDevice);

    /* warmup iterations for halo exchange/SpMV */
    for (int i = 0; i < warmup; i++)
    {
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, xreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
#ifndef USE_MERGE_BASED_SPMV
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
#else
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_x, d_r, d_rowptr, d_colidx, d_a, -1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
            return err;
#endif
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(xreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, xreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            err = hipEventRecord(xreadytosend, stream);
            if (err)
                return ACG_ERR_HIP;
        }

        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, preadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
#ifndef USE_MERGE_BASED_SPMV
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecp, d_zero, vect, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
#else
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_t, d_p, d_rowptr, d_colidx, d_a, 1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
            return err;
#endif
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(preceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, preceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecpo, d_one, vecto, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            err = hipEventRecord(preadytosend, stream);
            if (err)
                return ACG_ERR_HIP;
        }
    }

    /* warmup iterations for axpy */
    for (int i = 0; i < warmup; i++)
    {
        err = acgsolverhip_daxpy_alpha(cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_zero, d_one, d_p, d_x);
        if (err)
            return err;
        err = acgsolverhip_daypx_beta(cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_one, d_p, d_r);
        if (err)
            return err;
    }

    /* warmup iterations for copy */
    for (int i = 0; i < warmup; i++)
    {
        err = hipblasDcopy(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPBLAS;
        }
        err = hipblasDcopy(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p, 1);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPBLAS;
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
    err = acgcomm_barrier_hip(stream, comm, errcode);
    if (err)
        return err;
    hipStreamSynchronize(stream);
    gettime(&t0);

    /* compute right-hand side norm */
    double bnrm2sqr;
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
    cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
    if (commsize > 1)
    {
        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
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
    err = hipMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;
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
    acgEventRecord(tcopy[2 * ncopy + 0], 0);
    err = hipblasDcopy(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tcopy[2 * ncopy + 1], 0);
    ncopy++;
    cg->ncopy++;
    cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

    if (commsize > 1)
    {
        err = acghalo_exchange_hip_begin(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
#ifndef USE_MERGE_BASED_SPMV
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
#else
    err = acgsolverhip_csrgemv_merge(
        A->nownedrows, d_x, d_r, d_rowptr, d_colidx, d_a, -1.0, 1.0,
        nstartrows, d_startrows, stream);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
    }
#endif
    if (commsize > 1)
    {
        acgEventRecord(thalo[2 * nhalo + 0], 0);
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
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
        err = hipEventRecord(xreceived, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(stream, xreceived, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

    /* compute initial search direction: p = r₀ */
    acgEventRecord(tcopy[2 * ncopy + 0], 0);
    err = hipblasDcopy(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p, 1);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tcopy[2 * ncopy + 1], 0);
    ncopy++;
    cg->ncopy++;
    cg->Bcopy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->r.x));
    err = hipEventRecord(preadytosend, stream);
    if (err)
        return ACG_ERR_HIP;

    /* compute initial residual norm */
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = hipblasDdot(hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
    cg->Bnrm2 += (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);
    if (commsize > 1)
    {
        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
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
    err = hipMemcpy(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;
    cg->rnrm2 = cg->r0nrm2 = sqrt(*rnrm2sqr);
    residualrtol *= cg->r0nrm2;

    /* initial convergence test */
    if ((residualatol > 0 && cg->rnrm2 < residualatol) ||
        (residualrtol > 0 && cg->rnrm2 < residualrtol))
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
            err = hipStreamWaitEvent(commstream, preadytosend, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
        }
        acgEventRecord(tgemv[2 * ngemv + 0], 0);
#ifndef USE_MERGE_BASED_SPMV
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecp, d_zero, vect, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
#else
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_t, d_p, d_rowptr, d_colidx, d_a, 1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
#endif
        if (commsize > 1)
        {
            acgEventRecord(thalo[2 * nhalo + 0], 0);
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                cg->p.num_nonzeros, d_p, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
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
            err = hipEventRecord(preceived, commstream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipStreamWaitEvent(stream, preceived, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecpo, d_one, vecto, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIPSPARSE;
            }
        }
        acgEventRecord(tgemv[2 * ngemv + 1], 0);
        ngemv++;
        cg->ngemv++;
        cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
        cg->Bgemv +=
            (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->t.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->p.num_nonzeros * sizeof(*cg->p.x);

        /* compute (p,Ap) */
        acgEventRecord(tdot[2 * ndot + 0], 0);
        err = hipblasDdot(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t, 1, d_pdott);
        if (err)
        {
            if (errcode)
                *errcode = err;
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
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
            err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
#else
            double pdott;
            hipMemcpy(&pdott, d_pdott, sizeof(*d_pdott), hipMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, &pdott, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
            hipMemcpy(d_pdott, &pdott, sizeof(*d_pdott), hipMemcpyHostToDevice);
#endif
            acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
            nallreduce++;
            cg->nallreduce++;
            cg->Ballreduce += sizeof(*d_pdott);
        }
        err = hipMemcpyAsync(d_rnrm2sqr_prev, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }

        /* update residual, rₖ = -αt + rₖ₋₁, where α = (rₖ₋₁,rₖ₋₁)/(p,t) */
        acgEventRecord(taxpy[2 * naxpy + 0], 0);
#ifndef NO_FUSED_KERNELS
        err = acgsolverhip_daxpy_minus_alpha(cg->t.num_nonzeros - cg->t.num_ghost_nonzeros, d_rnrm2sqr, d_pdott, d_t, d_r);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
#else
        err = acgsolverhip_alpha(d_alpha, d_minus_alpha, d_rnrm2sqr, d_pdott);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
        err = hipblasDaxpy(hipblas, cg->t.num_nonzeros - cg->t.num_ghost_nonzeros, d_minus_alpha, d_t, 1, d_r, 1);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
#endif
        acgEventRecord(taxpy[2 * naxpy + 1], 0);
        naxpy++;
        cg->naxpy++;
        cg->nflops += 2 * (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros);
        cg->Baxpy += (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * (sizeof(*cg->t.x) + sizeof(*cg->r.x));

        /* compute residual norm */
        acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
        err = hipblasDdot(hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err)
        {
            if (errcode)
                *errcode = err;
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
        acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
        nnrm2++;
        cg->nnrm2++;
        cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
        cg->Bnrm2 += (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);
#ifndef HOST_ALLREDUCE
        if (commsize > 1)
        {
            acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
            err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
            acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
            nallreduce++;
            cg->nallreduce++;
            cg->Ballreduce += sizeof(*d_rnrm2sqr);
        }
        err = hipEventRecord(rnrm2sqrready, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(copystream, rnrm2sqrready, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost, copystream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        /* err = hipEventRecord(rnrm2sqrreceived, copystream); */
        /* if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return ACG_ERR_HIP; } */
#else
        if (commsize > 1)
        {
            acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
            hipMemcpy(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost);
            MPI_Allreduce(MPI_IN_PLACE, rnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm->mpicomm);
            err = hipMemcpyAsync(d_rnrm2sqr, rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyHostToDevice, copystream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
            nallreduce++;
            cg->nallreduce++;
            cg->Ballreduce += sizeof(*rnrm2sqr);
        }
        else
        {
            err = hipEventRecord(rnrm2sqrready, stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipStreamWaitEvent(copystream, rnrm2sqrready, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost, copystream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
        }
#endif

        /* update solution, x = αp + x, where α = (r,r)/(p,t) */
        acgEventRecord(taxpy[2 * naxpy + 0], 0);
#ifndef NO_FUSED_KERNELS
        err = acgsolverhip_daxpy_alpha(cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_rnrm2sqr_prev, d_pdott, d_p, d_x);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
#else
        err = hipblasDaxpy(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_alpha, d_p, 1, d_x, 1);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
#endif
        acgEventRecord(taxpy[2 * naxpy + 1], 0);
        naxpy++;
        cg->naxpy++;
        cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
        cg->Baxpy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*x->x));

        /* update search direction, p = βp + r, where β = (rₖ,rₖ)/(rₖ₋₁,rₖₖ₋₁) */
        acgEventRecord(taxpy[2 * naxpy + 0], 0);
#ifndef NO_FUSED_KERNELS
        err = acgsolverhip_daypx_beta(cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_rnrm2sqr, d_rnrm2sqr_prev, d_p, d_r);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
#else
        err = acgsolverhip_beta(d_beta, d_rnrm2sqr, d_rnrm2sqr_prev);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
        err = hipblasDscal(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_beta, d_p, 1);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
        err = hipblasDaxpy(hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_r, 1, d_p, 1);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
#endif
        acgEventRecord(taxpy[2 * naxpy + 1], 0);
        naxpy++;
        cg->naxpy++;
        cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
        cg->Baxpy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->r.x));
        err = hipEventRecord(preadytosend, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }

        /* convergence tests */
        /* hipEventSynchronize(rnrm2sqrreceived); */
        hipStreamSynchronize(copystream);
        cg->rnrm2 = sqrt(*rnrm2sqr);
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) ||
            (diffrtol > 0 && cg->dxnrm2 < diffrtol) ||
            (residualatol > 0 && cg->rnrm2 < residualatol) ||
            (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            hipStreamSynchronize(stream);
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
        hipEventSynchronize(tgemv[2 * i + 1]);
        hipEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
        cg->tgemv += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < ndot; i++)
    {
        hipEventSynchronize(tdot[2 * i + 1]);
        hipEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
        cg->tdot += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nnrm2; i++)
    {
        hipEventSynchronize(tnrm2[2 * i + 1]);
        hipEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
        cg->tnrm2 += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < naxpy; i++)
    {
        hipEventSynchronize(taxpy[2 * i + 1]);
        hipEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
        cg->taxpy += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < ncopy; i++)
    {
        hipEventSynchronize(tcopy[2 * i + 1]);
        hipEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
        cg->tcopy += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nallreduce; i++)
    {
        hipEventSynchronize(tallreduce[2 * i + 1]);
        hipEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
        cg->tallreduce += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nhalo; i++)
    {
        hipEventSynchronize(thalo[2 * i + 1]);
        hipEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
        cg->thalo += 1.0e-3 * t;
    }
#endif

    /* copy solution back to host */
    err = hipMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;

    /* free hipsparse matrix and vectors */
    hipsparseDestroyDnVec(vecx);
    hipsparseDestroyDnVec(vecr);
    hipsparseDestroyDnVec(vecp);
    hipsparseDestroyDnVec(vect);
    if (commsize > 1)
    {
        hipsparseDestroyDnVec(vecxo);
        hipsparseDestroyDnVec(vecro);
        hipsparseDestroyDnVec(vecpo);
        hipsparseDestroyDnVec(vecto);
    }
    hipsparseDestroySpMat(matA);
    hipFree(d_buffer);
    if (commsize > 1)
    {
        hipsparseDestroySpMat(matO);
        hipFree(d_obuffer);
    }
    hipFree(d_x);
    hipFree(d_b);
    hipHostFree(rnrm2sqr);
    hipStreamDestroy(commstream);
    hipStreamDestroy(copystream);

    /* reset hipsparse and hipblas pointer modes */
    err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipblasSetPointerMode(hipblas, hipblaspointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    /* check for HIP errors */
    if (hipGetLastError() != hipSuccess)
        return ACG_ERR_HIP;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged)
        return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
}
#endif

/**
 * ‘acgsolverhip_solve_pipelined()’ solves the given linear system,
 * Ax=b, using a pipelined conjugate gradient method. The linear
 * system may be distributed across multiple processes and
 * communication is handled using MPI.
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
int acgsolverhip_solve_pipelined(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
        err = hipMalloc((void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->q)
    {
        cg->q = malloc(sizeof(*cg->q));
        if (!cg->q)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->q, x);
        if (err)
            return err;
        err = hipMalloc((void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->z)
    {
        cg->z = malloc(sizeof(*cg->z));
        if (!cg->z)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->z, x);
        if (err)
            return err;
        err = hipMalloc((void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
        if (err)
            return ACG_ERR_HIP;
    }

    /* /\* If the stopping criterion is based on the difference in */
    /*  * solution from one iteration to the next, then allocate */
    /*  * additional storage for storing the difference. *\/ */
    /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
    /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO; */
    /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
    /* } */

    hipStream_t stream = 0;
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

    /* configure hipblas and hipsparse to use device-side pointers */
    hipblasPointerMode_t hipblaspointermode;
    err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
    if (err)
        return ACG_ERR_HIPBLAS;
    err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
    if (err)
        return ACG_ERR_HIPBLAS;
    hipsparsePointerMode_t hipsparsepointermode;
    err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    double *rnrm2sqr;
    err = hipHostMalloc((void **)&rnrm2sqr, sizeof(*rnrm2sqr), hipHostMallocNumaUser);
    if (err)
        return ACG_ERR_HIP;
    hipStream_t copystream;
    err = hipStreamCreateWithFlags(&copystream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t rnrm2sqrready;
    hipEventCreateWithFlags(&rnrm2sqrready, hipEventDisableTiming);

    /* copy right-hand side and initial guess to device */
    double *d_b;
    err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    double *d_x;
    err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;

    /* used to overlap P2P communication with SpMV */
    hipStream_t commstream;
    err = hipStreamCreateWithFlags(&commstream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t xreadytosend, xreceived;
    err = hipEventCreateWithFlags(&xreadytosend, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(xreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&xreceived, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t rreadytosend, rreceived;
    err = hipEventCreateWithFlags(&rreadytosend, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(rreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&rreceived, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t wreadytosend, wreceived;
    err = hipEventCreateWithFlags(&wreadytosend, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(wreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&wreceived, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;

    /* create hipsparse matrix and vectors */
    hipsparseDnVecDescr_t vecx, vecr, vecw, vecq;
    err = hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecw, A->nownedrows, d_w, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecq, A->nownedrows, d_q, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    hipsparseDnVecDescr_t vecxo, vecro, vecwo, vecqo;
    if (commsize > 1)
    {
        err = hipsparseCreateDnVec(&vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecqo, A->nborderrows + A->nghostrows, d_q + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }

    hipsparseSpMatDescr_t matA;
    err = hipsparseCreateCsr(
        &matA, A->nownedrows, A->nownedrows, A->fnpnzs,
        d_rowptr, d_colidx, d_a,
        HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    size_t buffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    void *d_buffer;
    err = hipMalloc(&d_buffer, buffersize);
    if (err)
        return ACG_ERR_HIP;
    /* Note: Disable hipsparseSpMV_preprocess, because it degrades
     * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
    err = hipsparseSpMV_preprocess(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_HIPSPARSE; }
#endif

    hipsparseSpMatDescr_t matO;
    void *d_obuffer;
    if (commsize > 1)
    {
        err = hipsparseCreateCsr(
            &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows, A->onpnzs,
            d_orowptr, d_ocolidx, d_oa,
            HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
            HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        size_t obuffersize;
        err = hipsparseSpMV_bufferSize(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMalloc(&d_obuffer, obuffersize);
        if (err)
            return ACG_ERR_HIP;
        /* Note: Disable hipsparseSpMV_preprocess, because it degrades
         * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
        err = hipsparseSpMV_preprocess(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_HIPSPARSE; }
#endif
    }

    /* create timing events for profiling */
    acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0, nallreduce = 0, nhalo = 0;
    hipEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
    tgemv = malloc(2 * (maxits + 2) * sizeof(*tgemv));
    if (!tgemv)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 2); i++)
        hipEventCreate(&tgemv[i]);
    tdot = malloc(2 * maxits * sizeof(*tdot));
    if (!tdot)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * maxits; i++)
        hipEventCreate(&tdot[i]);
    tnrm2 = malloc(2 * (maxits + 1) * sizeof(*tnrm2));
    if (!tnrm2)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 1); i++)
        hipEventCreate(&tnrm2[i]);
    taxpy = malloc(2 * maxits * sizeof(*taxpy));
    if (!taxpy)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * maxits; i++)
        hipEventCreate(&taxpy[i]);
    tcopy = malloc(2 * 1 * sizeof(*tcopy));
    if (!tcopy)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * 1; i++)
        hipEventCreate(&tcopy[i]);
    tallreduce = malloc(2 * (maxits + 1) * sizeof(*tallreduce));
    if (!tallreduce)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 1); i++)
        hipEventCreate(&tallreduce[i]);
    thalo = malloc(2 * (maxits + 2) * sizeof(*thalo));
    if (!thalo)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 2); i++)
        hipEventCreate(&thalo[i]);
#endif

    /* warmup iterations for dot/allreduce */
    for (int i = 0; i < warmup; i++)
    {
        hipMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
        hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
        hipMemcpy(d_delta, d_zero, sizeof(*d_delta), hipMemcpyDeviceToDevice);
        err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = hipblasDdot(hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err)
            return ACG_ERR_HIPBLAS;
        err = hipblasDdot(hipblas, cg->w->num_nonzeros - cg->w->num_ghost_nonzeros, d_w, 1, d_r, 1, d_delta);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        hipStreamSynchronize(stream);
    }
    hipMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_delta, d_zero, sizeof(*d_delta), hipMemcpyDeviceToDevice);

    /* warmup iterations for halo exchange/SpMV */
    for (int i = 0; i < warmup; i++)
    {
        /* r = b-Ax */
        err = hipblasDcopy(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPBLAS;
        }
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, xreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(xreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, xreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            err = hipEventRecord(xreadytosend, stream);
            if (err)
                return ACG_ERR_HIP;
        }

        /* w = Ar */
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, rreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecr, d_zero, vecw, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(rreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, rreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecro, d_one, vecwo, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
        }
        err = hipEventRecord(wreadytosend, stream);
        if (err)
            return ACG_ERR_HIP;

        /* q = Aw */
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, wreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecw, d_zero, vecq, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(wreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, wreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecwo, d_one, vecqo, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
        }
    }

    /* warmup iterations for axpy */
    err = hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_z, 0, (cg->z->num_nonzeros - cg->z->num_ghost_nonzeros) * sizeof(*d_z));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_t, 0, (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_p, 0, (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*d_p));
    if (err)
        return ACG_ERR_HIP;
    for (int i = 0; i < warmup; i++)
    {
        err = hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = hipMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = hipMemcpy(d_delta, d_inf, sizeof(*d_delta), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = acgsolverhip_pipelined_daxpy_fused(
            cg->t.num_nonzeros - cg->t.num_ghost_nonzeros,
            d_rnrm2sqr, d_rnrm2sqr_prev, d_delta,
            d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha, stream);
        if (err)
            return err;
    }
    err = hipMemset(d_r, 0, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_w, 0, (cg->w->num_nonzeros - cg->w->num_ghost_nonzeros) * sizeof(*d_w));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_q, 0, (cg->q->num_nonzeros - cg->q->num_ghost_nonzeros) * sizeof(*d_q));
    if (err)
        return ACG_ERR_HIP;

    /* set scalars to infinity (needed to produce correct results on
     * the first call to acgsolverhip_pipelined_daxpy_fused) */
    err = hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;

    /* set the vectors z, t and p to zero */
    err = hipMemset(d_z, 0, (cg->z->num_nonzeros - cg->z->num_ghost_nonzeros) * sizeof(*d_z));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_t, 0, (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_p, 0, (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*d_p));
    if (err)
        return ACG_ERR_HIP;

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
    err = acgcomm_barrier_hip(stream, comm, errcode);
    if (err)
        return err;
    hipStreamSynchronize(stream);
    gettime(&t0);

    /* compute right-hand side norm */
    double bnrm2sqr;
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
    cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
    if (commsize > 1)
    {
        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
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
    err = hipMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;
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
    acgEventRecord(tcopy[2 * ncopy + 0], 0);
    err = hipblasDcopy(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tcopy[2 * ncopy + 1], 0);
    ncopy++;
    cg->ncopy++;
    cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

    if (commsize > 1)
    {
        err = acghalo_exchange_hip_begin(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    if (commsize > 1)
    {
        acgEventRecord(thalo[2 * nhalo + 0], 0);
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
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
        err = hipEventRecord(xreceived, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(stream, xreceived, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipEventRecord(rreadytosend, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

    /* compute w = Ar */
    if (commsize > 1)
    {
        err = hipStreamWaitEvent(commstream, rreadytosend, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = acghalo_exchange_hip_begin(
            cg->halo, cg->haloexchange,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_one, matA, vecr, d_zero, vecw, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    if (commsize > 1)
    {
        acgEventRecord(thalo[2 * nhalo + 0], 0);
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            cg->r.num_nonzeros, d_r, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
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
        err = hipEventRecord(rreceived, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(stream, rreceived, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matO, vecro, d_one, vecwo, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->w->x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->r.num_nonzeros * sizeof(*cg->r.x);
    err = hipEventRecord(wreadytosend, stream);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIP;
    }

    /* iterative solver loop */
    for (int k = 0; k < maxits; k++)
    {

        /* compute residual norm (r,r) */
        acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
        err = hipblasDdot(hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err)
        {
            if (errcode)
                *errcode = err;
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
        acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
        nnrm2++;
        cg->nnrm2++;
        cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
        cg->Bnrm2 += (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);

        /* compute (w,r) */
        acgEventRecord(tdot[2 * ndot + 0], 0);
        err = hipblasDdot(hipblas, cg->w->num_nonzeros - cg->w->num_ghost_nonzeros, d_w, 1, d_r, 1, d_delta);
        if (err)
        {
            if (errcode)
                *errcode = err;
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
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
            err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
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
        err = hipEventRecord(rnrm2sqrready, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(copystream, rnrm2sqrready, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost, copystream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }

        /* compute q = Aw */
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, wreadytosend, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
        }
        acgEventRecord(tgemv[2 * ngemv + 0], 0);
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecw, d_zero, vecq, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        if (commsize > 1)
        {
            acgEventRecord(thalo[2 * nhalo + 0], 0);
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
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
            err = hipEventRecord(wreceived, commstream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipStreamWaitEvent(stream, wreceived, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecwo, d_one, vecqo, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
        }
        acgEventRecord(tgemv[2 * ngemv + 1], 0);
        ngemv++;
        cg->ngemv++;
        cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
        cg->Bgemv +=
            (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->q->x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->w->num_nonzeros * sizeof(*cg->w->x);

        /* wait for host to receive updated residual norm */
        hipStreamSynchronize(copystream);
        cg->rnrm2 = sqrt(*rnrm2sqr);
        if (k == 0)
        {
            cg->r0nrm2 = cg->rnrm2;
            residualrtol *= cg->r0nrm2;
        }

        /* convergence tests */
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) ||
            (diffrtol > 0 && cg->dxnrm2 < diffrtol) ||
            (residualatol > 0 && cg->rnrm2 < residualatol) ||
            (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            hipStreamSynchronize(stream);
            converged = true;
            break;
        }

        /* update vectors */
        acgEventRecord(taxpy[2 * naxpy + 0], 0);
        err = acgsolverhip_pipelined_daxpy_fused(
            cg->t.num_nonzeros - cg->t.num_ghost_nonzeros,
            d_rnrm2sqr, d_rnrm2sqr_prev, d_delta,
            d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha, stream);
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
        err = hipEventRecord(wreadytosend, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
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
        hipEventSynchronize(tgemv[2 * i + 1]);
        hipEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
        cg->tgemv += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < ndot; i++)
    {
        hipEventSynchronize(tdot[2 * i + 1]);
        hipEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
        cg->tdot += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nnrm2; i++)
    {
        hipEventSynchronize(tnrm2[2 * i + 1]);
        hipEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
        cg->tnrm2 += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < naxpy; i++)
    {
        hipEventSynchronize(taxpy[2 * i + 1]);
        hipEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
        cg->taxpy += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < ncopy; i++)
    {
        hipEventSynchronize(tcopy[2 * i + 1]);
        hipEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
        cg->tcopy += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nallreduce; i++)
    {
        hipEventSynchronize(tallreduce[2 * i + 1]);
        hipEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
        cg->tallreduce += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nhalo; i++)
    {
        hipEventSynchronize(thalo[2 * i + 1]);
        hipEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
        cg->thalo += 1.0e-3 * t;
    }
#endif

    /* copy solution back to host */
    err = hipMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;

    /* free hipsparse matrix and vectors */
    hipsparseDestroyDnVec(vecx);
    hipsparseDestroyDnVec(vecr);
    hipsparseDestroyDnVec(vecw);
    hipsparseDestroyDnVec(vecq);
    if (commsize > 1)
    {
        hipsparseDestroyDnVec(vecxo);
        hipsparseDestroyDnVec(vecro);
        hipsparseDestroyDnVec(vecwo);
        hipsparseDestroyDnVec(vecqo);
    }
    hipsparseDestroySpMat(matA);
    hipFree(d_buffer);
    if (commsize > 1)
    {
        hipsparseDestroySpMat(matO);
        hipFree(d_obuffer);
    }
    hipFree(d_x);
    hipFree(d_b);
    hipHostFree(rnrm2sqr);
    hipStreamDestroy(copystream);
    hipStreamDestroy(commstream);

    /* reset hipsparse and hipblas pointer modes */
    err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipblasSetPointerMode(hipblas, hipblaspointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPBLAS;
    }

    /* check for HIP errors */
    if (hipGetLastError() != hipSuccess)
        return ACG_ERR_HIP;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged)
        return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
}

/**
 * ‘acgsolverhip_solve_preconditioned()’ solves the given linear system,
 * Ax=b, using a preconditioned conjugate gradient method. The linear
 * system may be distributed across multiple processes and
 * communication is handled using MPI.
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
int acgsolverhip_solve_preconditioned(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
    err = hipMalloc(
        (void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->q)
  {
    cg->q = malloc(sizeof(*cg->q));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->q, x);
    if (err)
      return err;
    err = hipMalloc(
        (void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->z)
  {
    cg->z = malloc(sizeof(*cg->z));
    if (!cg->z)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->z, x);
    if (err)
      return err;
    err = hipMalloc(
        (void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->m)
  {
    cg->m = malloc(sizeof(*cg->m));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->m, x);
    if (err)
      return err;
    err = hipMalloc(
        (void **)&cg->d_m, cg->q->num_nonzeros * sizeof(*cg->d_m));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->n)
  {
    cg->n = malloc(sizeof(*cg->n));
    if (!cg->n)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->n, x);
    if (err)
      return err;
    err = hipMalloc(
        (void **)&cg->d_n, cg->n->num_nonzeros * sizeof(*cg->d_n));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->u)
  {
    cg->u = malloc(sizeof(*cg->u));
    if (!cg->u)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->u, x);
    if (err)
      return err;
    err = hipMalloc(
        (void **)&cg->d_u, cg->u->num_nonzeros * sizeof(*cg->d_u));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->y)
  {
    cg->y = malloc(sizeof(*cg->y));
    if (!cg->y)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->y, x);
    if (err)
      return err;
    err = hipMalloc(
        (void **)&cg->d_y, cg->y->num_nonzeros * sizeof(*cg->d_y));
    if (err)
      return ACG_ERR_HIP;
  }

  hipStream_t stream = 0;
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

  hipStream_t collective_stream;
  err = hipStreamCreateWithFlags(&collective_stream, hipStreamNonBlocking);
  if (err)
    return ACG_ERR_HIP;

  /* configure hipblas and hipsparse to use device-side pointers */
  hipblasPointerMode_t hipblaspointermode;
  err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_HIPBLAS;
  hipsparsePointerMode_t hipsparsepointermode;
  err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  double *rnrm2sqr;
  err = hipHostMalloc((void **)&rnrm2sqr, sizeof(*rnrm2sqr), hipHostMallocNumaUser);
  if (err)
    return ACG_ERR_HIP;

  double *rnrm2sqr_prev;
  err = hipHostMalloc((void **)&rnrm2sqr_prev, sizeof(*rnrm2sqr_prev), hipHostMallocNumaUser);
  if (err)
    return ACG_ERR_HIP;

  hipStream_t copystream;
  err = hipStreamCreateWithFlags(&copystream, hipStreamNonBlocking);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t rnrm2sqrready;
  hipEventCreateWithFlags(&rnrm2sqrready, hipEventDisableTiming);
  /* hipEvent_t rnrm2sqrreceived; */
  /* err = hipEventCreateWithFlags(&rnrm2sqrreceived,
   * hipEventDisableTiming); if (err) return ACG_ERR_HIP; */

  /* copy right-hand side and initial guess to device */
  double *d_b;
  err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpy(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice);
  if (err)
    return ACG_ERR_HIP;
  double *d_x;
  err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpy(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice);
  if (err)
    return ACG_ERR_HIP;

  /* used to overlap P2P communication with SpMV */
  hipStream_t commstream;
  err = hipStreamCreateWithFlags(&commstream, hipStreamNonBlocking);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t xreadytosend, xreceived;
  err = hipEventCreateWithFlags(&xreadytosend, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  err = hipEventRecord(xreadytosend, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipEventCreateWithFlags(&xreceived, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t preadytosend, preceived, reduced;
  err = hipEventCreateWithFlags(&preadytosend, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  err = hipEventRecord(preadytosend, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipEventCreateWithFlags(&preceived, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t mreadytosend, mreceived;
  err = hipEventCreateWithFlags(&mreadytosend, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  err = hipEventCreateWithFlags(&mreceived, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  err = hipEventCreateWithFlags(&reduced, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t ureadytosend, ureceived;
  err = hipEventCreateWithFlags(&ureadytosend, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;
  // err = hipEventRecord(ureadytosend, stream);
  // if (err)
  //     return ACG_ERR_HIP;
  err = hipEventCreateWithFlags(&ureceived, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;

  hipEvent_t dotEvent;
  err = hipEventCreateWithFlags(&dotEvent, hipEventDisableTiming);
  if (err)
    return ACG_ERR_HIP;

  /* create hipsparse matrix and vectors */
  hipsparseDnVecDescr_t vecx, vecr, vecp, vect;
  err = hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseCreateDnVec(&vecp, A->nownedrows, d_p, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseCreateDnVec(&vect, A->nownedrows, d_t, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  hipsparseDnVecDescr_t vecxo, vecro, vecpo, vecto;
  if (commsize > 1)
  {
    err = hipsparseCreateDnVec(
        &vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &vecpo, A->nborderrows + A->nghostrows, d_p + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &vecto, A->nborderrows + A->nghostrows, d_t + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
  }

  hipsparseDnVecDescr_t vecz, vecn, vecm, vecw, vecy, vecu;
  err = hipsparseCreateDnVec(&vecz, A->nownedrows, d_z, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  err = hipsparseCreateDnVec(&vecm, A->nownedrows, d_m, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  err = hipsparseCreateDnVec(&vecn, A->nownedrows, d_n, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  err = hipsparseCreateDnVec(&vecw, A->nownedrows, d_w, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseCreateDnVec(&vecy, A->nownedrows, d_y, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseCreateDnVec(&vecu, A->nownedrows, d_u, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  hipsparseDnVecDescr_t vecno, vecmo, vecwo, veczo, vecuo;
  if (commsize > 1)
  {
    err = hipsparseCreateDnVec(
        &vecno, A->nborderrows + A->nghostrows, d_n + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &vecmo, A->nborderrows + A->nghostrows, d_m + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &veczo, A->nborderrows + A->nghostrows, d_z + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(
        &vecuo, A->nborderrows + A->nghostrows, d_u + A->borderrowoffset,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
  }

  hipsparseSpMatDescr_t matA;
  err = hipsparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      HIPSPARSE_IDX_T, HIPSPARSE_IDX_T, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  size_t buffersize;
  err = hipsparseSpMV_bufferSize(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  void *d_buffer;
  err = hipMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_HIP;

  /* Note: Disable hipsparseSpMV_preprocess, because it degrades
   * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
  err = hipsparseSpMV_preprocess(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
#endif

  hipsparseSpMatDescr_t matO;
  void *d_obuffer;
  if (commsize > 1)
  {
    err = hipsparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows,
        A->nborderrows + A->nghostrows, A->onpnzs, d_orowptr, d_ocolidx,
        d_oa, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T, HIPSPARSE_INDEX_BASE_ZERO,
        HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    size_t obuffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_HIP;
    /* Note: Disable hipsparseSpMV_preprocess, because it degrades
     * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
    err = hipsparseSpMV_preprocess(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
#endif
  }

  /* create timing events for profiling */
  acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0,
           nallreduce = 0, nhalo = 0;
  hipEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
  tgemv = malloc(2 * (maxits + 2) * sizeof(*tgemv));
  if (!tgemv)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    hipEventCreate(&tgemv[i]);
  tdot = malloc(2 * (maxits) * sizeof(*tdot));
  if (!tdot)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits); i++)
    hipEventCreate(&tdot[i]);
  tnrm2 = malloc(2 * (maxits + 2) * sizeof(*tnrm2));
  if (!tnrm2)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    hipEventCreate(&tnrm2[i]);
  taxpy = malloc(2 * maxits * sizeof(*taxpy));
  if (!taxpy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * maxits; i++)
    hipEventCreate(&taxpy[i]);
  tcopy = malloc(2 * 2 * sizeof(*tcopy));
  if (!tcopy)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * 2; i++)
    hipEventCreate(&tcopy[i]);
  tallreduce = malloc(2 * (maxits + 1) * sizeof(*tallreduce));
  if (!tallreduce)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 1); i++)
    hipEventCreate(&tallreduce[i]);
  thalo = malloc(2 * (maxits + 2) * sizeof(*thalo));
  if (!thalo)
    return ACG_ERR_ERRNO;
  for (int i = 0; i < 2 * (maxits + 2); i++)
    hipEventCreate(&thalo[i]);
#endif

  /* warmup iterations for dot/allreduce */
  for (int i = 0; i < warmup; i++)
  {
    hipMemcpy(
        d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(
        d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_pdott, d_zero, sizeof(*d_pdott), hipMemcpyDeviceToDevice);
    err = hipblasDdot(
        hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
        d_bnrm2sqr);
    if (err)
      return ACG_ERR_HIPBLAS;
    if (commsize > 1)
      acgcomm_allreduce_hip(
          ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = hipblasDdot(
        hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r,
        1, d_rnrm2sqr);
    if (err)
      return ACG_ERR_HIPBLAS;
    if (commsize > 1)
      acgcomm_allreduce_hip(
          ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
    err = hipblasDdot(
        hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_p, 1, d_t,
        1, d_pdott);
    if (err)
      return ACG_ERR_HIPBLAS;
    if (commsize > 1)
      acgcomm_allreduce_hip(
          ACG_IN_PLACE, d_pdott, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
          NULL);
  }
  hipMemcpy(
      d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
  hipMemcpy(
      d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
  hipMemcpy(d_pdott, d_zero, sizeof(*d_pdott), hipMemcpyDeviceToDevice);

  /* warmup iterations for halo exchange/SpMV */
  for (int i = 0; i < warmup; i++)
  {
    if (commsize > 1)
    {
      err = hipStreamWaitEvent(commstream, xreadytosend, 0);
      if (err)
        return ACG_ERR_HIP;
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0,
          commstream);
      if (err)
        return err;
    }
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
        d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_hip_end(
          cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
          x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0,
          commstream);
      if (err)
        return err;
      err = hipEventRecord(xreceived, commstream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(stream, xreceived, 0);
      if (err)
        return ACG_ERR_HIP;
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
          vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
          d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
      err = hipEventRecord(xreadytosend, stream);
      if (err)
        return ACG_ERR_HIP;
    }

    if (commsize > 1)
    {
      err = hipStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
        return ACG_ERR_HIP;
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0,
          commstream);
      if (err)
        return err;
    }
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    if (commsize > 1)
    {
      err = acghalo_exchange_hip_end(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0,
          commstream);
      if (err)
        return err;
      err = hipEventRecord(preceived, commstream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(stream, preceived, 0);
      if (err)
        return ACG_ERR_HIP;
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
      err = hipEventRecord(preadytosend, stream);
      if (err)
        return ACG_ERR_HIP;
    }
  }

  /* warmup iterations for axpy */
  for (int i = 0; i < warmup; i++)
  {
    err = acgsolverhip_daxpy_alpha(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_zero, d_one, d_p,
        d_x);
    if (err)
      return err;
    err = acgsolverhip_daypx_beta(
        cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_one, d_one, d_p,
        d_r);
    if (err)
      return err;
  }

  /* warmup iterations for copy */
  for (int i = 0; i < warmup; i++)
  {
    err = hipblasDcopy(
        hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPBLAS;
    }
    err = hipblasDcopy(
        hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_p,
        1);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPBLAS;
    }
  }

  /* set scalars to infinity (needed to produce correct results on
   * the first call to acgsolverhip_pipelined_daxpy_fused) */
  err =
      hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpy(
      d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev),
      hipMemcpyDeviceToDevice);
  if (err)
    return ACG_ERR_HIP;

  double *d_M_inv;
  double *h_M_inv;
  hipsparseSpMatDescr_t matM_lower, matM_upper;
  hipsparseSpSVDescr_t spSVDescrL, spSVDescrU;
  size_t bufferSizeL, bufferSizeU;
  void *d_bufferL, *d_bufferU;
  err = hipsparseSpSV_createDescr(&spSVDescrU);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseSpSV_createDescr(&spSVDescrL);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  /**
   * Preconditioner setup
   */
  if (preconditioner == 1)
  {
    err = hipMalloc(&d_M_inv, (A->nprows) * sizeof(double));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
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

    err = hipMemcpy(d_M_inv, h_M_inv, (A->nprows) * sizeof(double), hipMemcpyHostToDevice);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }
    free(h_M_inv);
  }
  else if (preconditioner == 2)
  {
    hipsparseMatDescr_t matLU;
    acgidx_t *d_M_rowptr = d_rowptr;
    acgidx_t *d_M_colidx = d_colidx;
    double *d_M_values;
    hipsparseFillMode_t fill_lower = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseFillMode_t fill_upper = HIPSPARSE_FILL_MODE_UPPER;
    hipsparseDiagType_t diag_unit = HIPSPARSE_DIAG_TYPE_UNIT;
    hipsparseDiagType_t diag_nonunit = HIPSPARSE_DIAG_TYPE_NON_UNIT;

    err = hipMalloc(
        &d_M_values, A->fnpnzs * sizeof(*d_M_values));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }
    err = hipMemcpy(
        d_M_values, d_a, A->fnpnzs * sizeof(*d_M_values),
        hipMemcpyDeviceToDevice);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }

    // matM_lower
    err = hipsparseCreateCsr(
        &matM_lower, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
        d_M_colidx, d_M_values, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSpMatSetAttribute(
        matM_lower, HIPSPARSE_SPMAT_FILL_MODE, &fill_lower,
        sizeof(fill_lower));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSpMatSetAttribute(
        matM_lower, HIPSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }

    // matM_upper
    err = hipsparseCreateCsr(
        &matM_upper, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
        d_M_colidx, d_M_values, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSpMatSetAttribute(
        matM_upper, HIPSPARSE_SPMAT_FILL_MODE, &fill_upper,
        sizeof(fill_upper));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSpMatSetAttribute(
        matM_upper, HIPSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit,
        sizeof(diag_nonunit));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }

    // ILU factorization part
    csrilu02Info_t infoM = NULL;
    int bufferSizeLU = 0;
    void *d_bufferLU;
    err = hipsparseCreateMatDescr(&matLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSetMatType(matLU, HIPSPARSE_MATRIX_TYPE_GENERAL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSetMatIndexBase(matLU, HIPSPARSE_INDEX_BASE_ZERO);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseCreateCsrilu02Info(&infoM);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseDcsrilu02_bufferSize(
        hipsparse, A->nownedrows, A->fnpnzs, matLU,
        d_M_values, d_rowptr, d_colidx, infoM, &bufferSizeLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMalloc(&d_bufferLU, bufferSizeLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }

    err = hipsparseDcsrilu02_analysis(
        hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }

    int structural_zero;
    err = hipsparseXcsrilu02_zeroPivot(
        hipsparse, infoM, &structural_zero);
    if (structural_zero >= 0)
    {
      fprintf(stderr, "ACG: structural zero at index %d\n", structural_zero);
      // return ACG_ERR_HIPSPARSE;
    }
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     return ACG_ERR_HIPSPARSE;
    // }

    err = hipsparseDcsrilu02(
        hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values,
        d_rowptr, d_colidx, infoM, HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
        d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    int numerical_zero;
    err = hipsparseXcsrilu02_zeroPivot(
        hipsparse, infoM, &numerical_zero);
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     return ACG_ERR_HIPSPARSE;
    // }
    if (numerical_zero >= 0)
    {
      fprintf(stderr, "ACG: numerical zero at index %d\n", numerical_zero);
      // return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseDestroyCsrilu02Info(infoM);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseDestroyMatDescr(matLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipFree(d_bufferLU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }

    // lower

    err = hipsparseSpSV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT,
        spSVDescrL, &bufferSizeL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMalloc(&d_bufferL, bufferSizeL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }
    err = hipsparseSpSV_analysis(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrL,
        d_bufferL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMemset(d_y, 0x0, A->nownedrows * sizeof(*d_y));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }

    // upper
    err = hipsparseSpSV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT,
        spSVDescrU, &bufferSizeU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMalloc(&d_bufferU, bufferSizeU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }
    err = hipsparseSpSV_analysis(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrU,
        d_bufferU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMemset(d_u, 0x0, A->nownedrows * sizeof(*d_u));
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
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
  err = acgcomm_barrier_hip(stream, comm, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  gettime(&t0);

  /* compute right-hand side norm */
  double bnrm2sqr;
  acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
  err = hipblasDdot(
      hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
      d_bnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_HIPBLAS;
  }
  acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
  nnrm2++;
  cg->nnrm2++;
  cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
  cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
  if (commsize > 1)
  {
    acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    err = acgcomm_allreduce_hip(
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
  err = hipMemcpy(
      &bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
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
  err = hipblasDcopy(
      hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_HIPBLAS;
  }
  acgEventRecord(tcopy[2 * ncopy + 1], 0);
  ncopy++;
  cg->ncopy++;
  cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

  if (commsize > 1)
  {
    err = acghalo_exchange_hip_begin(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0,
        commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return err;
    }
  }
  acgEventRecord(tgemv[2 * ngemv + 0], 0);
  err = hipsparseSpMV(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  if (commsize > 1)
  {
    acgEventRecord(thalo[2 * nhalo + 0], 0);
    err = acghalo_exchange_hip_end(
        cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
        x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0,
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
    err = hipEventRecord(xreceived, commstream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIP;
    }
    err = hipStreamWaitEvent(stream, xreceived, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIP;
    }
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
        vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
        d_obuffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
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
    err = hipMemcpy(d_u, d_r, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r), hipMemcpyDeviceToDevice);

    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }
  }
  else if (preconditioner == 1)
  {
    acgsolverhip_apply_jacobi_preconditioner((A->nprows - A->nghostrows), d_M_inv, d_r, d_u, stream);
  }
  else if (preconditioner == 2)
  {

    err = hipsparseSpSV_solve(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
        vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseSpSV_solve(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
        vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
  }
  /* compute (r, u) */
  acgEventRecord(tdot[2 * ndot + 0], 0);
  err = hipblasDdot(
      hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_u,
      1, d_rnrm2sqr);
  if (err)
  {
    if (errcode)
      *errcode = err;
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_HIPBLAS;
  }
  acgEventRecord(tdot[2 * ndot + 1], 0);
  ndot++;
  cg->ndot++;
  cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
  cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) *
              (sizeof(*cg->p.x) + sizeof(*cg->t.x));

  if (commsize > 1)
  {
    if (comm->type == acgcomm_rccl || comm->type == acgcomm_rocshmem || comm->type == acgcomm_rccl_split)
    {

      acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
      err = acgcomm_allreduce_hip(
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
      hipStreamSynchronize(stream);
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

  err = hipMemcpy(
      rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  cg->rnrm2 = cg->r0nrm2 = sqrt(*rnrm2sqr);
  residualrtol *= cg->r0nrm2;

  err = hipblasDcopy(
      hipblas, cg->u->num_nonzeros - cg->u->num_ghost_nonzeros,
      d_u, 1, d_p, 1);
  if (err)
  {
    if (errcode)
    {
      *errcode = err;
    }
    return ACG_ERR_HIPBLAS;
  }

  err = hipEventRecord(preadytosend, stream);
  if (err)
  {
    gettime(&t1);
    cg->tsolve += elapsed(t0, t1);
    return ACG_ERR_HIP;
  }
  /* iterative solver loop */
  for (int k = 0; k < maxits; k++)
  {
    // SpMV
    /* compute t = Ap */
    if (commsize > 1)
    {
      err = hipStreamWaitEvent(commstream, preadytosend, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIP;
      }
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0,
          commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
      }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecp,
        d_zero, vect, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    if (commsize > 1)
    {
      acgEventRecord(thalo[2 * nhalo + 0], 0);
      err = acghalo_exchange_hip_end(
          cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_p, ACG_DOUBLE,
          cg->p.num_nonzeros, d_p, ACG_DOUBLE, comm, tag, errcode, 0,
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
      err = hipEventRecord(mreceived, commstream);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIP;
      }
      err = hipStreamWaitEvent(stream, mreceived, 0);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIP;
      }
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecpo,
          d_one, vecto, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
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
    err = hipblasDdot(
        hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_t, 1, d_p,
        1, d_delta);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tdot[2 * ndot + 1], 0);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

    if (commsize > 1)
    {
      if (comm->type == acgcomm_rccl || comm->type == acgcomm_rocshmem || comm->type == acgcomm_rccl_split)
      {

        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(
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
        hipStreamSynchronize(stream);
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
    acgsolverhip_preconditioned_daxpy_fused(
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
      err = hipMemcpy(
          d_u, d_r,
          (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r),
          hipMemcpyDeviceToDevice);

      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIP;
      }
    }
    else if (preconditioner == 1)
    {
      acgsolverhip_apply_jacobi_preconditioner((A->nprows - A->nghostrows),
                                               d_M_inv, d_r, d_u,
                                               stream);
    }
    else if (preconditioner == 2)
    {

      // err = hipMemset(d_y, 0x0, A->nownedrows * sizeof(*d_y));
      // if (err)
      // {
      //     if (errcode)
      //         *errcode = err;
      //     return ACG_ERR_HIP;
      // }
      // err = hipMemset(d_u, 0x0, A->nownedrows * sizeof(*d_y));
      // if (err)
      // {
      //     if (errcode)
      //         *errcode = err;
      //     return ACG_ERR_HIP;
      // }

      err = hipsparseSpSV_solve(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
          vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
      err = hipsparseSpSV_solve(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
          vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
    }

    // dot (r, u) -> gamma
    acgEventRecord(tdot[2 * ndot + 0], 0);
    err = hipblasDdot(
        hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_u,
        1, d_rnrm2sqr);
    if (err)
    {
      if (errcode)
        *errcode = err;
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tdot[2 * ndot + 1], 0);
    ndot++;
    cg->ndot++;
    cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
    cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

    if (commsize > 1)
    {
      if (comm->type == acgcomm_rccl || comm->type == acgcomm_rocshmem || comm->type == acgcomm_rccl_split)
      {

        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(
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
        hipStreamSynchronize(stream);
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

    err = hipEventRecord(rnrm2sqrready, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIP;
    }
    err = hipStreamWaitEvent(copystream, rnrm2sqrready, 0);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIP;
    }

    err = hipMemcpyAsync(rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr),
                         hipMemcpyDeviceToHost, copystream);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIP;
    }

    /* update search direction, p = βp + r, where β = (rₖ,rₖ)/(rₖ₋₁,rₖₖ₋₁)
     */
    acgEventRecord(taxpy[2 * naxpy + 0], 0);
    err = acgsolverhip_daypx_beta(
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
    err = hipEventRecord(preadytosend, stream);
    if (err)
    {
      gettime(&t1);
      cg->tsolve += elapsed(t0, t1);
      return ACG_ERR_HIP;
    }

    /* convergence tests */
    /* hipEventSynchronize(rnrm2sqrreceived); */
    hipStreamSynchronize(copystream);
    cg->rnrm2 = sqrt(*rnrm2sqr);
    if ((diffatol > 0 && cg->dxnrm2 < diffatol) || (diffrtol > 0 && cg->dxnrm2 < diffrtol) || (residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    {
      hipStreamSynchronize(stream);
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
    hipEventSynchronize(tgemv[2 * i + 1]);
    hipEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
    cg->tgemv += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ndot; i++)
  {
    hipEventSynchronize(tdot[2 * i + 1]);
    hipEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
    cg->tdot += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nnrm2; i++)
  {
    hipEventSynchronize(tnrm2[2 * i + 1]);
    hipEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
    cg->tnrm2 += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < naxpy; i++)
  {
    hipEventSynchronize(taxpy[2 * i + 1]);
    hipEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
    cg->taxpy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < ncopy; i++)
  {
    hipEventSynchronize(tcopy[2 * i + 1]);
    hipEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
    cg->tcopy += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nallreduce; i++)
  {
    hipEventSynchronize(tallreduce[2 * i + 1]);
    hipEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
    cg->tallreduce += 1.0e-3 * t;
  }
  for (acgidx_t i = 0; i < nhalo; i++)
  {
    hipEventSynchronize(thalo[2 * i + 1]);
    hipEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
    cg->thalo += 1.0e-3 * t;
  }
#endif

  /* copy solution back to host */
  err = hipMemcpy(
      x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;

  /* free hipsparse matrix and vectors */
  hipsparseDestroyDnVec(vecx);
  hipsparseDestroyDnVec(vecr);
  hipsparseDestroyDnVec(vecp);
  hipsparseDestroyDnVec(vect);
  hipsparseDestroyDnVec(vecw);
  hipsparseDestroyDnVec(vecz);
  hipsparseDestroyDnVec(vecn);
  hipsparseDestroyDnVec(vecm);
  if (commsize > 1)
  {
    hipsparseDestroyDnVec(vecxo);
    hipsparseDestroyDnVec(vecro);
    hipsparseDestroyDnVec(vecpo);
    hipsparseDestroyDnVec(vecto);
    hipsparseDestroyDnVec(vecno);
    hipsparseDestroyDnVec(vecmo);
  }
  hipsparseDestroySpMat(matA);
  hipFree(d_buffer);
  if (commsize > 1)
  {
    hipsparseDestroySpMat(matO);
    hipFree(d_obuffer);
  }
  hipFree(d_x);
  hipFree(d_b);
  hipFree(d_z);
  hipFree(d_w);
  hipFree(d_n);
  hipFree(d_m);
  hipFree(d_q);
  hipFree(d_u);
  //    hipFree(d_merged_dots);
  hipHostFree(rnrm2sqr);
  hipStreamDestroy(commstream);
  hipStreamDestroy(copystream);
  hipStreamDestroy(collective_stream);

  /* reset hipsparse and hipblas pointer modes */
  err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipblasSetPointerMode(hipblas, hipblaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPBLAS;
  }

  /* check for HIP errors */
  if (hipGetLastError() != hipSuccess)
    return ACG_ERR_HIP;

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

/**
 * ‘acgsolverhip_solve_pipelined_preconditioned()’ solves the given linear system,
 * Ax=b, using a pipelined conjugate gradient method. The linear
 * system may be distributed across multiple processes and
 * communication is handled using MPI.
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
int acgsolverhip_solve_pipelined_preconditioned(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
        err = hipMalloc((void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->q)
    {
        cg->q = malloc(sizeof(*cg->q));
        if (!cg->q)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->q, x);
        if (err)
            return err;
        err = hipMalloc((void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->z)
    {
        cg->z = malloc(sizeof(*cg->z));
        if (!cg->z)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->z, x);
        if (err)
            return err;
        err = hipMalloc((void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->m)
    {
        cg->m = malloc(sizeof(*cg->m));
        if (!cg->m)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->m, x);
        if (err)
            return err;
        err = hipMalloc(
            (void **)&cg->d_m, cg->m->num_nonzeros * sizeof(*cg->d_m));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->n)
    {
        cg->n = malloc(sizeof(*cg->n));
        if (!cg->n)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->n, x);
        if (err)
            return err;
        err = hipMalloc(
            (void **)&cg->d_n, cg->n->num_nonzeros * sizeof(*cg->d_n));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->u)
    {
        cg->u = malloc(sizeof(*cg->u));
        if (!cg->u)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->u, x);
        if (err)
            return err;
        err = hipMalloc(
            (void **)&cg->d_u, cg->u->num_nonzeros * sizeof(*cg->d_u));
        if (err)
            return ACG_ERR_HIP;
    }
    if (!cg->y)
    {
        cg->y = malloc(sizeof(*cg->y));
        if (!cg->y)
            return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->y, x);
        if (err)
            return err;
        err = hipMalloc(
            (void **)&cg->d_y, cg->y->num_nonzeros * sizeof(*cg->d_y));
        if (err)
            return ACG_ERR_HIP;
    }

    /* /\* If the stopping criterion is based on the difference in */
    /*  * solution from one iteration to the next, then allocate */
    /*  * additional storage for storing the difference. *\/ */
    /* if ((diffatol > 0 || diffrtol > 0) && !cg->dx) { */
    /*     cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO; */
    /*     int err = acgvector_init_copy(cg->dx, x); if (err) return err; */
    /* } */

    hipStream_t stream = 0;
    const struct acghalo *halo = cg->halo;
    double *d_bnrm2sqr = cg->d_bnrm2sqr;
    double *d_rnrm2sqr = &cg->d_rnrm2sqr[0];
    double *d_delta = &cg->d_rnrm2sqr[1];
    double *d_rnrm2sqr_prev = cg->d_rnrm2sqr_prev;
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
    double *d_m = cg->d_m;
    double *d_n = cg->d_n;
    double *d_u = cg->d_u;
    double *d_y = cg->d_y;
    acgidx_t *d_rowptr = cg->d_rowptr;
    acgidx_t *d_colidx = cg->d_colidx;
    double *d_a = cg->d_a;
    acgidx_t *d_orowptr = cg->d_orowptr;
    acgidx_t *d_ocolidx = cg->d_ocolidx;
    double *d_oa = cg->d_oa;
    MPI_Request request;

    /* configure hipblas and hipsparse to use device-side pointers */
    hipblasPointerMode_t hipblaspointermode;
    err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
    if (err)
        return ACG_ERR_HIPBLAS;
    err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
    if (err)
        return ACG_ERR_HIPBLAS;
    hipsparsePointerMode_t hipsparsepointermode;
    err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    hipStream_t collective_stream;
    err = hipStreamCreateWithFlags(&collective_stream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;

    double *rnrm2sqr;
    err = hipHostMalloc((void **)&rnrm2sqr, sizeof(*rnrm2sqr), hipHostMallocNumaUser);
    if (err)
        return ACG_ERR_HIP;
    hipStream_t copystream;
    err = hipStreamCreateWithFlags(&copystream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t rnrm2sqrready;
    hipEventCreateWithFlags(&rnrm2sqrready, hipEventDisableTiming);

    /* copy right-hand side and initial guess to device */
    double *d_b;
    err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;
    double *d_x;
    err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice);
    if (err)
        return ACG_ERR_HIP;

    /* used to overlap P2P communication with SpMV */
    hipStream_t commstream;
    err = hipStreamCreateWithFlags(&commstream, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t xreadytosend, xreceived;
    err = hipEventCreateWithFlags(&xreadytosend, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(xreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&xreceived, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t rreadytosend, rreceived;
    err = hipEventCreateWithFlags(&rreadytosend, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(rreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&rreceived, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t wreadytosend, wreceived;
    err = hipEventCreateWithFlags(&wreadytosend, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventRecord(wreadytosend, stream);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&wreceived, hipStreamNonBlocking);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t mreadytosend, mreceived, reduced;
    err = hipEventCreateWithFlags(&mreadytosend, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&mreceived, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&reduced, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t ureadytosend, ureceived;
    err = hipEventCreateWithFlags(&ureadytosend, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    err = hipEventCreateWithFlags(&ureceived, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;
    hipEvent_t dotEvent;
    err = hipEventCreateWithFlags(&dotEvent, hipEventDisableTiming);
    if (err)
        return ACG_ERR_HIP;

    /* create hipsparse matrix and vectors */
    hipsparseDnVecDescr_t vecx, vecr, vecw, vecq;
    err = hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecw, A->nownedrows, d_w, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecq, A->nownedrows, d_q, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    hipsparseDnVecDescr_t vecxo, vecro, vecwo, vecqo;
    if (commsize > 1)
    {
        err = hipsparseCreateDnVec(&vecxo, A->nborderrows + A->nghostrows, d_x + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecro, A->nborderrows + A->nghostrows, d_r + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(&vecqo, A->nborderrows + A->nghostrows, d_q + A->borderrowoffset, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }

    hipsparseDnVecDescr_t vecz, vecn, vecm, vecy, vecu;
    err = hipsparseCreateDnVec(&vecz, A->nownedrows, d_z, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseCreateDnVec(&vecm, A->nownedrows, d_m, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseCreateDnVec(&vecn, A->nownedrows, d_n, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    err = hipsparseCreateDnVec(&vecw, A->nownedrows, d_w, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecy, A->nownedrows, d_y, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseCreateDnVec(&vecu, A->nownedrows, d_u, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    hipsparseDnVecDescr_t vecno, vecmo, veczo, vecuo;
    if (commsize > 1)
    {
        err = hipsparseCreateDnVec(
            &vecno, A->nborderrows + A->nghostrows, d_n + A->borderrowoffset,
            HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(
            &vecmo, A->nborderrows + A->nghostrows, d_m + A->borderrowoffset,
            HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(
            &vecwo, A->nborderrows + A->nghostrows, d_w + A->borderrowoffset,
            HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(
            &veczo, A->nborderrows + A->nghostrows, d_z + A->borderrowoffset,
            HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseCreateDnVec(
            &vecuo, A->nborderrows + A->nghostrows, d_u + A->borderrowoffset,
            HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }

    hipsparseSpMatDescr_t matA;
    err = hipsparseCreateCsr(
        &matA, A->nownedrows, A->nownedrows, A->fnpnzs,
        d_rowptr, d_colidx, d_a,
        HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    size_t buffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    void *d_buffer;
    err = hipMalloc(&d_buffer, buffersize);
    if (err)
        return ACG_ERR_HIP;
    /* Note: Disable hipsparseSpMV_preprocess, because it degrades
     * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
    err = hipsparseSpMV_preprocess(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_HIPSPARSE; }
#endif

    hipsparseSpMatDescr_t matO;
    void *d_obuffer;
    if (commsize > 1)
    {
        err = hipsparseCreateCsr(
            &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows, A->onpnzs,
            d_orowptr, d_ocolidx, d_oa,
            HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
            HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        size_t obuffersize;
        err = hipsparseSpMV_bufferSize(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMalloc(&d_obuffer, obuffersize);
        if (err)
            return ACG_ERR_HIP;
        /* Note: Disable hipsparseSpMV_preprocess, because it degrades
         * performance by a factor of about 2x on LUMI. */
#if 0 && (hipsparseVersionMajor >= 3)
        err = hipsparseSpMV_preprocess(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_HIPSPARSE; }
#endif
    }

#if !defined(ACG_USE_HIPSPARSE)
    /* setup for merge-based SpMV */
    const int TASKS_PER_THREAD_MERGE = 10;
    acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
    acgidx_t nstartrows = (ntasks + TASKS_PER_THREAD_MERGE - 1) / TASKS_PER_THREAD_MERGE;
    acgidx_t *d_startrows;
    err = hipMalloc((void **)&d_startrows, nstartrows * sizeof(*d_startrows));
    if (err)
        return ACG_ERR_HIP;
    acgsolverhip_csrgemv_merge_startrows(
        A->nprows - A->nghostrows, d_rowptr, nstartrows, d_startrows, stream);
    if (hipPeekAtLastError())
        return ACG_ERR_HIP;
    hipStreamSynchronize(stream);
#endif

    /* create timing events for profiling */
    acgidx_t ngemv = 0, ndot = 0, nnrm2 = 0, naxpy = 0, ncopy = 0, nallreduce = 0, nhalo = 0;
    hipEvent_t *tgemv, *tdot, *tnrm2, *taxpy, *tcopy, *tallreduce, *thalo;
#if defined(ACG_ENABLE_PROFILING)
    tgemv = malloc(2 * (maxits + 2) * sizeof(*tgemv));
    if (!tgemv)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 2); i++)
        hipEventCreate(&tgemv[i]);
    tdot = malloc(2 * maxits * sizeof(*tdot));
    if (!tdot)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * maxits; i++)
        hipEventCreate(&tdot[i]);
    tnrm2 = malloc(2 * (maxits + 1) * sizeof(*tnrm2));
    if (!tnrm2)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 1); i++)
        hipEventCreate(&tnrm2[i]);
    taxpy = malloc(2 * maxits * sizeof(*taxpy));
    if (!taxpy)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * maxits; i++)
        hipEventCreate(&taxpy[i]);
    tcopy = malloc(2 * 1 * sizeof(*tcopy));
    if (!tcopy)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * 1; i++)
        hipEventCreate(&tcopy[i]);
    tallreduce = malloc(2 * (maxits + 1) * sizeof(*tallreduce));
    if (!tallreduce)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 1); i++)
        hipEventCreate(&tallreduce[i]);
    thalo = malloc(2 * (maxits + 2) * sizeof(*thalo));
    if (!thalo)
        return ACG_ERR_ERRNO;
    for (int i = 0; i < 2 * (maxits + 2); i++)
        hipEventCreate(&thalo[i]);
#endif

    /* warmup iterations for dot/allreduce */
    for (int i = 0; i < warmup; i++)
    {
        hipMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
        hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
        hipMemcpy(d_delta, d_zero, sizeof(*d_delta), hipMemcpyDeviceToDevice);
        err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        err = hipblasDdot(hipblas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1, d_rnrm2sqr);
        if (err)
            return ACG_ERR_HIPBLAS;
        err = hipblasDdot(hipblas, cg->w->num_nonzeros - cg->w->num_ghost_nonzeros, d_w, 1, d_r, 1, d_delta);
        if (err)
            return ACG_ERR_HIPBLAS;
        if (commsize > 1)
            acgcomm_allreduce_hip(ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM, stream, comm, NULL);
        hipStreamSynchronize(stream);
    }
    hipMemcpy(d_bnrm2sqr, d_zero, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
    hipMemcpy(d_delta, d_zero, sizeof(*d_delta), hipMemcpyDeviceToDevice);

    /* warmup iterations for halo exchange/SpMV */
    for (int i = 0; i < warmup; i++)
    {
        /* r = b-Ax */
        err = hipblasDcopy(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPBLAS;
        }
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, xreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
        #if defined(ACG_USE_HIPSPARSE)
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        #else
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
            return err;
        #endif
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                x->num_nonzeros, d_x, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(xreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, xreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            err = hipEventRecord(xreadytosend, stream);
            if (err)
                return ACG_ERR_HIP;
        }

        /* w = Ar */
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, rreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
        #if defined(ACG_USE_HIPSPARSE)
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecr, d_zero, vecw, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        #else
        hipMemset(d_w, 0, A->nownedrows * sizeof(*d_w));
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_w, d_r, d_rowptr, d_colidx, d_a, 1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
            return err;
        #endif
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                cg->r.num_nonzeros, d_r, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(rreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, rreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecro, d_one, vecwo, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
        }
        err = hipEventRecord(wreadytosend, stream);
        if (err)
            return ACG_ERR_HIP;

        /* q = Aw */
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, wreadytosend, 0);
            if (err)
                return ACG_ERR_HIP;
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
        }
        #if defined(ACG_USE_HIPSPARSE)
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecw, d_zero, vecq, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        #else
        hipMemset(d_q, 0, A->nownedrows * sizeof(*d_q));
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_q, d_w, d_rowptr, d_colidx, d_a, 1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
            return err;
        #endif
        if (commsize > 1)
        {
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                cg->w->num_nonzeros, d_w, ACG_DOUBLE,
                comm, tag, errcode, 0, commstream);
            if (err)
                return err;
            err = hipEventRecord(wreceived, commstream);
            if (err)
                return ACG_ERR_HIP;
            err = hipStreamWaitEvent(stream, wreceived, 0);
            if (err)
                return ACG_ERR_HIP;
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                d_one, matO, vecwo, d_one, vecqo, HIP_R_64F,
                HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
        }
    }

    /* warmup iterations for axpy */
    err = hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_z, 0, (cg->z->num_nonzeros - cg->z->num_ghost_nonzeros) * sizeof(*d_z));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_t, 0, (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_p, 0, (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*d_p));
    if (err)
        return ACG_ERR_HIP;
    for (int i = 0; i < warmup; i++)
    {
        err = hipMemcpy(d_rnrm2sqr, d_zero, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = hipMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = hipMemcpy(d_delta, d_inf, sizeof(*d_delta), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
        if (err)
            return ACG_ERR_HIP;
        err = acgsolverhip_pipelined_daxpy_fused(
            cg->t.num_nonzeros - cg->t.num_ghost_nonzeros,
            d_rnrm2sqr, d_rnrm2sqr_prev, d_delta,
            d_q, d_p, d_r, d_t, d_x, d_z, d_w, d_alpha, stream);
        if (err)
            return err;
    }
    err = hipMemset(d_r, 0, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_w, 0, (cg->w->num_nonzeros - cg->w->num_ghost_nonzeros) * sizeof(*d_w));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_q, 0, (cg->q->num_nonzeros - cg->q->num_ghost_nonzeros) * sizeof(*d_q));
    if (err)
        return ACG_ERR_HIP;

    /* set scalars to infinity (needed to produce correct results on
     * the first call to acgsolverhip_pipelined_daxpy_fused) */
    err = hipMemcpy(d_alpha, d_inf, sizeof(*d_alpha), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;
    err = hipMemcpy(d_rnrm2sqr_prev, d_inf, sizeof(*d_rnrm2sqr_prev), hipMemcpyDeviceToDevice);
    if (err)
        return ACG_ERR_HIP;

    /* set the vectors z, t and p to zero */
    err = hipMemset(d_z, 0, (cg->z->num_nonzeros - cg->z->num_ghost_nonzeros) * sizeof(*d_z));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_t, 0, (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_t));
    if (err)
        return ACG_ERR_HIP;
    err = hipMemset(d_p, 0, (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*d_p));
    if (err)
        return ACG_ERR_HIP;

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
    err = acgcomm_barrier_hip(stream, comm, errcode);
    if (err)
        return err;
    hipStreamSynchronize(stream);
    gettime(&t0);

    /* compute right-hand side norm */
    double bnrm2sqr;
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
    cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
    if (commsize > 1)
    {
        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(ACG_IN_PLACE, d_bnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
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
    err = hipMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;
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
    acgEventRecord(tcopy[2 * ncopy + 0], 0);
    err = hipblasDcopy(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tcopy[2 * ncopy + 1], 0);
    ncopy++;
    cg->ncopy++;
    cg->Bcopy += (b->num_nonzeros - b->num_ghost_nonzeros) * (sizeof(*cg->r.x) + sizeof(*b->x));

    if (commsize > 1)
    {
        err = acghalo_exchange_hip_begin(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    #if defined(ACG_USE_HIPSPARSE)
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F,
        HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    #else
    err = acgsolverhip_csrgemv_merge(
        A->nownedrows, d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0,
        nstartrows, d_startrows, stream);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
    }
    #endif
    if (commsize > 1)
    {
        acgEventRecord(thalo[2 * nhalo + 0], 0);
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            x->num_nonzeros, d_x, ACG_DOUBLE,
            comm, tag, errcode, 0, commstream);
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
        err = hipEventRecord(xreceived, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(stream, xreceived, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F,
            HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipEventRecord(rreadytosend, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 1], 0);
    ngemv++;
    cg->ngemv++;
    cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
    cg->Bgemv +=
        (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->r.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + x->num_nonzeros * sizeof(*x->x);

    /////
    double *d_diag, *d_M_inv;
    hipsparseSpMatDescr_t matM_lower, matM_upper;
    hipsparseSpSVDescr_t spSVDescrL, spSVDescrU;
    err = hipsparseSpSV_createDescr(&spSVDescrU);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipsparseSpSV_createDescr(&spSVDescrL);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }

    //////
    if (preconditioner == 0)
    {
        err = hipMemcpy(d_u, d_r, (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*d_r), hipMemcpyDeviceToDevice);

        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
    }
    if (preconditioner == 1)
    {

        hipMalloc(&d_diag, A->nownedrows * sizeof(double));
        hipMalloc(&d_M_inv, A->nownedrows * sizeof(double));

        acgsolverhip_jacobi_extract_diagonal(A->nownedrows, d_rowptr, d_colidx, d_a, d_diag, stream);

        acgsolverhip_jacobi_preconditioner(A->nownedrows, d_diag, d_M_inv, stream);

        acgsolverhip_apply_jacobi_preconditioner(A->nownedrows, d_M_inv, d_r, d_u, stream);
    }
    else if (preconditioner == 2)
    {
        // will add ILU

        fprintf(stderr, "\nUsing ILU preconditioner\n");

        hipsparseMatDescr_t matLU;
        acgidx_t *d_M_rowptr = d_rowptr;
        acgidx_t *d_M_colidx = d_colidx;
        double *d_M_values;
        hipsparseFillMode_t fill_lower = HIPSPARSE_FILL_MODE_LOWER;
        hipsparseFillMode_t fill_upper = HIPSPARSE_FILL_MODE_UPPER;
        hipsparseDiagType_t diag_unit = HIPSPARSE_DIAG_TYPE_UNIT;
        hipsparseDiagType_t diag_nonunit = HIPSPARSE_DIAG_TYPE_NON_UNIT;

        err = hipMalloc(
            &d_M_values, A->fnpnzs * sizeof(*d_M_values));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
        err = hipMemcpy(
            d_M_values, d_a, A->fnpnzs * sizeof(*d_M_values),
            hipMemcpyDeviceToDevice);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }

        // matM_lower
        err = hipsparseCreateCsr(
            &matM_lower, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
            d_M_colidx, d_M_values, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
            HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseSpMatSetAttribute(
            matM_lower, HIPSPARSE_SPMAT_FILL_MODE, &fill_lower,
            sizeof(fill_lower));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseSpMatSetAttribute(
            matM_lower, HIPSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }

        // matM_upper
        err = hipsparseCreateCsr(
            &matM_upper, A->nownedrows, A->nownedrows, A->fnpnzs, d_M_rowptr,
            d_M_colidx, d_M_values, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
            HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseSpMatSetAttribute(
            matM_upper, HIPSPARSE_SPMAT_FILL_MODE, &fill_upper,
            sizeof(fill_upper));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseSpMatSetAttribute(
            matM_upper, HIPSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit,
            sizeof(diag_nonunit));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }

        fprintf(stderr, "ILU factorization part\n");

        // ILU factorization part
        csrilu02Info_t infoM = NULL;
        int bufferSizeLU = 0;
        void *d_bufferLU;
        err = hipsparseCreateMatDescr(&matLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseSetMatType(matLU, HIPSPARSE_MATRIX_TYPE_GENERAL);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseSetMatIndexBase(matLU, HIPSPARSE_INDEX_BASE_ZERO);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }

        err = hipsparseCreateCsrilu02Info(&infoM);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }

        fprintf(stderr, "factorization part: bufferSize\n");

        err = hipsparseDcsrilu02_bufferSize(
            hipsparse, A->nownedrows, A->fnpnzs, matLU,
            d_M_values, d_rowptr, d_colidx, infoM, &bufferSizeLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMalloc(&d_bufferLU, bufferSizeLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }

        fprintf(stderr, "factorization part: analysis begins\n");

        err = hipsparseDcsrilu02_analysis(
            hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
            d_colidx, infoM, HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        fprintf(stderr, "factorization part: analysis ends\n");

        int structural_zero;
        err = hipsparseXcsrilu02_zeroPivot(
            hipsparse, infoM, &structural_zero);
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
        fprintf(stderr, "factorization part: end of zero pivot check\n");

        fprintf(stderr, "ACG: structural factorization begins\n");
        err = hipsparseDcsrilu02(
            hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values,
            d_rowptr, d_colidx, infoM, HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
            d_bufferLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        fprintf(stderr, "ACG: structural factorization done\n");
        int numerical_zero;
        err = hipsparseXcsrilu02_zeroPivot(
            hipsparse, infoM, &numerical_zero);
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

        err = hipsparseDestroyCsrilu02Info(infoM);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipsparseDestroyMatDescr(matLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipFree(d_bufferLU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
        fprintf(stderr, "ILU solver part:\n");

        size_t bufferSizeL, bufferSizeU;
        void *d_bufferL, *d_bufferU;

        err = hipsparseSpSV_bufferSize(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrL, &bufferSizeL);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMalloc(&d_bufferL, bufferSizeL);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
        err = hipsparseSpSV_analysis(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrL,
            d_bufferL);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMemset(d_y, 0x0, A->nownedrows * sizeof(*d_y));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
        err = hipsparseSpSV_solve(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower,
            vecr, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrL);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }

        // upper

        err = hipsparseSpSV_bufferSize(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
            vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT,
            spSVDescrU, &bufferSizeU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMalloc(&d_bufferU, bufferSizeU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
        err = hipsparseSpSV_analysis(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
            vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrU,
            d_bufferU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        err = hipMemset(d_u, 0x0, A->nownedrows * sizeof(*d_u));
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIP;
        }
        err = hipsparseSpSV_solve(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper,
            vecy, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spSVDescrU);
        if (err)
        {
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }

    err = hipEventRecord(ureadytosend, stream);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIP;
    }

    // buraya

    /* compute right-hand side norm */
    // double bnrm2sqr;
    acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    err = hipblasDdot(
        hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1,
        d_bnrm2sqr);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    nnrm2++;
    cg->nnrm2++;
    cg->nflops += 2 * (b->num_nonzeros - b->num_ghost_nonzeros);
    cg->Bnrm2 += (b->num_nonzeros - b->num_ghost_nonzeros) * sizeof(*b->x);
    if (commsize > 1)
    {

        acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
        err = acgcomm_allreduce_hip(
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
    err = hipMemcpy(
        &bnrm2sqr, d_bnrm2sqr, sizeof(*d_bnrm2sqr), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;
    cg->bnrm2 = sqrt(bnrm2sqr);
    cg->r0nrm2 = cg->bnrm2;
    cg->rnrm2 = *rnrm2sqr;
    *rnrm2sqr *= cg->bnrm2;

    /* compute initial search direction: p = z₀ */
    acgEventRecord(tcopy[2 * ncopy + 0], 0);
    err = hipblasDcopy(
        hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_u, 1, d_p, 1);
    if (err)
    {
        if (errcode)
            *errcode = err;
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return ACG_ERR_HIPBLAS;
    }
    acgEventRecord(tcopy[2 * ncopy + 1], 0);
    ncopy++;
    cg->ncopy++;
    cg->Bcopy += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->r.x));
    // err = cudaEventRecord(preadytosend, stream);
    // if (err)
    //     return ACG_ERR_CUDA;

    /* compute initial residual norm */
    // acgEventRecord(tnrm2[2 * nnrm2 + 0], 0);
    // err = cublasDdot(
    //     cublas, cg->r.num_nonzeros - cg->r.num_ghost_nonzeros, d_r, 1, d_r, 1,
    //     d_rnrm2sqr);
    // if (err)
    // {
    //     if (errcode)
    //         *errcode = err;
    //     gettime(&t1);
    //     cg->tsolve += elapsed(t0, t1);
    //     return ACG_ERR_CUBLAS;
    // }
    // acgEventRecord(tnrm2[2 * nnrm2 + 1], 0);
    // nnrm2++;
    // cg->nnrm2++;
    // cg->nflops += 2 * (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros);
    // cg->Bnrm2 +=
    //     (cg->r.num_nonzeros - cg->r.num_ghost_nonzeros) * sizeof(*cg->r.x);
    // if (commsize > 1)
    // {
    //     acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
    //     err = acgcomm_allreduce(
    //         ACG_IN_PLACE, d_rnrm2sqr, 1, ACG_DOUBLE, ACG_SUM, stream, comm,
    //         errcode);
    //     if (err)
    //     {
    //         gettime(&t1);
    //         cg->tsolve += elapsed(t0, t1);
    //         return err;
    //     }
    //     acgEventRecord(tallreduce[2 * nallreduce + 1], 0);
    //     nallreduce++;
    //     cg->nallreduce++;
    //     cg->Ballreduce += sizeof(*rnrm2sqr);
    // }
    // err = cudaMemcpy(
    //     rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), cudaMemcpyDeviceToHost);
    // if (err)
    //     return ACG_ERR_CUDA;
    // cg->rnrm2 = cg->r0nrm2 = sqrt(*rnrm2sqr);
    // residualrtol *= cg->r0nrm2;

    // /* initial convergence test */
    // if ((residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
    // {
    //     gettime(&t1);
    //     cg->tsolve += elapsed(t0, t1);
    //     return ACG_SUCCESS;
    // }

    // w = Au
    if (commsize > 1)
    {
        err = hipStreamWaitEvent(commstream, ureadytosend, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
        err = acghalo_exchange_hip_begin(
            cg->halo, cg->haloexchange, x->num_nonzeros, d_u, ACG_DOUBLE,
            x->num_nonzeros, d_u, ACG_DOUBLE, comm, tag, errcode, 0,
            commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
    }
    acgEventRecord(tgemv[2 * ngemv + 0], 0);
    #if defined(ACG_USE_HIPSPARSE)
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecu, d_one,
        vecw, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    #else
    err = acgsolverhip_csrgemv_merge(
        A->nownedrows, d_w, d_u, d_rowptr, d_colidx, d_a, 1.0, 1.0,
        nstartrows, d_startrows, stream);
    if (err)
    {
        gettime(&t1);
        cg->tsolve += elapsed(t0, t1);
        return err;
    }
    #endif
    if (commsize > 1)
    {
        acgEventRecord(thalo[2 * nhalo + 0], 0);
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange, x->num_nonzeros, d_u, ACG_DOUBLE,
            x->num_nonzeros, d_u, ACG_DOUBLE, comm, tag, errcode, 0,
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
        err = hipEventRecord(xreceived, commstream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipStreamWaitEvent(stream, xreceived, 0);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecuo,
            d_one, vecwo, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
    }
    if (rank == 0)
    {
        int ver, subver;
        MPI_Get_version(&ver, &subver);

        fprintf(stderr, "MPI Version: %d.%d\n", ver, subver);
    }

    /////

    /* iterative solver loop */
    for (int k = 0; k < maxits; k++)
    {
        // fprintf(stderr, "iter:%d\n", k);

        // dot
        /* compute (r, u) */
        acgEventRecord(tdot[2 * ndot + 0], 0);
        err = hipblasDdot(
            hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_r, 1, d_u,
            1, d_rnrm2sqr);
        if (err)
        {
            if (errcode)
                *errcode = err;
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
        acgEventRecord(tdot[2 * ndot + 1], 0);
        ndot++;
        cg->ndot++;
        cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
        cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

        // dot (w, u)
        acgEventRecord(tdot[2 * ndot + 0], 0);
        err = hipblasDdot(
            hipblas, cg->p.num_nonzeros - cg->p.num_ghost_nonzeros, d_w, 1, d_u,
            1, d_delta);
        if (err)
        {
            if (errcode)
                *errcode = err;
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIPBLAS;
        }
        acgEventRecord(tdot[2 * ndot + 1], 0);
        ndot++;
        cg->ndot++;
        cg->nflops += 2 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros);
        cg->Bdot += (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * (sizeof(*cg->p.x) + sizeof(*cg->t.x));

        if (comm->type == acgcomm_rccl || comm->type == acgcomm_rccl_split || comm->type == acgcomm_rocshmem)
        {
            err = hipEventRecord(dotEvent, stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
        }

        if (commsize > 1)
        {
            if (comm->type == acgcomm_rccl || comm->type == acgcomm_rccl_split || comm->type == acgcomm_rocshmem)
            {
                err = hipStreamWaitEvent(collective_stream, dotEvent, 0);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return err;
                }
                acgEventRecord(tallreduce[2 * nallreduce + 0], 0);
                err = acgcomm_allreduce_hip(
                    ACG_IN_PLACE, d_rnrm2sqr, 2, ACG_DOUBLE, ACG_SUM,
                    collective_stream, comm, errcode);
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
                hipStreamSynchronize(stream);
                /**
                 * TODO: will change the following part. That will go to the
                 * acgcomm_allreduce() method once I find a clear way of doing it.
                 **/
                err = MPI_Iallreduce(MPI_IN_PLACE, d_rnrm2sqr, 2, MPI_DOUBLE, MPI_SUM, comm->mpicomm, &request);
                if (err)
                {
                    if (errcode)
                        *errcode = err;
                    return ACG_ERR_MPI;
                }
            }

            nallreduce++;
            cg->nallreduce++;
            cg->Ballreduce += sizeof(*d_rnrm2sqr);
        }

        if (comm->type == acgcomm_rccl || comm->type == acgcomm_rccl_split || comm->type == acgcomm_rocshmem)
        {

            err = hipMemcpyAsync(
                rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost, collective_stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }

            err = hipEventRecord(reduced, collective_stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
        }

        // preconditioning
        if (preconditioner == 0)
        {
            err = hipMemcpy(d_m, d_w,
                (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros) * sizeof(*d_w),
                hipMemcpyDeviceToDevice);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIP;
            }
        }
        else if (preconditioner == 1)
        {
            acgsolverhip_apply_jacobi_preconditioner(A->nownedrows, d_M_inv, d_w, d_m, stream);
        }
        else if (preconditioner == 2)
        {
#ifdef ACG_HAVE_HYPRE
//            apply_ilu_preconditioner(ilu_precond, parcsr_A, d_w, d_m, A->nownedrows);
#endif
        }

        err = hipEventRecord(mreadytosend, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return ACG_ERR_HIP;
        }

        // SpMV
        /* compute n = Am */
        if (commsize > 1)
        {
            err = hipStreamWaitEvent(commstream, mreadytosend, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = acghalo_exchange_hip_begin(
                cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_m, ACG_DOUBLE,
                cg->p.num_nonzeros, d_m, ACG_DOUBLE, comm, tag, errcode, 0,
                commstream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
        }
        acgEventRecord(tgemv[2 * ngemv + 0], 0);
        #if defined(ACG_USE_HIPSPARSE)
        err = hipsparseSpMV(
            hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecm,
            d_zero, vecn, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            if (errcode)
                *errcode = err;
            return ACG_ERR_HIPSPARSE;
        }
        #else
        hipMemset(d_n, 0, A->nownedrows * sizeof(*d_n));
        err = acgsolverhip_csrgemv_merge(
            A->nownedrows, d_n, d_m, d_rowptr, d_colidx, d_a, 1.0, 1.0,
            nstartrows, d_startrows, stream);
        if (err)
        {
            gettime(&t1);
            cg->tsolve += elapsed(t0, t1);
            return err;
        }
        #endif
        if (commsize > 1)
        {
            acgEventRecord(thalo[2 * nhalo + 0], 0);
            err = acghalo_exchange_hip_end(
                cg->halo, cg->haloexchange, cg->p.num_nonzeros, d_m, ACG_DOUBLE,
                cg->p.num_nonzeros, d_m, ACG_DOUBLE, comm, tag, errcode, 0,
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
            err = hipEventRecord(mreceived, commstream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipStreamWaitEvent(stream, mreceived, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecmo,
                d_one, vecno, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
        }
        acgEventRecord(tgemv[2 * ngemv + 1], 0);
        ngemv++;
        cg->ngemv++;
        cg->nflops += 3 * (int64_t)(A->fnpnzs + A->onpnzs);
        cg->Bgemv += (int64_t)(A->fnpnzs + A->onpnzs) * (sizeof(*A->fa) + sizeof(*A->fcolidx)) + A->nownedrows * (sizeof(*A->frowptr) + sizeof(*cg->t.x)) + (A->nborderrows + A->nghostrows) * sizeof(*A->orowptr) + cg->p.num_nonzeros * sizeof(*cg->p.x);

        if (comm->type == acgcomm_rccl || comm->type == acgcomm_rccl_split || comm->type == acgcomm_rocshmem)
        {
            err = hipStreamWaitEvent(stream, reduced, 0);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
        }
        else if (comm->type == acgcomm_mpi)
        {

            acgcomm_wait(comm, errcode, &request);

            err = hipMemcpy(
                rnrm2sqr, d_rnrm2sqr, sizeof(*d_rnrm2sqr), hipMemcpyDeviceToHost);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }
        }

        /* wait for host to receive updated residual norm */
        hipStreamSynchronize(stream);
        cg->rnrm2 = sqrt(*rnrm2sqr);
        // fprintf(stderr, "cg->rnrm2: %f\n", cg->rnrm2);
        // if (k == 0)
        // {
        //     cg->r0nrm2 = cg->rnrm2;
        //     residualrtol *= cg->r0nrm2;
        // }

        /* convergence tests */
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) || (diffrtol > 0 && cg->dxnrm2 < diffrtol) || (residualatol > 0 && cg->rnrm2 < residualatol) || (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            hipStreamSynchronize(stream);
            converged = true;
            break;
        }

        /* update vectors */
        acgEventRecord(taxpy[2 * naxpy + 0], 0);
        err = acgsolverhip_preconditioned_pipelined_daxpy_fused(
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
        acgEventRecord(taxpy[2 * naxpy + 1], 0);
        naxpy++;
        cg->naxpy++;
        cg->nflops += 16 * (cg->t.num_nonzeros - cg->t.num_ghost_nonzeros);
        cg->Baxpy += 8 * (cg->p.num_nonzeros - cg->p.num_ghost_nonzeros) * sizeof(*cg->p.x);

        // Implemented as part of residual replecament strategy as mentioned in the original paper
        if (k % 100 == 0)
        {
            // r = b - Ax
            err = hipblasDcopy(
                hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_r, 1);
            if (err)
            {
                if (errcode)
                    *errcode = err;
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIPBLAS;
            }

            if (commsize > 1)
            {
                err = acghalo_exchange_hip_begin(
                    cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
                    x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0,
                    commstream);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return err;
                }
            }
            #if defined(ACG_USE_HIPSPARSE)
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
                d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            #else
            err = acgsolverhip_csrgemv_merge(
                A->nownedrows, d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0,
                nstartrows, d_startrows, stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
            #endif
            if (commsize > 1)
            {
                err = acghalo_exchange_hip_end(
                    cg->halo, cg->haloexchange, x->num_nonzeros, d_x, ACG_DOUBLE,
                    x->num_nonzeros, d_x, ACG_DOUBLE, comm, tag, errcode, 0,
                    commstream);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return err;
                }
                err = hipEventRecord(xreceived, commstream);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return ACG_ERR_HIP;
                }
                err = hipStreamWaitEvent(stream, xreceived, 0);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return ACG_ERR_HIP;
                }
                err = hipsparseSpMV(
                    hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO,
                    vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT,
                    d_obuffer);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    if (errcode)
                        *errcode = err;
                    return ACG_ERR_HIPSPARSE;
                }
            }

            // preconditioner, u = (M^-1)r

            // preconditioning
            if (preconditioner == 1)
            {
                acgsolverhip_apply_jacobi_preconditioner(A->nownedrows, d_M_inv, d_r, d_u, stream);
            }
            else if (preconditioner == 2)
            {
#ifdef ACG_HAVE_HYPRE
                apply_ilu_preconditioner(ilu_precond, parcsr_A, d_w, d_m, A->nownedrows);
#endif
            }

            err = hipEventRecord(ureadytosend, stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return ACG_ERR_HIP;
            }

            // w = Au
            if (commsize > 1)
            {
                err = hipStreamWaitEvent(commstream, ureadytosend, 0);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return err;
                }
                err = acghalo_exchange_hip_begin(
                    cg->halo, cg->haloexchange, x->num_nonzeros, d_u, ACG_DOUBLE,
                    x->num_nonzeros, d_u, ACG_DOUBLE, comm, tag, errcode, 0,
                    commstream);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return err;
                }
            }
            acgEventRecord(tgemv[2 * ngemv + 0], 0);
            #if defined(ACG_USE_HIPSPARSE)
            err = hipsparseSpMV(
                hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecu, d_one,
                vecw, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                if (errcode)
                    *errcode = err;
                return ACG_ERR_HIPSPARSE;
            }
            #else
            err = acgsolverhip_csrgemv_merge(
                A->nownedrows, d_w, d_u, d_rowptr, d_colidx, d_a, 1.0, 1.0,
                nstartrows, d_startrows, stream);
            if (err)
            {
                gettime(&t1);
                cg->tsolve += elapsed(t0, t1);
                return err;
            }
            #endif
            if (commsize > 1)
            {
                acgEventRecord(thalo[2 * nhalo + 0], 0);
                err = acghalo_exchange_hip_end(
                    cg->halo, cg->haloexchange, x->num_nonzeros, d_u, ACG_DOUBLE,
                    x->num_nonzeros, d_u, ACG_DOUBLE, comm, tag, errcode, 0,
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
                err = hipEventRecord(xreceived, commstream);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return ACG_ERR_HIP;
                }
                err = hipStreamWaitEvent(stream, xreceived, 0);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    return ACG_ERR_HIP;
                }
                err = hipsparseSpMV(
                    hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecuo,
                    d_one, vecwo, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
                if (err)
                {
                    gettime(&t1);
                    cg->tsolve += elapsed(t0, t1);
                    if (errcode)
                        *errcode = err;
                    return ACG_ERR_HIPSPARSE;
                }
            }
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
        hipEventSynchronize(tgemv[2 * i + 1]);
        hipEventElapsedTime(&t, tgemv[2 * i + 0], tgemv[2 * i + 1]);
        cg->tgemv += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < ndot; i++)
    {
        hipEventSynchronize(tdot[2 * i + 1]);
        hipEventElapsedTime(&t, tdot[2 * i + 0], tdot[2 * i + 1]);
        cg->tdot += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nnrm2; i++)
    {
        hipEventSynchronize(tnrm2[2 * i + 1]);
        hipEventElapsedTime(&t, tnrm2[2 * i + 0], tnrm2[2 * i + 1]);
        cg->tnrm2 += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < naxpy; i++)
    {
        hipEventSynchronize(taxpy[2 * i + 1]);
        hipEventElapsedTime(&t, taxpy[2 * i + 0], taxpy[2 * i + 1]);
        cg->taxpy += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < ncopy; i++)
    {
        hipEventSynchronize(tcopy[2 * i + 1]);
        hipEventElapsedTime(&t, tcopy[2 * i + 0], tcopy[2 * i + 1]);
        cg->tcopy += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nallreduce; i++)
    {
        hipEventSynchronize(tallreduce[2 * i + 1]);
        hipEventElapsedTime(&t, tallreduce[2 * i + 0], tallreduce[2 * i + 1]);
        cg->tallreduce += 1.0e-3 * t;
    }
    for (acgidx_t i = 0; i < nhalo; i++)
    {
        hipEventSynchronize(thalo[2 * i + 1]);
        hipEventElapsedTime(&t, thalo[2 * i + 0], thalo[2 * i + 1]);
        cg->thalo += 1.0e-3 * t;
    }
#endif

    /* copy solution back to host */
    err = hipMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
    if (err)
        return ACG_ERR_HIP;

    /* free hipsparse matrix and vectors */
    hipsparseDestroyDnVec(vecx);
    hipsparseDestroyDnVec(vecr);
    hipsparseDestroyDnVec(vecw);
    hipsparseDestroyDnVec(vecq);
    hipsparseDestroyDnVec(vecz);
    hipsparseDestroyDnVec(vecn);
    hipsparseDestroyDnVec(vecm);
    hipsparseDestroyDnVec(vecy);
    hipsparseDestroyDnVec(vecu);
    if (commsize > 1)
    {
        hipsparseDestroyDnVec(vecxo);
        hipsparseDestroyDnVec(vecro);
        hipsparseDestroyDnVec(vecwo);
        hipsparseDestroyDnVec(vecqo);
        hipsparseDestroyDnVec(vecno);
        hipsparseDestroyDnVec(vecmo);
        hipsparseDestroyDnVec(veczo);
        hipsparseDestroyDnVec(vecuo);
    }
    hipsparseDestroySpMat(matA);
    hipFree(d_buffer);
    if (commsize > 1)
    {
        hipsparseDestroySpMat(matO);
        hipFree(d_obuffer);
    }
#if !defined(ACG_USE_HIPSPARSE)
    hipFree(d_startrows);
#endif
    hipFree(d_x);
    hipFree(d_b);
    hipHostFree(rnrm2sqr);
    hipStreamDestroy(copystream);
    hipStreamDestroy(commstream);
    hipStreamDestroy(collective_stream);

    /* reset hipsparse and hipblas pointer modes */
    err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPSPARSE;
    }
    err = hipblasSetPointerMode(hipblas, hipblaspointermode);
    if (err)
    {
        if (errcode)
            *errcode = err;
        return ACG_ERR_HIPBLAS;
    }

    /* check for HIP errors */
    if (hipGetLastError() != hipSuccess)
        return ACG_ERR_HIP;

    /* if the solver converged or the only stopping criteria is a
     * maximum number of iterations, then the solver succeeded */
    if (converged)
        return ACG_SUCCESS;
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
}

/*
 * ω update selector for BiCGStab:
 *   0 -> plain, ω = (t,s)/(t,t)
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
    hipStream_t stream,
    hipStream_t collective_stream,
    hipEvent_t dotEvent,
    int *errcode)
{
  int err;
  if (!(commsize > 1 && !nocomm_allreduce))
    return ACG_SUCCESS;
  if (comm->type == acgcomm_rccl_split)
  {
    hipEventRecord(dotEvent, stream);
    hipStreamWaitEvent(collective_stream, dotEvent, 0);
    err = acgcomm_allreduce_hip(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, collective_stream,
        comm, errcode);
    if (err)
      return err;
    hipStreamSynchronize(collective_stream);
  }
  else if (comm->type == acgcomm_rccl || comm->type == acgcomm_rocshmem)
  {
    err = acgcomm_allreduce_hip(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, stream, comm, errcode);
    if (err)
      return err;
  }
  else if (comm->type == acgcomm_mpi)
  {
    hipStreamSynchronize(stream);
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
 * solvers: the diagonal block honors the ACG_USE_HIPSPARSE guard (hipSPARSE
 * or the merge-based backend), while the off-diagonal (border) block always
 * uses hipSPARSE with an accumulate (beta=1), matching the established HIP
 * solvers.
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
    hipStream_t stream,
    hipStream_t commstream,
    int nnz_full,
    double *d_in,
    double *d_out,
    hipEvent_t inreadytosend,
    hipEvent_t inreceived,
    double *d_one,
    double *d_zero,
    hipsparseHandle_t hipsparse,
    hipsparseSpMatDescr_t matA,
    hipsparseSpMatDescr_t matO,
    hipsparseDnVecDescr_t vecin,
    hipsparseDnVecDescr_t vecout,
    hipsparseDnVecDescr_t vecino,
    hipsparseDnVecDescr_t vecouto,
    void *d_buffer,
    void *d_obuffer,
    acgidx_t *d_rowptr,
    acgidx_t *d_colidx,
    double *d_a,
    acgidx_t nstartrows,
    acgidx_t *d_startrows)
{
  int err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = hipEventRecord(inreadytosend, stream);
    if (err)
      return ACG_ERR_HIP;
    err = hipStreamWaitEvent(commstream, inreadytosend, 0);
    if (err)
      return ACG_ERR_HIP;
    err = acghalo_exchange_hip_begin(
        halo, haloexchange, nnz_full, d_in, ACG_DOUBLE, nnz_full, d_in,
        ACG_DOUBLE, comm, tag, errcode, 0, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_HIPSPARSE)
  err = hipsparseSpMV(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecin, d_zero,
      vecout, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
#else
  err = acgsolverhip_csrgemv_merge(
      (A->nprows - A->nghostrows), d_out, d_in, d_rowptr, d_colidx, d_a, 1.0,
      0.0, nstartrows, d_startrows, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_hip_end(
          halo, haloexchange, nnz_full, d_in, ACG_DOUBLE, nnz_full, d_in,
          ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
      err = hipEventRecord(inreceived, commstream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(stream, inreceived, 0);
      if (err)
        return ACG_ERR_HIP;
    }
    /* off-diagonal (border-row) product: always hipSPARSE, accumulate */
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecino, d_one,
        vecouto, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
  }
  return ACG_SUCCESS;
}

/*
 * ‘pbicgstab_reduce_begin()’ starts an in-place sum-allreduce of a small
 * device buffer that is meant to be overlapped with a subsequent SpMV. For
 * the RCCL communicators the reduction is enqueued on a dedicated
 * collective stream (after waiting for the producing dot products on the
 * compute stream); for MPI a non-blocking MPI_Iallreduce is launched. It is
 * a no-op for a single process or when collectives are disabled.
 */
static int pbicgstab_reduce_begin(
    double *d_buf,
    int count,
    struct acgcomm *comm,
    int commsize,
    int nocomm_allreduce,
    hipStream_t stream,
    hipStream_t collective_stream,
    hipEvent_t dotEvent,
    MPI_Request *request,
    int *errcode)
{
  int err;
  if (!(commsize > 1 && !nocomm_allreduce))
    return ACG_SUCCESS;
  if (comm->type == acgcomm_rccl_split || comm->type == acgcomm_rccl ||
      comm->type == acgcomm_rocshmem)
  {
    err = hipEventRecord(dotEvent, stream);
    if (err)
      return ACG_ERR_HIP;
    err = hipStreamWaitEvent(collective_stream, dotEvent, 0);
    if (err)
      return ACG_ERR_HIP;
    err = acgcomm_allreduce_hip(
        ACG_IN_PLACE, d_buf, count, ACG_DOUBLE, ACG_SUM, collective_stream,
        comm, errcode);
    if (err)
      return err;
  }
  else if (comm->type == acgcomm_mpi)
  {
    hipStreamSynchronize(stream);
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
    hipStream_t stream,
    hipStream_t collective_stream,
    hipEvent_t redEvent,
    MPI_Request *request)
{
  int err;
  if (!(commsize > 1 && !nocomm_allreduce))
    return ACG_SUCCESS;
  if (comm->type == acgcomm_rccl_split || comm->type == acgcomm_rccl ||
      comm->type == acgcomm_rocshmem)
  {
    err = hipEventRecord(redEvent, collective_stream);
    if (err)
      return ACG_ERR_HIP;
    err = hipStreamWaitEvent(stream, redEvent, 0);
    if (err)
      return ACG_ERR_HIP;
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
static hipStream_t pbicgstab_reduce_ready_stream(
    const struct acgcomm *comm,
    int commsize,
    int nocomm_allreduce,
    hipStream_t stream,
    hipStream_t collective_stream)
{
  if (commsize > 1 && !nocomm_allreduce)
  {
    if (comm->type == acgcomm_rccl_split || comm->type == acgcomm_rccl ||
        comm->type == acgcomm_rocshmem)
      return collective_stream;
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
 * ‘acgsolverhip_pipelined_bicgstab_scalars()’ updates in place.
 */
static int pbicgstab_scalars_copy_begin(
    double *h_scalars,
    int nscalars,
    const double *d_rr,
    const double *d_omega,
    const double *d_rho,
    hipStream_t srcstream,
    hipStream_t copystream,
    hipEvent_t srcEvent,
    hipEvent_t copiedEvent)
{
  int err = hipEventRecord(srcEvent, srcstream);
  if (err)
    return ACG_ERR_HIP;
  err = hipStreamWaitEvent(copystream, srcEvent, 0);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      &h_scalars[0], d_rr, sizeof(*d_rr), hipMemcpyDeviceToHost, copystream);
  if (err)
    return ACG_ERR_HIP;
  if (nscalars > 1)
  {
    err = hipMemcpyAsync(
        &h_scalars[1], d_omega, sizeof(*d_omega), hipMemcpyDeviceToHost,
        copystream);
    if (err)
      return ACG_ERR_HIP;
    err = hipMemcpyAsync(
        &h_scalars[2], d_rho, sizeof(*d_rho), hipMemcpyDeviceToHost,
        copystream);
    if (err)
      return ACG_ERR_HIP;
  }
  err = hipEventRecord(copiedEvent, copystream);
  if (err)
    return ACG_ERR_HIP;
  return ACG_SUCCESS;
}

/**
 * ‘acgsolverhip_solve_preconditioned_bicgstab()’ solves the given linear
 * system, Ax=b, using a preconditioned stabilised bi-conjugate gradient
 * (BiCGStab) method. The linear system may be distributed across multiple
 * processes and communication is handled using MPI. The preconditioner is
 * selected by ‘preconditioner’: 0 -> none, 1 -> Jacobi, 2 -> ILU(0).
 *
 * The solver infrastructure (HIP streams, halo exchange, SpMV and the
 * hipSPARSE/hipBLAS handles) mirrors the pipelined preconditioned CG solver.
 */
int acgsolverhip_solve_preconditioned_bicgstab(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
    err = hipMalloc((void **)&cg->d_w, cg->w->num_nonzeros * sizeof(*cg->d_w));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->q)
  {
    cg->q = malloc(sizeof(*cg->q));
    if (!cg->q)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->q, x);
    if (err)
      return err;
    err = hipMalloc((void **)&cg->d_q, cg->q->num_nonzeros * sizeof(*cg->d_q));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->z)
  {
    cg->z = malloc(sizeof(*cg->z));
    if (!cg->z)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->z, x);
    if (err)
      return err;
    err = hipMalloc((void **)&cg->d_z, cg->z->num_nonzeros * sizeof(*cg->d_z));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->m)
  {
    cg->m = malloc(sizeof(*cg->m));
    if (!cg->m)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->m, x);
    if (err)
      return err;
    err = hipMalloc((void **)&cg->d_m, cg->m->num_nonzeros * sizeof(*cg->d_m));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->n)
  {
    cg->n = malloc(sizeof(*cg->n));
    if (!cg->n)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->n, x);
    if (err)
      return err;
    err = hipMalloc((void **)&cg->d_n, cg->n->num_nonzeros * sizeof(*cg->d_n));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->u)
  {
    cg->u = malloc(sizeof(*cg->u));
    if (!cg->u)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->u, x);
    if (err)
      return err;
    err = hipMalloc((void **)&cg->d_u, cg->u->num_nonzeros * sizeof(*cg->d_u));
    if (err)
      return ACG_ERR_HIP;
  }
  if (!cg->y)
  {
    cg->y = malloc(sizeof(*cg->y));
    if (!cg->y)
      return ACG_ERR_ERRNO;
    int err = acgvector_init_copy(cg->y, x);
    if (err)
      return err;
    err = hipMalloc((void **)&cg->d_y, cg->y->num_nonzeros * sizeof(*cg->d_y));
    if (err)
      return ACG_ERR_HIP;
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
  err = hipMalloc((void **)&d_rho, sizeof(*d_rho));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_rho_prev, sizeof(*d_rho_prev));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_omega, sizeof(*d_omega));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_rhat_v, sizeof(*d_rhat_v));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_nrm2sqr, sizeof(*d_nrm2sqr));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_dots, 2 * sizeof(*d_dots));
  if (err)
    return ACG_ERR_HIP;
  double *d_dot_num = &d_dots[0]; /* (t,s) or (M⁻¹t,M⁻¹s) */
  double *d_dot_den = &d_dots[1]; /* (t,t) or (M⁻¹t,M⁻¹t) */

  int leastPriority, greatestPriority;
  err = hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_HIP;

  hipStream_t stream;
  err = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t collective_stream;
  err = hipStreamCreateWithPriority(&collective_stream, hipStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t commstream;
  err = hipStreamCreateWithPriority(&commstream, hipStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_HIP;

  /* configure hipblas and hipsparse to use device-side pointers */
  hipblasPointerMode_t hipblaspointermode;
  err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_HIPBLAS;
  hipsparsePointerMode_t hipsparsepointermode;
  err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  hipEvent_t dotEvent;
  hipEventCreateWithFlags(&dotEvent, hipEventDisableTiming);
  hipEvent_t yreadytosend, yreceived, zreadytosend, zreceived, xreadytosend, xreceived;
  hipEventCreateWithFlags(&yreadytosend, hipEventDisableTiming);
  hipEventCreateWithFlags(&yreceived, hipEventDisableTiming);
  hipEventCreateWithFlags(&zreadytosend, hipEventDisableTiming);
  hipEventCreateWithFlags(&zreceived, hipEventDisableTiming);
  hipEventCreateWithFlags(&xreadytosend, hipEventDisableTiming);
  hipEventCreateWithFlags(&xreceived, hipEventDisableTiming);

  /* copy right-hand side and initial guess to device */
  double *d_b, *d_x;
  err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;

  /* create hipsparse dense vectors and matrices (created unconditionally;
   * the off-diagonal border product always uses hipSPARSE) */
  hipsparseDnVecDescr_t vecx, vecr, vecp, vecn, vecq, vect, vecy, vecz, vecm, vecu;
  hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
  hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
  hipsparseCreateDnVec(&vecp, A->nownedrows, d_p, HIP_R_64F);
  hipsparseCreateDnVec(&vecn, A->nownedrows, d_n, HIP_R_64F);
  hipsparseCreateDnVec(&vecq, A->nownedrows, d_q, HIP_R_64F);
  hipsparseCreateDnVec(&vect, A->nownedrows, d_t, HIP_R_64F);
  hipsparseCreateDnVec(&vecy, A->nownedrows, d_y, HIP_R_64F);
  hipsparseCreateDnVec(&vecz, A->nownedrows, d_z, HIP_R_64F);
  hipsparseCreateDnVec(&vecm, A->nownedrows, d_m, HIP_R_64F);
  hipsparseCreateDnVec(&vecu, A->nownedrows, d_u, HIP_R_64F);
  hipsparseDnVecDescr_t vecxo = NULL, vecro = NULL, vecyo = NULL, vecqo = NULL,
                        veczo = NULL, vecto = NULL;
  if (commsize > 1)
  {
    int no = A->nborderrows + A->nghostrows;
    hipsparseCreateDnVec(&vecxo, no, d_x + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecro, no, d_r + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecyo, no, d_y + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecqo, no, d_q + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&veczo, no, d_z + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecto, no, d_t + A->borderrowoffset, HIP_R_64F);
  }
  hipsparseSpMatDescr_t matA;
  err = hipsparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      HIPSPARSE_IDX_T, HIPSPARSE_IDX_T, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  size_t buffersize;
  err = hipsparseSpMV_bufferSize(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  void *d_buffer;
  err = hipMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_HIP;
  hipsparseSpMatDescr_t matO = NULL;
  void *d_obuffer = NULL;
  if (commsize > 1)
  {
    err = hipsparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows,
        A->onpnzs, d_orowptr, d_ocolidx, d_oa, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    size_t obuffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo,
        d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
    err = hipMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_HIP;
  }

#if !defined(ACG_USE_HIPSPARSE)
  /* setup for merge-based SpMV (diagonal block) */
  const int TASKS_PER_THREAD = 10;
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows_v = (ntasks + TASKS_PER_THREAD - 1) / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = hipMalloc((void **)&d_startrows, nstartrows_v * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_HIP;
  err = acgsolverhip_csrgemv_merge_startrows(
      (A->nprows - A->nghostrows), d_rowptr, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
  err = hipStreamSynchronize(stream);
  if (err)
    return ACG_ERR_HIP;
#else
  acgidx_t nstartrows_v = 0;
  acgidx_t *d_startrows = NULL;
#endif

  /*
   * Preconditioner setup (mirrors the pipelined preconditioned solver).
   */
  double *d_M_inv = NULL;
  hipsparseSpMatDescr_t matM_lower = NULL, matM_upper = NULL;
  double *d_M_values = NULL;
  hipsparseSpSVDescr_t spL_p = NULL, spU_y = NULL, spL_s = NULL, spU_z = NULL,
                       spL_t = NULL, spU_u = NULL;
  void *d_bufL_p = NULL, *d_bufU_y = NULL, *d_bufL_s = NULL, *d_bufU_z = NULL,
       *d_bufL_t = NULL, *d_bufU_u = NULL;

  if (preconditioner == 1)
  {
    err = hipMalloc(&d_M_inv, (A->nprows) * sizeof(double));
    if (err)
      return ACG_ERR_HIP;
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
    err = hipMemcpyAsync(d_M_inv, h_M_inv, (A->nprows) * sizeof(double), hipMemcpyHostToDevice, stream);
    if (!err)
      err = hipStreamSynchronize(stream); /* ‘h_M_inv’ is freed below */
    free(h_M_inv);
    if (err)
      return ACG_ERR_HIP;
  }
  else if (preconditioner == 2)
  {
    hipsparseMatDescr_t matLU;
    hipsparseFillMode_t fill_lower = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseFillMode_t fill_upper = HIPSPARSE_FILL_MODE_UPPER;
    hipsparseDiagType_t diag_unit = HIPSPARSE_DIAG_TYPE_UNIT;
    hipsparseDiagType_t diag_nonunit = HIPSPARSE_DIAG_TYPE_NON_UNIT;

    err = hipMalloc(&d_M_values, A->fnpnzs * sizeof(*d_M_values));
    if (err)
      return ACG_ERR_HIP;
    err = hipMemcpyAsync(d_M_values, d_a, A->fnpnzs * sizeof(*d_M_values), hipMemcpyDeviceToDevice, stream);
    if (err)
      return ACG_ERR_HIP;

    err = hipsparseCreateCsr(
        &matM_lower, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr,
        d_colidx, d_M_values, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipsparseSpMatSetAttribute(matM_lower, HIPSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower));
    hipsparseSpMatSetAttribute(matM_lower, HIPSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit));

    err = hipsparseCreateCsr(
        &matM_upper, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr,
        d_colidx, d_M_values, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipsparseSpMatSetAttribute(matM_upper, HIPSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper));
    hipsparseSpMatSetAttribute(matM_upper, HIPSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit, sizeof(diag_nonunit));

    /* incomplete-LU factorisation, in place on d_M_values */
    csrilu02Info_t infoM = NULL;
    int bufferSizeLU = 0;
    void *d_bufferLU;
    hipsparseCreateMatDescr(&matLU);
    hipsparseSetMatType(matLU, HIPSPARSE_MATRIX_TYPE_GENERAL);
    hipsparseSetMatIndexBase(matLU, HIPSPARSE_INDEX_BASE_ZERO);
    hipsparseCreateCsrilu02Info(&infoM);
    err = hipsparseDcsrilu02_bufferSize(
        hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, &bufferSizeLU);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipMalloc(&d_bufferLU, bufferSizeLU);
    if (err)
      return ACG_ERR_HIP;
    err = hipsparseDcsrilu02_analysis(
        hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
      return ACG_ERR_HIPSPARSE;
    int structural_zero;
    if (hipsparseXcsrilu02_zeroPivot(hipsparse, infoM, &structural_zero) == HIPSPARSE_STATUS_ZERO_PIVOT)
      fprintf(stderr, "ACG: structural zero at index %d\n", structural_zero);
    err = hipsparseDcsrilu02(
        hipsparse, A->nownedrows, A->fnpnzs, matLU, d_M_values, d_rowptr,
        d_colidx, infoM, HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferLU);
    if (err)
      return ACG_ERR_HIPSPARSE;
    int numerical_zero;
    if (hipsparseXcsrilu02_zeroPivot(hipsparse, infoM, &numerical_zero) == HIPSPARSE_STATUS_ZERO_PIVOT)
      fprintf(stderr, "ACG: numerical zero at index %d\n", numerical_zero);
    hipsparseDestroyCsrilu02Info(infoM);
    hipsparseDestroyMatDescr(matLU);
    hipFree(d_bufferLU);

    /*
     * Create and analyse a separate triangular-solve descriptor for each
     * (matrix, input, output) site used in the iteration:
     *   ŷ = M⁻¹p : (L) p->m, (U) m->y
     *   ẑ = M⁻¹s : (L) s->m, (U) m->z
     *   M⁻¹t     : (L) t->m, (U) m->u   (only for the preconditioned ω)
     */
    size_t bsz;
    hipsparseSpSV_createDescr(&spL_p);
    hipsparseSpSV_createDescr(&spU_y);
    hipsparseSpSV_createDescr(&spL_s);
    hipsparseSpSV_createDescr(&spU_z);
    err = hipsparseSpSV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecp, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_p, &bsz);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipMalloc(&d_bufL_p, bsz);
    err = hipsparseSpSV_analysis(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecp, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_p, d_bufL_p);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipsparseSpSV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_y, &bsz);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipMalloc(&d_bufU_y, bsz);
    err = hipsparseSpSV_analysis(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_y, d_bufU_y);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipsparseSpSV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecn, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_s, &bsz);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipMalloc(&d_bufL_s, bsz);
    err = hipsparseSpSV_analysis(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecn, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_s, d_bufL_s);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipsparseSpSV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecz, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_z, &bsz);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipMalloc(&d_bufU_z, bsz);
    err = hipsparseSpSV_analysis(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecz, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_z, d_bufU_z);
    if (err)
      return ACG_ERR_HIPSPARSE;
#if ACG_BICGSTAB_OMEGA_PRECONDITIONED
    hipsparseSpSV_createDescr(&spL_t);
    hipsparseSpSV_createDescr(&spU_u);
    err = hipsparseSpSV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vect, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_t, &bsz);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipMalloc(&d_bufL_t, bsz);
    err = hipsparseSpSV_analysis(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vect, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_t, d_bufL_t);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipsparseSpSV_bufferSize(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_u, &bsz);
    if (err)
      return ACG_ERR_HIPSPARSE;
    hipMalloc(&d_bufU_u, bsz);
    err = hipsparseSpSV_analysis(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_u, d_bufU_u);
    if (err)
      return ACG_ERR_HIPSPARSE;
#endif
  }

  err = hipblasSetStream(hipblas, stream);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipsparseSetStream(hipsparse, stream);
  if (err)
    return ACG_ERR_HIPSPARSE;

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
  err = acgcomm_barrier_hip(stream, comm, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  hipStreamSynchronize(commstream);
  hipStreamSynchronize(collective_stream);
  gettime(&t0);

  /* ‖b‖₂ */
  err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  double bnrm2sqr;
  err = hipMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(bnrm2sqr), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* r₀ = b − A·x₀ */
  err = acgsolverhip_dcopy(b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
  if (err)
    return err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = hipEventRecord(xreadytosend, stream);
    if (err)
      return ACG_ERR_HIP;
    err = hipStreamWaitEvent(commstream, xreadytosend, 0);
    if (err)
      return ACG_ERR_HIP;
    err = acghalo_exchange_hip_begin(
        cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x,
        ACG_DOUBLE, comm, tag, errcode, 0, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_HIPSPARSE)
  err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
    return ACG_ERR_HIPSPARSE;
#else
  err = acgsolverhip_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_hip_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
      err = hipEventRecord(xreceived, commstream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(stream, xreceived, 0);
      if (err)
        return ACG_ERR_HIP;
    }
    /* off-diagonal (border-row) product: always hipSPARSE, accumulate */
    err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
      return ACG_ERR_HIPSPARSE;
  }
  cg->ngemv++;

  /* r̂ = r₀ (shadow residual) */
  err = acgsolverhip_dcopy(n_owned, d_w, d_r, stream);
  if (err)
    return err;

  /* p = 0, v = 0 */
  err = hipMemsetAsync(d_p, 0, nnz_full * sizeof(*d_p), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_q, 0, nnz_full * sizeof(*d_q), stream);
  if (err)
    return ACG_ERR_HIP;

  /* scalars: ρ_prev = ω = α = 1 */
  err = hipMemcpyAsync(d_rho_prev, d_one, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(d_omega, d_one, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(d_alpha, d_one, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;

  /* ‖r₀‖₂ */
  err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, d_nrm2sqr);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = bicgstab_allreduce(d_nrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  double nrm2sqr;
  err = hipMemcpy(&nrm2sqr, d_nrm2sqr, sizeof(nrm2sqr), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
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
    err = hipblasDdot(hipblas, n_owned, d_w, 1, d_r, 1, d_rho);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = bicgstab_allreduce(d_rho, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* p = r + β(p − ω·v),  β = (ρ/ρ_prev)(α/ω) */
    err = acgsolverhip_bicgstab_p_update(n_owned, k, d_rho, d_rho_prev, d_alpha, d_omega, d_p, d_r, d_q, stream);
    if (err)
      return err;

    /* ŷ = M⁻¹p */
    if (preconditioner == 0)
    {
      err = hipMemcpyAsync(d_y, d_p, n_owned * sizeof(*d_y), hipMemcpyDeviceToDevice, stream);
      if (err)
        return ACG_ERR_HIP;
    }
    else if (preconditioner == 1)
    {
      err = acgsolverhip_apply_jacobi_preconditioner(n_owned, d_M_inv, d_p, d_y, stream);
      if (err)
        return err;
    }
    else if (preconditioner == 2)
    {
      hipMemsetAsync(d_m, 0, A->nownedrows * sizeof(*d_m), stream);
      hipMemsetAsync(d_y, 0, A->nownedrows * sizeof(*d_y), stream);
      err = hipsparseSpSV_solve(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecp, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_p);
      if (err)
        return ACG_ERR_HIPSPARSE;
      err = hipsparseSpSV_solve(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecy, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_y);
      if (err)
        return ACG_ERR_HIPSPARSE;
    }

    /* v = A·ŷ */
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        stream, commstream, nnz_full, d_y, d_q, yreadytosend, yreceived,
        d_one, d_zero, hipsparse, matA, matO, vecy, vecq, vecyo, vecqo, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
    if (err)
      return err;
    cg->ngemv++;

    /* (r̂, v) */
    err = hipblasDdot(hipblas, n_owned, d_w, 1, d_q, 1, d_rhat_v);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = bicgstab_allreduce(d_rhat_v, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* α = ρ/(r̂,v),  s = r − α·v */
    err = acgsolverhip_bicgstab_s_update(n_owned, d_alpha, d_rho, d_rhat_v, d_n, d_r, d_q, stream);
    if (err)
      return err;

    /* ‖s‖₂ — half-step convergence test */
    err = hipblasDdot(hipblas, n_owned, d_n, 1, d_n, 1, d_nrm2sqr);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = bicgstab_allreduce(d_nrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;
    hipStreamSynchronize(stream);
    err = hipMemcpy(&nrm2sqr, d_nrm2sqr, sizeof(nrm2sqr), hipMemcpyDeviceToHost);
    if (err)
      return ACG_ERR_HIP;
    double snrm2 = sqrt(nrm2sqr);
    cg->ndot++;

    if ((residualatol > 0 && snrm2 < residualatol) || (residualrtol > 0 && snrm2 < residualrtol))
    {
      /* x = x + α·ŷ */
      err = acgsolverhip_bicgstab_x_halfstep(n_owned, d_alpha, d_x, d_y, stream);
      if (err)
        return err;
      hipStreamSynchronize(stream);
      cg->rnrm2 = snrm2;
      cg->ntotaliterations++;
      cg->niterations++;
      converged = true;
      break;
    }

    /* ẑ = M⁻¹s */
    if (preconditioner == 0)
    {
      err = hipMemcpyAsync(d_z, d_n, n_owned * sizeof(*d_z), hipMemcpyDeviceToDevice, stream);
      if (err)
        return ACG_ERR_HIP;
    }
    else if (preconditioner == 1)
    {
      err = acgsolverhip_apply_jacobi_preconditioner(n_owned, d_M_inv, d_n, d_z, stream);
      if (err)
        return err;
    }
    else if (preconditioner == 2)
    {
      hipMemsetAsync(d_m, 0, A->nownedrows * sizeof(*d_m), stream);
      hipMemsetAsync(d_z, 0, A->nownedrows * sizeof(*d_z), stream);
      err = hipsparseSpSV_solve(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vecn, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_s);
      if (err)
        return ACG_ERR_HIPSPARSE;
      err = hipsparseSpSV_solve(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecz, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_z);
      if (err)
        return ACG_ERR_HIPSPARSE;
    }

    /* t = A·ẑ */
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        stream, commstream, nnz_full, d_z, d_t, zreadytosend, zreceived,
        d_one, d_zero, hipsparse, matA, matO, vecz, vect, veczo, vecto, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
    if (err)
      return err;
    cg->ngemv++;

    /* ω numerator/denominator */
#if ACG_BICGSTAB_OMEGA_PRECONDITIONED
    /* M⁻¹t into d_u, then ω = (M⁻¹t,M⁻¹s)/(M⁻¹t,M⁻¹t) with M⁻¹s = ẑ = d_z */
    if (preconditioner == 0)
    {
      err = hipMemcpyAsync(d_u, d_t, n_owned * sizeof(*d_u), hipMemcpyDeviceToDevice, stream);
      if (err)
        return ACG_ERR_HIP;
    }
    else if (preconditioner == 1)
    {
      err = acgsolverhip_apply_jacobi_preconditioner(n_owned, d_M_inv, d_t, d_u, stream);
      if (err)
        return err;
    }
    else if (preconditioner == 2)
    {
      hipMemsetAsync(d_m, 0, A->nownedrows * sizeof(*d_m), stream);
      hipMemsetAsync(d_u, 0, A->nownedrows * sizeof(*d_u), stream);
      err = hipsparseSpSV_solve(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_lower, vect, vecm, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spL_t);
      if (err)
        return ACG_ERR_HIPSPARSE;
      err = hipsparseSpSV_solve(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matM_upper, vecm, vecu, HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spU_u);
      if (err)
        return ACG_ERR_HIPSPARSE;
    }
    err = hipblasDdot(hipblas, n_owned, d_u, 1, d_z, 1, d_dot_num);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_u, 1, d_u, 1, d_dot_den);
    if (err)
      return ACG_ERR_HIPBLAS;
#else
    /* plain ω = (t,s)/(t,t) */
    err = hipblasDdot(hipblas, n_owned, d_t, 1, d_n, 1, d_dot_num);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_t, 1, d_t, 1, d_dot_den);
    if (err)
      return ACG_ERR_HIPBLAS;
#endif
    err = bicgstab_allreduce(d_dots, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* ω = num/den,  x += α·ŷ + ω·ẑ,  r = s − ω·t,  ρ_prev = ρ */
    err = acgsolverhip_bicgstab_xr_update(n_owned, d_omega, d_rho_prev, d_dot_num, d_dot_den, d_rho, d_alpha, d_x, d_y, d_z, d_r, d_n, d_t, stream);
    if (err)
      return err;

    /* ‖r‖₂ */
    err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, d_nrm2sqr);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = bicgstab_allreduce(d_nrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;
    hipStreamSynchronize(stream);
    err = hipMemcpy(&nrm2sqr, d_nrm2sqr, sizeof(nrm2sqr), hipMemcpyDeviceToHost);
    if (err)
      return ACG_ERR_HIP;
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
      err = hipMemcpy(&omega, d_omega, sizeof(omega), hipMemcpyDeviceToHost);
      if (err)
        return ACG_ERR_HIP;
      err = hipMemcpy(&rho_prev, d_rho_prev, sizeof(rho_prev), hipMemcpyDeviceToHost);
      if (err)
        return ACG_ERR_HIP;
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
  /* Drain the compute stream before stopping the clock, so that the reported
   * time covers all of the work and the solution copied out below is complete.
   * The solver's streams are non-blocking, so the copy on the null stream is
   * not ordered against them. */
  hipStreamSynchronize(stream);
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

  /* copy solution back to host */
  err = hipMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;

  hipsparseDestroyDnVec(vecx);
  hipsparseDestroyDnVec(vecr);
  hipsparseDestroyDnVec(vecp);
  hipsparseDestroyDnVec(vecn);
  hipsparseDestroyDnVec(vecq);
  hipsparseDestroyDnVec(vect);
  hipsparseDestroyDnVec(vecy);
  hipsparseDestroyDnVec(vecz);
  hipsparseDestroyDnVec(vecm);
  hipsparseDestroyDnVec(vecu);
  if (commsize > 1)
  {
    hipsparseDestroyDnVec(vecxo);
    hipsparseDestroyDnVec(vecro);
    hipsparseDestroyDnVec(vecyo);
    hipsparseDestroyDnVec(vecqo);
    hipsparseDestroyDnVec(veczo);
    hipsparseDestroyDnVec(vecto);
  }
  hipsparseDestroySpMat(matA);
  hipFree(d_buffer);
  if (commsize > 1)
  {
    hipsparseDestroySpMat(matO);
    hipFree(d_obuffer);
  }
  if (preconditioner == 2)
  {
    if (matM_lower)
      hipsparseDestroySpMat(matM_lower);
    if (matM_upper)
      hipsparseDestroySpMat(matM_upper);
    if (spL_p)
      hipsparseSpSV_destroyDescr(spL_p);
    if (spU_y)
      hipsparseSpSV_destroyDescr(spU_y);
    if (spL_s)
      hipsparseSpSV_destroyDescr(spL_s);
    if (spU_z)
      hipsparseSpSV_destroyDescr(spU_z);
    if (spL_t)
      hipsparseSpSV_destroyDescr(spL_t);
    if (spU_u)
      hipsparseSpSV_destroyDescr(spU_u);
    hipFree(d_bufL_p);
    hipFree(d_bufU_y);
    hipFree(d_bufL_s);
    hipFree(d_bufU_z);
    hipFree(d_bufL_t);
    hipFree(d_bufU_u);
    hipFree(d_M_values);
  }
#if !defined(ACG_USE_HIPSPARSE)
  hipFree(d_startrows);
#endif
  if (preconditioner == 1)
    hipFree(d_M_inv);

  hipFree(d_x);
  hipFree(d_b);
  hipFree(d_rho);
  hipFree(d_rho_prev);
  hipFree(d_omega);
  hipFree(d_rhat_v);
  hipFree(d_nrm2sqr);
  hipFree(d_dots);
  hipEventDestroy(dotEvent);
  hipEventDestroy(yreadytosend);
  hipEventDestroy(yreceived);
  hipEventDestroy(zreadytosend);
  hipEventDestroy(zreceived);
  hipEventDestroy(xreadytosend);
  hipEventDestroy(xreceived);
  hipStreamDestroy(stream);
  hipStreamDestroy(commstream);
  hipStreamDestroy(collective_stream);

  /* reset hipsparse and hipblas pointer modes */
  err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipblasSetPointerMode(hipblas, hipblaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPBLAS;
  }

  if (hipGetLastError() != hipSuccess)
    return ACG_ERR_HIP;

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
 * ‘acgsolverhip_solve_pipelined_bicgstab()’ solves the given linear system,
 * Ax=b, using the communication-hiding pipelined BiCGStab method (Cools &
 * Vanroose, 2017), without a preconditioner. The linear system may be
 * distributed across multiple processes; the two global reductions per
 * iteration are overlapped with the two sparse matrix-vector products.
 */
int acgsolverhip_solve_pipelined_bicgstab(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
      err = hipMalloc((void **)devptrs[i], (*vecptrs[i])->num_nonzeros * sizeof(double));
      if (err)
        return ACG_ERR_HIP;
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
  err = hipMalloc((void **)&d_alpha, sizeof(*d_alpha));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_beta, sizeof(*d_beta));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_omega, sizeof(*d_omega));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_rho, sizeof(*d_rho));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_R1, 2 * sizeof(*d_R1));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_R2, 5 * sizeof(*d_R2));
  if (err)
    return ACG_ERR_HIP;
  double *d_qy = &d_R1[0]; /* (q,y) */
  double *d_yy = &d_R1[1]; /* (y,y) */
  double *d_d1 = &d_R2[0]; /* (r̂0,r) */
  double *d_d2 = &d_R2[1]; /* (r̂0,w) */
  double *d_d3 = &d_R2[2]; /* (r̂0,s) */
  double *d_d4 = &d_R2[3]; /* (r̂0,z) */
  double *d_rr = &d_R2[4]; /* (r,r) for the residual norm */

  int leastPriority, greatestPriority;
  err = hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t stream;
  err = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t collective_stream;
  err = hipStreamCreateWithPriority(&collective_stream, hipStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t commstream;
  err = hipStreamCreateWithPriority(&commstream, hipStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_HIP;

  /* configure hipblas and hipsparse to use device-side pointers */
  hipblasPointerMode_t hipblaspointermode;
  err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_HIPBLAS;
  hipsparsePointerMode_t hipsparsepointermode;
  err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  hipEvent_t dotEvent, redEvent, haloReady, haloRecv;
  hipEventCreateWithFlags(&dotEvent, hipEventDisableTiming);
  hipEventCreateWithFlags(&redEvent, hipEventDisableTiming);
  hipEventCreateWithFlags(&haloReady, hipEventDisableTiming);
  hipEventCreateWithFlags(&haloRecv, hipEventDisableTiming);
  MPI_Request request;

  /* a dedicated stream for fetching the handful of scalars that the host needs
   * for the convergence tests, together with pinned staging memory for ‖r‖₂²,
   * ω and ρ, so that those transfers neither block nor are blocked by the work
   * queued on the compute stream */
  hipStream_t copystream;
  err = hipStreamCreateWithFlags(&copystream, hipStreamNonBlocking);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t scalarsReduced, scalarsCopied;
  hipEventCreateWithFlags(&scalarsReduced, hipEventDisableTiming);
  hipEventCreateWithFlags(&scalarsCopied, hipEventDisableTiming);
  double *h_scalars;
  err = hipHostMalloc(
      (void **)&h_scalars, 3 * sizeof(*h_scalars), hipHostMallocNumaUser);
  if (err)
    return ACG_ERR_HIP;

  /* copy right-hand side and initial guess to device */
  double *d_b, *d_x;
  err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;

  /* create hipsparse dense vectors and matrices (created unconditionally;
   * the off-diagonal border product always uses hipSPARSE) */
  hipsparseDnVecDescr_t vecx, vecr, vecw, vect, vecz, vecv;
  hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
  hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
  hipsparseCreateDnVec(&vecw, A->nownedrows, d_w, HIP_R_64F);
  hipsparseCreateDnVec(&vect, A->nownedrows, d_t, HIP_R_64F);
  hipsparseCreateDnVec(&vecz, A->nownedrows, d_z, HIP_R_64F);
  hipsparseCreateDnVec(&vecv, A->nownedrows, d_v, HIP_R_64F);
  hipsparseDnVecDescr_t vecxo = NULL, vecro = NULL, vecwo = NULL, vecto = NULL,
                        veczo = NULL, vecvo = NULL;
  if (commsize > 1)
  {
    int no = A->nborderrows + A->nghostrows;
    hipsparseCreateDnVec(&vecxo, no, d_x + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecro, no, d_r + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecwo, no, d_w + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecto, no, d_t + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&veczo, no, d_z + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecvo, no, d_v + A->borderrowoffset, HIP_R_64F);
  }
  hipsparseSpMatDescr_t matA;
  err = hipsparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      HIPSPARSE_IDX_T, HIPSPARSE_IDX_T, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
  if (err)
    return ACG_ERR_HIPSPARSE;
  size_t buffersize;
  err = hipsparseSpMV_bufferSize(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
    return ACG_ERR_HIPSPARSE;
  void *d_buffer;
  err = hipMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_HIP;
  hipsparseSpMatDescr_t matO = NULL;
  void *d_obuffer = NULL;
  if (commsize > 1)
  {
    err = hipsparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows,
        A->onpnzs, d_orowptr, d_ocolidx, d_oa, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
      return ACG_ERR_HIPSPARSE;
    size_t obuffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo,
        d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_HIP;
  }

#if !defined(ACG_USE_HIPSPARSE)
  /* setup for merge-based SpMV (diagonal block) */
  const int TASKS_PER_THREAD = 10;
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows_v = (ntasks + TASKS_PER_THREAD - 1) / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = hipMalloc((void **)&d_startrows, nstartrows_v * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_HIP;
  err = acgsolverhip_csrgemv_merge_startrows(
      (A->nprows - A->nghostrows), d_rowptr, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
  err = hipStreamSynchronize(stream);
  if (err)
    return ACG_ERR_HIP;
#else
  acgidx_t nstartrows_v = 0;
  acgidx_t *d_startrows = NULL;
#endif

  err = hipblasSetStream(hipblas, stream);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipsparseSetStream(hipsparse, stream);
  if (err)
    return ACG_ERR_HIPSPARSE;

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
    err = hipMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_HIP;
  }
  double *d_scratchscalars[] = {d_alpha, d_beta, d_omega, d_rho};
  for (int i = 0; i < (int)(sizeof(d_scratchscalars) / sizeof(*d_scratchscalars)); i++)
  {
    err = hipMemcpyAsync(
        d_scratchscalars[i], d_one, sizeof(**d_scratchscalars),
        hipMemcpyDeviceToDevice, stream);
    if (err)
      return ACG_ERR_HIP;
  }


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
    err = acgsolverhip_dcopy(n_owned, d_r, d_b, stream);
    if (err)
      return err;
    err = hipblasDdot(hipblas, n_owned, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* steps 1-5: the p,s,z and q,y recurrences */
    err = acgsolverhip_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;
    err = acgsolverhip_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* the first reduction, overlapped with v = A·z */
    err = hipblasDdot(hipblas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
        d_one, d_zero, hipsparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
    if (err)
      return err;
    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11, with ω taken from ‘d_one’ rather than from the reduction */
    err = acgsolverhip_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_one, d_one, d_alpha, d_p, d_q, d_y, d_t, d_v, d_rhat, d_r, d_w, stream);
    if (err)
      return err;

    /* the second reduction, overlapped with t = A·w */
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = pbicgstab_reduce_begin(d_R2, 5, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
        d_one, d_zero, hipsparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
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
    err = hipStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_HIP;

    /* steps 14-15, again fed from ‘d_one’ to keep β and α at one */
    err = acgsolverhip_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_one, d_one, d_one, d_one, stream);
    if (err)
      return err;

    err = hipStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_HIP;
  }

  /* discard what the warmup left behind; the scalars and the owned parts of
   * the vectors are assigned again by the initialisation below, and the ghost
   * parts by the halo exchange preceding each SpMV that reads them */
  for (int i = 0; i < nscratchvecs; i++)
  {
    err = hipMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_HIP;
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
  err = acgcomm_barrier_hip(stream, comm, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  hipStreamSynchronize(commstream);
  hipStreamSynchronize(collective_stream);
  hipStreamSynchronize(copystream);
  gettime(&t0);

  /* ‖b‖₂ */
  err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  double bnrm2sqr;
  err = hipMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(bnrm2sqr), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* r₀ = b − A·x₀ */
  err = acgsolverhip_dcopy(b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
  if (err)
    return err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = hipEventRecord(haloReady, stream);
    if (err)
      return ACG_ERR_HIP;
    err = hipStreamWaitEvent(commstream, haloReady, 0);
    if (err)
      return ACG_ERR_HIP;
    err = acghalo_exchange_hip_begin(
        cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x,
        ACG_DOUBLE, comm, tag, errcode, 0, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_HIPSPARSE)
  err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
    return ACG_ERR_HIPSPARSE;
#else
  err = acgsolverhip_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_hip_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
      err = hipEventRecord(haloRecv, commstream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(stream, haloRecv, 0);
      if (err)
        return ACG_ERR_HIP;
    }
    /* off-diagonal (border-row) product: always hipSPARSE, accumulate */
    err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
      return ACG_ERR_HIPSPARSE;
  }
  cg->ngemv++;

  /* r̂0 = r₀ */
  err = acgsolverhip_dcopy(n_owned, d_rhat, d_r, stream);
  if (err)
    return err;

  /* w₀ = A·r₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      stream, commstream, nnz_full, d_r, d_w, haloReady, haloRecv,
      d_one, d_zero, hipsparse, matA, matO, vecr, vecw, vecro, vecwo, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* t₀ = A·w₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
      d_one, d_zero, hipsparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* initial dot products: (r₀,r₀) and (r₀,w₀) */
  err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, &d_R2[0]);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipblasDdot(hipblas, n_owned, d_r, 1, d_w, 1, &d_R2[1]);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = bicgstab_allreduce(d_R2, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  double rr0, rw0;
  err = hipMemcpy(&rr0, &d_R2[0], sizeof(rr0), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpy(&rw0, &d_R2[1], sizeof(rw0), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  cg->r0nrm2 = cg->rnrm2 = sqrt(rr0);
  if (residualrtol > 0)
    residualrtol *= cg->r0nrm2;

  /*
   * Initial scalars: α₀ = (r,r)/(r,w), ρ₀ = (r,r), β = ω = 0, and p = s = z =
   * v = 0 (the β₋₁ = 0 recurrence then yields p₀=r, s₀=w, z₀=t).
   *
   * All of this must be enqueued on the compute stream. The solver's streams
   * are created with ‘hipStreamNonBlocking’, so they are not ordered against
   * the null stream that the synchronous ‘hipMemcpy()’ and ‘hipMemset()’ would
   * use. Those calls are also asynchronous with respect to the host for device
   * memory, and the four vector fills are large -- a few hundred megabytes on a
   * large system -- so they can still be in flight once the solver loop below
   * has started, and land on top of the p, s, z and v that the first iterations
   * have already updated.
   */
  double alpha0 = rr0 / rw0;
  err = hipMemcpyAsync(
      d_alpha, &alpha0, sizeof(alpha0), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_rho, &rr0, sizeof(rr0), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_beta, d_zero, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_omega, d_zero, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_p, 0, nnz_full * sizeof(*d_p), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_s, 0, nnz_full * sizeof(*d_s), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_z, 0, nnz_full * sizeof(*d_z), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_v, 0, nnz_full * sizeof(*d_v), stream);
  if (err)
    return ACG_ERR_HIP;
  /* ‘alpha0’ and ‘rr0’ are read by the asynchronous copies above, so they must
   * stay put until those have run */
  err = hipStreamSynchronize(stream);
  if (err)
    return ACG_ERR_HIP;

  /* iterative solver loop */
  bool breakdown = false;
  /* fixed-iteration benchmark mode: no stopping criteria are set, so run
   * exactly ‘maxits’ iterations and disable the breakdown early-out */
  bool fixed_iterations =
      (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0);
  /* the stream that makes a reduction readable, and how many scalars the host
   * needs each iteration: the residual norm always, ω and ρ only when the
   * breakdown test is active */
  hipStream_t readystream = pbicgstab_reduce_ready_stream(
      comm, commsize, nocomm_allreduce, stream, collective_stream);
  int nscalars = fixed_iterations ? 1 : 3;
  for (int k = 0; k < maxits; k++)
  {
    /* steps 1-3: p,s,z recurrences */
    err = acgsolverhip_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;

    /*
    begin halo exchange
    */
    if (commsize > 1 && !nocomm_p2p)
    {
      err = hipEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_HIP;
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
          ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
    }

    /* steps 4-5: q = r − α·s, y = w − α·z */
    err = acgsolverhip_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* reduction R1 = {(q,y),(y,y)}, overlapped with v = A·z */
    err = hipblasDdot(hipblas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;

    // spmv diag
#if defined(ACG_USE_HIPSPARSE)
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecz, d_zero,
        vecv, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
#else
    err = acgsolverhip_csrgemv_merge(
        (A->nprows - A->nghostrows), d_v, d_z, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows_v, d_startrows, stream);
    if (err)
      return err;
#endif

    // end halo exchange
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
            ACG_DOUBLE, comm, tag, errcode, 0, commstream);
        if (err)
          return err;
        err = hipEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_HIP;
        err = hipStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_HIP;
      }
      // spmv off-diagonal: always hipSPARSE, accumulate
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, veczo, d_one,
          vecvo, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
    }

    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11: ω, then x, r, w updates */
    err = acgsolverhip_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_qy, d_yy, d_alpha, d_p, d_q, d_y, d_t, d_v, d_x, d_r, d_w, stream);
    if (err)
      return err;

    // begin halo exchange
    if (commsize > 1 && !nocomm_p2p)
    {
      err = hipEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_HIP;
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
          ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
    }

    /* reduction R2 = {(r̂0,r),(r̂0,w),(r̂0,s),(r̂0,z),(r,r)}, overlapped with t = A·w */
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_HIPBLAS;
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

    // spmv diag
#if defined(ACG_USE_HIPSPARSE)
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecw, d_zero,
        vect, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
#else
    err = acgsolverhip_csrgemv_merge(
        (A->nprows - A->nghostrows), d_t, d_w, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows_v, d_startrows, stream);
    if (err)
      return err;
#endif

    // end halo exchange, spmv off-diagonal
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
            ACG_DOUBLE, comm, tag, errcode, 0, commstream);
        if (err)
          return err;
        err = hipEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_HIP;
        err = hipStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_HIP;
      }
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecwo, d_one,
          vecto, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
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
    err = hipStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_HIP;

    /* steps 14-15: β and α for the next iteration. Enqueued before the host
     * blocks on the residual norm below, so that the first dependency of the
     * next iteration is ready as soon as the trailing t = A·w retires. Its
     * result goes unused if one of the tests below leaves the loop. */
    err = acgsolverhip_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_d1, d_d2, d_d3, d_d4, stream);
    if (err)
      return err;

    /* residual norm for convergence. Only the few bytes staged on
     * ‘copystream’ are waited for, and they were gated on the reduction alone,
     * so the trailing sparse matrix-vector product, the scalar update and the
     * launches of the next iteration all overlap with this. Synchronising on
     * the compute stream instead would drain the whole queue every
     * iteration. */
    err = hipStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_HIP;
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
     * Breakdown / stagnation detection (see the non-pipelined solver above);
     * skipped in fixed-iteration benchmark mode.
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
  /* Drain the compute stream before stopping the clock, so that the reported
   * time covers all of the work and the solution copied out below is complete.
   * The solver's streams are non-blocking, so the copy on the null stream is
   * not ordered against them. */
  hipStreamSynchronize(stream);
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

  /* copy solution back to host */
  err = hipMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;

  hipsparseDestroyDnVec(vecx);
  hipsparseDestroyDnVec(vecr);
  hipsparseDestroyDnVec(vecw);
  hipsparseDestroyDnVec(vect);
  hipsparseDestroyDnVec(vecz);
  hipsparseDestroyDnVec(vecv);
  if (commsize > 1)
  {
    hipsparseDestroyDnVec(vecxo);
    hipsparseDestroyDnVec(vecro);
    hipsparseDestroyDnVec(vecwo);
    hipsparseDestroyDnVec(vecto);
    hipsparseDestroyDnVec(veczo);
    hipsparseDestroyDnVec(vecvo);
  }
  hipsparseDestroySpMat(matA);
  hipFree(d_buffer);
  if (commsize > 1)
  {
    hipsparseDestroySpMat(matO);
    hipFree(d_obuffer);
  }
#if !defined(ACG_USE_HIPSPARSE)
  hipFree(d_startrows);
#endif

  hipFree(d_x);
  hipFree(d_b);
  hipFree(d_alpha);
  hipFree(d_beta);
  hipFree(d_omega);
  hipFree(d_rho);
  hipFree(d_R1);
  hipFree(d_R2);
  hipEventDestroy(dotEvent);
  hipEventDestroy(redEvent);
  hipEventDestroy(haloReady);
  hipEventDestroy(haloRecv);
  hipStreamDestroy(stream);
  hipStreamDestroy(commstream);
  hipStreamDestroy(collective_stream);
  hipEventDestroy(scalarsReduced);
  hipEventDestroy(scalarsCopied);
  hipStreamDestroy(copystream);
  hipHostFree(h_scalars);

  /* reset hipsparse and hipblas pointer modes */
  err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipblasSetPointerMode(hipblas, hipblaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPBLAS;
  }

  if (hipGetLastError() != hipSuccess)
    return ACG_ERR_HIP;

  if (converged)
    return ACG_SUCCESS;
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
 * ‘acgsolverhip_solve_pipelined_bicgstab_rr()’ solves the given linear
 * system, Ax=b, using the communication-hiding pipelined BiCGStab method
 * (Cools & Vanroose, 2017), without a preconditioner, augmented with
 * PETSc-style periodic residual replacement: every ACG_BICGSTAB_RR_PERIOD
 * iterations the recurrence-propagated vectors r,w,t,s,z,v are recomputed
 * from the primary vectors x and p with explicit SpMVs, resetting the
 * accumulated rounding error (mirrors PETSc KSPPIPEBCGS). The linear system
 * may be distributed across multiple processes; the two global reductions
 * per iteration are overlapped with the two sparse matrix-vector products.
 */
int acgsolverhip_solve_pipelined_bicgstab_rr(
    struct acgsolverhip *cg,
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
    hipblasHandle_t hipblas,
    hipsparseHandle_t hipsparse)
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
      err = hipMalloc((void **)devptrs[i], (*vecptrs[i])->num_nonzeros * sizeof(double));
      if (err)
        return ACG_ERR_HIP;
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
  err = hipMalloc((void **)&d_alpha, sizeof(*d_alpha));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_beta, sizeof(*d_beta));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_omega, sizeof(*d_omega));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_rho, sizeof(*d_rho));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_R1, 2 * sizeof(*d_R1));
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_R2, 5 * sizeof(*d_R2));
  if (err)
    return ACG_ERR_HIP;
  double *d_qy = &d_R1[0]; /* (q,y) */
  double *d_yy = &d_R1[1]; /* (y,y) */
  double *d_d1 = &d_R2[0]; /* (r̂0,r) */
  double *d_d2 = &d_R2[1]; /* (r̂0,w) */
  double *d_d3 = &d_R2[2]; /* (r̂0,s) */
  double *d_d4 = &d_R2[3]; /* (r̂0,z) */
  double *d_rr = &d_R2[4]; /* (r,r) for the residual norm */

  int leastPriority, greatestPriority;
  err = hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t stream;
  err = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, leastPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t collective_stream;
  err = hipStreamCreateWithPriority(&collective_stream, hipStreamNonBlocking, greatestPriority);
  if (err)
    return ACG_ERR_HIP;
  hipStream_t commstream;
  err = hipStreamCreateWithPriority(&commstream, hipStreamNonBlocking, (leastPriority + greatestPriority) / 2);
  if (err)
    return ACG_ERR_HIP;

  /* configure hipblas and hipsparse to use device-side pointers */
  hipblasPointerMode_t hipblaspointermode;
  err = hipblasGetPointerMode(hipblas, &hipblaspointermode);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipblasSetPointerMode(hipblas, HIPBLAS_POINTER_MODE_DEVICE);
  if (err)
    return ACG_ERR_HIPBLAS;
  hipsparsePointerMode_t hipsparsepointermode;
  err = hipsparseGetPointerMode(hipsparse, &hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipsparseSetPointerMode(hipsparse, HIPSPARSE_POINTER_MODE_DEVICE);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }

  hipEvent_t dotEvent, redEvent, haloReady, haloRecv;
  hipEventCreateWithFlags(&dotEvent, hipEventDisableTiming);
  hipEventCreateWithFlags(&redEvent, hipEventDisableTiming);
  hipEventCreateWithFlags(&haloReady, hipEventDisableTiming);
  hipEventCreateWithFlags(&haloRecv, hipEventDisableTiming);
  MPI_Request request;

  /* a dedicated stream for fetching the handful of scalars that the host needs
   * for the convergence tests, together with pinned staging memory for ‖r‖₂²,
   * ω and ρ, so that those transfers neither block nor are blocked by the work
   * queued on the compute stream */
  hipStream_t copystream;
  err = hipStreamCreateWithFlags(&copystream, hipStreamNonBlocking);
  if (err)
    return ACG_ERR_HIP;
  hipEvent_t scalarsReduced, scalarsCopied;
  hipEventCreateWithFlags(&scalarsReduced, hipEventDisableTiming);
  hipEventCreateWithFlags(&scalarsCopied, hipEventDisableTiming);
  double *h_scalars;
  err = hipHostMalloc(
      (void **)&h_scalars, 3 * sizeof(*h_scalars), hipHostMallocNumaUser);
  if (err)
    return ACG_ERR_HIP;

  /* copy right-hand side and initial guess to device */
  double *d_b, *d_x;
  err = hipMalloc((void **)&d_b, b->num_nonzeros * sizeof(*d_b));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_b, b->x, b->num_nonzeros * sizeof(*d_b), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMalloc((void **)&d_x, x->num_nonzeros * sizeof(*d_x));
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_x, x->x, x->num_nonzeros * sizeof(*d_x), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;

  /* create hipsparse dense vectors and matrices (created unconditionally;
   * the off-diagonal border product always uses hipSPARSE) */
  hipsparseDnVecDescr_t vecx, vecr, vecw, vect, vecz, vecv;
  hipsparseCreateDnVec(&vecx, A->nownedrows, d_x, HIP_R_64F);
  hipsparseCreateDnVec(&vecr, A->nownedrows, d_r, HIP_R_64F);
  hipsparseCreateDnVec(&vecw, A->nownedrows, d_w, HIP_R_64F);
  hipsparseCreateDnVec(&vect, A->nownedrows, d_t, HIP_R_64F);
  hipsparseCreateDnVec(&vecz, A->nownedrows, d_z, HIP_R_64F);
  hipsparseCreateDnVec(&vecv, A->nownedrows, d_v, HIP_R_64F);
  /* extra descriptors for residual replacement (s = A·p, z = A·s) */
  hipsparseDnVecDescr_t vecp, vecs;
  hipsparseCreateDnVec(&vecp, A->nownedrows, d_p, HIP_R_64F);
  hipsparseCreateDnVec(&vecs, A->nownedrows, d_s, HIP_R_64F);
  hipsparseDnVecDescr_t vecxo = NULL, vecro = NULL, vecwo = NULL, vecto = NULL,
                        veczo = NULL, vecvo = NULL;
  hipsparseDnVecDescr_t vecpo = NULL, vecso = NULL;
  if (commsize > 1)
  {
    int no = A->nborderrows + A->nghostrows;
    hipsparseCreateDnVec(&vecxo, no, d_x + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecro, no, d_r + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecwo, no, d_w + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecto, no, d_t + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&veczo, no, d_z + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecvo, no, d_v + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecpo, no, d_p + A->borderrowoffset, HIP_R_64F);
    hipsparseCreateDnVec(&vecso, no, d_s + A->borderrowoffset, HIP_R_64F);
  }
  hipsparseSpMatDescr_t matA;
  err = hipsparseCreateCsr(
      &matA, A->nownedrows, A->nownedrows, A->fnpnzs, d_rowptr, d_colidx, d_a,
      HIPSPARSE_IDX_T, HIPSPARSE_IDX_T, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
  if (err)
    return ACG_ERR_HIPSPARSE;
  size_t buffersize;
  err = hipsparseSpMV_bufferSize(
      hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx,
      d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &buffersize);
  if (err)
    return ACG_ERR_HIPSPARSE;
  void *d_buffer;
  err = hipMalloc(&d_buffer, buffersize);
  if (err)
    return ACG_ERR_HIP;
  hipsparseSpMatDescr_t matO = NULL;
  void *d_obuffer = NULL;
  if (commsize > 1)
  {
    err = hipsparseCreateCsr(
        &matO, A->nborderrows + A->nghostrows, A->nborderrows + A->nghostrows,
        A->onpnzs, d_orowptr, d_ocolidx, d_oa, HIPSPARSE_IDX_T, HIPSPARSE_IDX_T,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F);
    if (err)
      return ACG_ERR_HIPSPARSE;
    size_t obuffersize;
    err = hipsparseSpMV_bufferSize(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo,
        d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, &obuffersize);
    if (err)
      return ACG_ERR_HIPSPARSE;
    err = hipMalloc(&d_obuffer, obuffersize);
    if (err)
      return ACG_ERR_HIP;
  }

#if !defined(ACG_USE_HIPSPARSE)
  /* setup for merge-based SpMV (diagonal block) */
  const int TASKS_PER_THREAD = 10;
  acgidx_t ntasks = (A->nprows - A->nghostrows) + A->fnpnzs;
  acgidx_t nstartrows_v = (ntasks + TASKS_PER_THREAD - 1) / TASKS_PER_THREAD;
  acgidx_t *d_startrows = NULL;
  err = hipMalloc((void **)&d_startrows, nstartrows_v * sizeof(*d_startrows));
  if (err)
    return ACG_ERR_HIP;
  err = acgsolverhip_csrgemv_merge_startrows(
      (A->nprows - A->nghostrows), d_rowptr, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
  err = hipStreamSynchronize(stream);
  if (err)
    return ACG_ERR_HIP;
#else
  acgidx_t nstartrows_v = 0;
  acgidx_t *d_startrows = NULL;
#endif

  err = hipblasSetStream(hipblas, stream);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipsparseSetStream(hipsparse, stream);
  if (err)
    return ACG_ERR_HIPSPARSE;

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
    err = hipMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_HIP;
  }
  double *d_scratchscalars[] = {d_alpha, d_beta, d_omega, d_rho};
  for (int i = 0; i < (int)(sizeof(d_scratchscalars) / sizeof(*d_scratchscalars)); i++)
  {
    err = hipMemcpyAsync(
        d_scratchscalars[i], d_one, sizeof(**d_scratchscalars),
        hipMemcpyDeviceToDevice, stream);
    if (err)
      return ACG_ERR_HIP;
  }


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
    err = acgsolverhip_dcopy(n_owned, d_r, d_b, stream);
    if (err)
      return err;
    err = hipblasDdot(hipblas, n_owned, d_b, 1, d_b, 1, d_bnrm2sqr);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
    if (err)
      return err;

    /* steps 1-5: the p,s,z and q,y recurrences */
    err = acgsolverhip_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;
    err = acgsolverhip_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* the first reduction, overlapped with v = A·z */
    err = hipblasDdot(hipblas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
        d_one, d_zero, hipsparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
    if (err)
      return err;
    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11, with ω taken from ‘d_one’ rather than from the reduction */
    err = acgsolverhip_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_one, d_one, d_alpha, d_p, d_q, d_y, d_t, d_v, d_rhat, d_r, d_w, stream);
    if (err)
      return err;

    /* the second reduction, overlapped with t = A·w */
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = pbicgstab_reduce_begin(d_R2, 5, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;
    err = bicgstab_spmv(
        A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
        stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
        d_one, d_zero, hipsparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
        d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
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
    err = hipStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_HIP;

    /* steps 14-15, again fed from ‘d_one’ to keep β and α at one */
    err = acgsolverhip_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_one, d_one, d_one, d_one, stream);
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
          stream, commstream, nnz_full, d_p, d_s, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecp, vecs, vecpo, vecso, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          stream, commstream, nnz_full, d_s, d_z, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecs, vecz, vecso, veczo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
    }

    err = hipStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_HIP;
  }

  /* discard what the warmup left behind; the scalars and the owned parts of
   * the vectors are assigned again by the initialisation below, and the ghost
   * parts by the halo exchange preceding each SpMV that reads them */
  for (int i = 0; i < nscratchvecs; i++)
  {
    err = hipMemsetAsync(
        d_scratchvecs[i], 0, nnz_full * sizeof(**d_scratchvecs), stream);
    if (err)
      return ACG_ERR_HIP;
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
  err = acgcomm_barrier_hip(stream, comm, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  hipStreamSynchronize(commstream);
  hipStreamSynchronize(collective_stream);
  hipStreamSynchronize(copystream);
  gettime(&t0);

  /* ‖b‖₂ */
  err = hipblasDdot(hipblas, b->num_nonzeros - b->num_ghost_nonzeros, d_b, 1, d_b, 1, d_bnrm2sqr);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = bicgstab_allreduce(d_bnrm2sqr, 1, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  double bnrm2sqr;
  err = hipMemcpy(&bnrm2sqr, d_bnrm2sqr, sizeof(bnrm2sqr), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  cg->bnrm2 = sqrt(bnrm2sqr);

  /* r₀ = b − A·x₀ */
  err = acgsolverhip_dcopy(b->num_nonzeros - b->num_ghost_nonzeros, d_r, d_b, stream);
  if (err)
    return err;
  if (commsize > 1 && !nocomm_p2p)
  {
    err = hipEventRecord(haloReady, stream);
    if (err)
      return ACG_ERR_HIP;
    err = hipStreamWaitEvent(commstream, haloReady, 0);
    if (err)
      return ACG_ERR_HIP;
    err = acghalo_exchange_hip_begin(
        cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x,
        ACG_DOUBLE, comm, tag, errcode, 0, commstream);
    if (err)
      return err;
  }
#if defined(ACG_USE_HIPSPARSE)
  err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
  if (err)
    return ACG_ERR_HIPSPARSE;
#else
  err = acgsolverhip_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0, nstartrows_v, d_startrows, stream);
  if (err)
    return err;
#endif
  if (commsize > 1)
  {
    if (!nocomm_p2p)
    {
      err = acghalo_exchange_hip_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
      err = hipEventRecord(haloRecv, commstream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(stream, haloRecv, 0);
      if (err)
        return ACG_ERR_HIP;
    }
    err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
    if (err)
      return ACG_ERR_HIPSPARSE;
  }
  cg->ngemv++;

  /* r̂0 = r₀ */
  err = acgsolverhip_dcopy(n_owned, d_rhat, d_r, stream);
  if (err)
    return err;

  /* w₀ = A·r₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      stream, commstream, nnz_full, d_r, d_w, haloReady, haloRecv,
      d_one, d_zero, hipsparse, matA, matO, vecr, vecw, vecro, vecwo, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* t₀ = A·w₀ */
  err = bicgstab_spmv(
      A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
      stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
      d_one, d_zero, hipsparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
      d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
  if (err)
    return err;
  cg->ngemv++;

  /* initial dot products: (r₀,r₀) and (r₀,w₀) */
  err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, &d_R2[0]);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = hipblasDdot(hipblas, n_owned, d_r, 1, d_w, 1, &d_R2[1]);
  if (err)
    return ACG_ERR_HIPBLAS;
  err = bicgstab_allreduce(d_R2, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, errcode);
  if (err)
    return err;
  hipStreamSynchronize(stream);
  double rr0, rw0;
  err = hipMemcpy(&rr0, &d_R2[0], sizeof(rr0), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpy(&rw0, &d_R2[1], sizeof(rw0), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;
  cg->r0nrm2 = cg->rnrm2 = sqrt(rr0);
  if (residualrtol > 0)
    residualrtol *= cg->r0nrm2;

  /*
   * Initial scalars: α₀ = (r,r)/(r,w), ρ₀ = (r,r), β = ω = 0, and p = s = z =
   * v = 0 (the β₋₁ = 0 recurrence then yields p₀=r, s₀=w, z₀=t).
   *
   * All of this must be enqueued on the compute stream. The solver's streams
   * are created with ‘hipStreamNonBlocking’, so they are not ordered against
   * the null stream that the synchronous ‘hipMemcpy()’ and ‘hipMemset()’ would
   * use. Those calls are also asynchronous with respect to the host for device
   * memory, and the four vector fills are large -- a few hundred megabytes on a
   * large system -- so they can still be in flight once the solver loop below
   * has started, and land on top of the p, s, z and v that the first iterations
   * have already updated.
   */
  double alpha0 = rr0 / rw0;
  err = hipMemcpyAsync(
      d_alpha, &alpha0, sizeof(alpha0), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_rho, &rr0, sizeof(rr0), hipMemcpyHostToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_beta, d_zero, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemcpyAsync(
      d_omega, d_zero, sizeof(double), hipMemcpyDeviceToDevice, stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_p, 0, nnz_full * sizeof(*d_p), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_s, 0, nnz_full * sizeof(*d_s), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_z, 0, nnz_full * sizeof(*d_z), stream);
  if (err)
    return ACG_ERR_HIP;
  err = hipMemsetAsync(d_v, 0, nnz_full * sizeof(*d_v), stream);
  if (err)
    return ACG_ERR_HIP;
  /* ‘alpha0’ and ‘rr0’ are read by the asynchronous copies above, so they must
   * stay put until those have run */
  err = hipStreamSynchronize(stream);
  if (err)
    return ACG_ERR_HIP;

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
  hipStream_t readystream = pbicgstab_reduce_ready_stream(
      comm, commsize, nocomm_allreduce, stream, collective_stream);
  int nscalars = fixed_iterations ? 1 : 3;
  for (int k = 0; k < maxits; k++)
  {
    /* steps 1-3: p,s,z recurrences */
    err = acgsolverhip_pipelined_bicgstab_psz_update(n_owned, d_beta, d_omega, d_r, d_w, d_t, d_p, d_s, d_z, d_v, stream);
    if (err)
      return err;

    /*
    begin halo exchange
    */
    if (commsize > 1 && !nocomm_p2p)
    {
      err = hipEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_HIP;
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
          ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
    }

    /* steps 4-5: q = r − α·s, y = w − α·z */
    err = acgsolverhip_pipelined_bicgstab_qy_update(n_owned, d_alpha, d_r, d_w, d_s, d_z, d_q, d_y, stream);
    if (err)
      return err;

    /* reduction R1 = {(q,y),(y,y)}, overlapped with v = A·z */
    err = hipblasDdot(hipblas, n_owned, d_q, 1, d_y, 1, d_qy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_y, 1, d_y, 1, d_yy);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = pbicgstab_reduce_begin(d_R1, 2, comm, commsize, nocomm_allreduce, stream, collective_stream, dotEvent, &request, errcode);
    if (err)
      return err;

    // spmv diag
#if defined(ACG_USE_HIPSPARSE)
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecz, d_zero,
        vecv, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
#else
    err = acgsolverhip_csrgemv_merge(
        (A->nprows - A->nghostrows), d_v, d_z, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows_v, d_startrows, stream);
    if (err)
      return err;
#endif

    // end halo exchange
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange, nnz_full, d_z, ACG_DOUBLE, nnz_full, d_z,
            ACG_DOUBLE, comm, tag, errcode, 0, commstream);
        if (err)
          return err;
        err = hipEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_HIP;
        err = hipStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_HIP;
      }
      // spmv off-diagonal: always hipSPARSE, accumulate
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, veczo, d_one,
          vecvo, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
    }

    err = pbicgstab_reduce_end(comm, commsize, nocomm_allreduce, stream, collective_stream, redEvent, &request);
    if (err)
      return err;

    /* steps 8-11: ω, then x, r, w updates */
    err = acgsolverhip_pipelined_bicgstab_xrw_update(n_owned, d_omega, d_qy, d_yy, d_alpha, d_p, d_q, d_y, d_t, d_v, d_x, d_r, d_w, stream);
    if (err)
      return err;

    // begin halo exchange
    if (commsize > 1 && !nocomm_p2p)
    {
      err = hipEventRecord(haloReady, stream);
      if (err)
        return ACG_ERR_HIP;
      err = hipStreamWaitEvent(commstream, haloReady, 0);
      if (err)
        return ACG_ERR_HIP;
      err = acghalo_exchange_hip_begin(
          cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
          ACG_DOUBLE, comm, tag, errcode, 0, commstream);
      if (err)
        return err;
    }

    /* reduction R2 = {(r̂0,r),(r̂0,w),(r̂0,s),(r̂0,z),(r,r)}, overlapped with t = A·w */
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_r, 1, d_d1);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_w, 1, d_d2);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_s, 1, d_d3);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_rhat, 1, d_z, 1, d_d4);
    if (err)
      return ACG_ERR_HIPBLAS;
    err = hipblasDdot(hipblas, n_owned, d_r, 1, d_r, 1, d_rr);
    if (err)
      return ACG_ERR_HIPBLAS;
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

    // spmv diag
#if defined(ACG_USE_HIPSPARSE)
    err = hipsparseSpMV(
        hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matA, vecw, d_zero,
        vect, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
    if (err)
    {
      if (errcode)
        *errcode = err;
      return ACG_ERR_HIPSPARSE;
    }
#else
    err = acgsolverhip_csrgemv_merge(
        (A->nprows - A->nghostrows), d_t, d_w, d_rowptr, d_colidx, d_a, 1.0,
        0.0, nstartrows_v, d_startrows, stream);
    if (err)
      return err;
#endif

    // end halo exchange, spmv off-diagonal
    if (commsize > 1)
    {
      if (!nocomm_p2p)
      {
        err = acghalo_exchange_hip_end(
            cg->halo, cg->haloexchange, nnz_full, d_w, ACG_DOUBLE, nnz_full, d_w,
            ACG_DOUBLE, comm, tag, errcode, 0, commstream);
        if (err)
          return err;
        err = hipEventRecord(haloRecv, commstream);
        if (err)
          return ACG_ERR_HIP;
        err = hipStreamWaitEvent(stream, haloRecv, 0);
        if (err)
          return ACG_ERR_HIP;
      }
      err = hipsparseSpMV(
          hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_one, matO, vecwo, d_one,
          vecto, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
      if (err)
      {
        if (errcode)
          *errcode = err;
        return ACG_ERR_HIPSPARSE;
      }
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
    err = hipStreamWaitEvent(stream, scalarsCopied, 0);
    if (err)
      return ACG_ERR_HIP;

    /* steps 14-15: β and α for the next iteration. Enqueued before the host
     * blocks on the residual norm below, so that the first dependency of the
     * next iteration is ready as soon as the trailing t = A·w retires. Its
     * result goes unused if one of the tests below leaves the loop. */
    err = acgsolverhip_pipelined_bicgstab_scalars(d_beta, d_alpha, d_rho, d_omega, d_d1, d_d2, d_d3, d_d4, stream);
    if (err)
      return err;

    /* residual norm for convergence. Only the few bytes staged on
     * ‘copystream’ are waited for, and they were gated on the reduction alone,
     * so the trailing sparse matrix-vector product, the scalar update and the
     * launches of the next iteration all overlap with this. Synchronising on
     * the compute stream instead would drain the whole queue every
     * iteration. */
    err = hipStreamSynchronize(copystream);
    if (err)
      return ACG_ERR_HIP;
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
     * Breakdown / stagnation detection (see the non-pipelined solver above);
     * skipped in fixed-iteration benchmark mode.
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
      err = acgsolverhip_dcopy(n_owned, d_r, d_b, stream);
      if (err)
        return err;
      if (commsize > 1 && !nocomm_p2p)
      {
        err = hipEventRecord(haloReady, stream);
        if (err)
          return ACG_ERR_HIP;
        err = hipStreamWaitEvent(commstream, haloReady, 0);
        if (err)
          return ACG_ERR_HIP;
        err = acghalo_exchange_hip_begin(
            cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full,
            d_x, ACG_DOUBLE, comm, tag, errcode, 0, commstream);
        if (err)
          return err;
      }
#if defined(ACG_USE_HIPSPARSE)
      err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matA, vecx, d_one, vecr, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer);
      if (err)
        return ACG_ERR_HIPSPARSE;
#else
      err = acgsolverhip_csrgemv_merge((A->nprows - A->nghostrows), d_r, d_x, d_rowptr, d_colidx, d_a, -1.0, 1.0, nstartrows_v, d_startrows, stream);
      if (err)
        return err;
#endif
      if (commsize > 1)
      {
        if (!nocomm_p2p)
        {
          err = acghalo_exchange_hip_end(cg->halo, cg->haloexchange, nnz_full, d_x, ACG_DOUBLE, nnz_full, d_x, ACG_DOUBLE, comm, tag, errcode, 0, commstream);
          if (err)
            return err;
          err = hipEventRecord(haloRecv, commstream);
          if (err)
            return ACG_ERR_HIP;
          err = hipStreamWaitEvent(stream, haloRecv, 0);
          if (err)
            return ACG_ERR_HIP;
        }
        err = hipsparseSpMV(hipsparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, d_minus_one, matO, vecxo, d_one, vecro, HIP_R_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_obuffer);
        if (err)
          return ACG_ERR_HIPSPARSE;
      }
      cg->ngemv++;

      /* w = A·r */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          stream, commstream, nnz_full, d_r, d_w, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecr, vecw, vecro, vecwo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* t = A·w */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          stream, commstream, nnz_full, d_w, d_t, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecw, vect, vecwo, vecto, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* s = A·p */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          stream, commstream, nnz_full, d_p, d_s, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecp, vecs, vecpo, vecso, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* z = A·s */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          stream, commstream, nnz_full, d_s, d_z, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecs, vecz, vecso, veczo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;

      /* v = A·z */
      err = bicgstab_spmv(
          A, comm, commsize, nocomm_p2p, cg->halo, cg->haloexchange, tag, errcode,
          stream, commstream, nnz_full, d_z, d_v, haloReady, haloRecv,
          d_one, d_zero, hipsparse, matA, matO, vecz, vecv, veczo, vecvo, d_buffer,
          d_obuffer, d_rowptr, d_colidx, d_a, nstartrows_v, d_startrows);
      if (err)
        return err;
      cg->ngemv++;
    }
  }
  /* Drain the compute stream before stopping the clock, so that the reported
   * time covers all of the work and the solution copied out below is complete.
   * The solver's streams are non-blocking, so the copy on the null stream is
   * not ordered against them. */
  hipStreamSynchronize(stream);
  gettime(&t1);
  cg->tsolve += elapsed(t0, t1);

  /* copy solution back to host */
  err = hipMemcpy(x->x, d_x, x->num_nonzeros * sizeof(*d_x), hipMemcpyDeviceToHost);
  if (err)
    return ACG_ERR_HIP;

  hipsparseDestroyDnVec(vecx);
  hipsparseDestroyDnVec(vecr);
  hipsparseDestroyDnVec(vecw);
  hipsparseDestroyDnVec(vect);
  hipsparseDestroyDnVec(vecz);
  hipsparseDestroyDnVec(vecv);
  hipsparseDestroyDnVec(vecp);
  hipsparseDestroyDnVec(vecs);
  if (commsize > 1)
  {
    hipsparseDestroyDnVec(vecxo);
    hipsparseDestroyDnVec(vecro);
    hipsparseDestroyDnVec(vecwo);
    hipsparseDestroyDnVec(vecto);
    hipsparseDestroyDnVec(veczo);
    hipsparseDestroyDnVec(vecvo);
    hipsparseDestroyDnVec(vecpo);
    hipsparseDestroyDnVec(vecso);
  }
  hipsparseDestroySpMat(matA);
  hipFree(d_buffer);
  if (commsize > 1)
  {
    hipsparseDestroySpMat(matO);
    hipFree(d_obuffer);
  }
#if !defined(ACG_USE_HIPSPARSE)
  hipFree(d_startrows);
#endif

  hipFree(d_x);
  hipFree(d_b);
  hipFree(d_alpha);
  hipFree(d_beta);
  hipFree(d_omega);
  hipFree(d_rho);
  hipFree(d_R1);
  hipFree(d_R2);
  hipEventDestroy(dotEvent);
  hipEventDestroy(redEvent);
  hipEventDestroy(haloReady);
  hipEventDestroy(haloRecv);
  hipStreamDestroy(stream);
  hipStreamDestroy(commstream);
  hipStreamDestroy(collective_stream);
  hipEventDestroy(scalarsReduced);
  hipEventDestroy(scalarsCopied);
  hipStreamDestroy(copystream);
  hipHostFree(h_scalars);

  /* reset hipsparse and hipblas pointer modes */
  err = hipsparseSetPointerMode(hipsparse, hipsparsepointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPSPARSE;
  }
  err = hipblasSetPointerMode(hipblas, hipblaspointermode);
  if (err)
  {
    if (errcode)
      *errcode = err;
    return ACG_ERR_HIPBLAS;
  }

  if (hipGetLastError() != hipSuccess)
    return ACG_ERR_HIP;

  if (converged)
    return ACG_SUCCESS;
  if (breakdown)
    return ACG_ERR_NOT_CONVERGED;
  if (diffatol == 0 && diffrtol == 0 && residualatol == 0 && residualrtol == 0)
    return ACG_SUCCESS;
  return ACG_ERR_NOT_CONVERGED;
}

/*
 * output solver info
 */

static void findent(FILE *f, int indent) { fprintf(f, "%*c", indent, ' '); }

/**
 * ‘acgsolverhip_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolverhip_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolverhip_fwrite(
    FILE *f,
    const struct acgsolverhip *cg,
    int indent)
{
    double tother = cg->tsolve -
                    (cg->tgemv + cg->tdot + cg->tnrm2 + cg->taxpy + cg->tcopy + cg->tallreduce + cg->thalo);
    findent(f, indent);
    fprintf(f, "unknowns: %'" PRIdx "\n", cg->p.size);
    findent(f, indent);
    fprintf(f, "solves: %'d\n", cg->nsolves);
    findent(f, indent);
    fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
    findent(f, indent);
    fprintf(f, "total flops: %'.3f Gflop\n", 1.0e-9 * cg->nflops);
    findent(f, indent);
    fprintf(f, "total flop rate: %'.3f Gflop/s\n", cg->tsolve > 0 ? 1.0e-9 * cg->nflops / cg->tsolve : 0);
    findent(f, indent);
    fprintf(f, "total solver time: %'.6f seconds\n", cg->tsolve);
    findent(f, indent);
    fprintf(f, "performance breakdown:\n");
    findent(f, indent);
    fprintf(f, "  gemv: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->tgemv, cg->ngemv, cg->Bgemv, cg->tgemv > 0 ? 1.0e-9 * cg->Bgemv / cg->tgemv : 0.0);
    findent(f, indent);
    fprintf(f, "  dot: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->tdot, cg->ndot, cg->Bdot, cg->tdot > 0 ? 1.0e-9 * cg->Bdot / cg->tdot : 0.0);
    findent(f, indent);
    fprintf(f, "  nrm2: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->tnrm2, cg->nnrm2, cg->Bnrm2, cg->tnrm2 > 0 ? 1.0e-9 * cg->Bnrm2 / cg->tnrm2 : 0.0);
    findent(f, indent);
    fprintf(f, "  axpy: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->taxpy, cg->naxpy, cg->Baxpy, cg->taxpy > 0 ? 1.0e-9 * cg->Baxpy / cg->taxpy : 0.0);
    findent(f, indent);
    fprintf(f, "  copy: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->tcopy, cg->ncopy, cg->Bcopy, cg->tcopy > 0 ? 1.0e-9 * cg->Bcopy / cg->tcopy : 0.0);
    findent(f, indent);
    fprintf(f, "  MPI_Allreduce: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->tallreduce, cg->nallreduce, cg->Ballreduce, cg->tallreduce > 0 ? 1.0e-9 * cg->Ballreduce / cg->tallreduce : 0.0);
    findent(f, indent);
    fprintf(f, "  MPI_HaloExchange: %'.6f seconds %'" PRId64 " times %'" PRId64 " B %'.3f GB/s\n",
            cg->thalo, cg->nhalo, cg->Bhalo, cg->thalo > 0 ? 1.0e-9 * cg->Bhalo / cg->thalo : 0.0);
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
    fprintf(f, "    tolerance for relative residual: %.*g\n", DBL_DIG, cg->residualrtol);
    findent(f, indent);
    fprintf(f, "    tolerance for difference in solution iterates: %.*g\n", DBL_DIG, cg->diffatol);
    findent(f, indent);
    fprintf(f, "    tolerance for relative difference in solution iterates: %.*g\n", DBL_DIG, cg->diffrtol);
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
    fprintf(f, "  difference in solution iterates 2-norm: %.*g\n", DBL_DIG, cg->dxnrm2);
    findent(f, indent);
    fprintf(f, "  floating-point exceptions: %s\n", acgerrcodestr(ACG_ERR_FEXCEPT, 0));
    return ACG_SUCCESS;
}

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolverhip_fwritempi()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolverhip_solvempi()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolverhip_fwritempi(
    FILE *f,
    const struct acgsolverhip *cg,
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
    double tprecond = cg->tprecond;
    double tother = tsolve - (tgemv + tdot + tnrm2 + taxpy + tcopy + tallreduce + thalo + tprecond);
    int64_t ngemv = cg->ngemv, Bgemv = cg->Bgemv;
    int64_t ndot = cg->ndot, Bdot = cg->Bdot;
    int64_t nnrm2 = cg->nnrm2, Bnrm2 = cg->Bnrm2;
    int64_t naxpy = cg->naxpy, Baxpy = cg->Baxpy;
    int64_t ncopy = cg->ncopy, Bcopy = cg->Bcopy;
    int64_t nallreduce = cg->nallreduce, Ballreduce = cg->Ballreduce;
    int64_t nprecond = cg->nprecond;
    int64_t nhalo = cg->nhalo, Bhalo = cg->Bhalo;
    int64_t nhalopack = cg->halo->npack, Bhalopack = cg->halo->Bpack;
    int64_t nhalounpack = cg->halo->nunpack, Bhalounpack = cg->halo->Bunpack;
    int64_t nhalompiirecv = cg->halo->nmpiirecv, Bhalompiirecv = cg->halo->Bmpiirecv;
    int64_t nhalompisend = cg->halo->nmpisend, Bhalompisend = cg->halo->Bmpisend;
    int64_t nhalomsgs = cg->nhalomsgs;
    MPI_Reduce(&cg->nflops, &nflops, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->tsolve, &tsolve, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(&cg->tgemv, &tgemv, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tgemv /= commsize;
    MPI_Reduce(&cg->tdot, &tdot, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tdot /= commsize;
    MPI_Reduce(&cg->tnrm2, &tnrm2, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tnrm2 /= commsize;
    MPI_Reduce(&cg->taxpy, &taxpy, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    taxpy /= commsize;
    MPI_Reduce(&cg->tcopy, &tcopy, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tcopy /= commsize;
    MPI_Reduce(&cg->tallreduce, &tallreduce, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tallreduce /= commsize;
    MPI_Reduce(&cg->tprecond, &tprecond, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tprecond /= commsize;
    MPI_Reduce(&cg->thalo, &thalo, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    thalo /= commsize;
    MPI_Reduce(rank == root ? MPI_IN_PLACE : &tother, &tother, 1, MPI_DOUBLE, MPI_SUM, root, comm);
    tother /= commsize;
    MPI_Reduce(&cg->ngemv, &ngemv, 1, MPI_INT64_T, MPI_SUM, root, comm);
    ngemv /= commsize;
    MPI_Reduce(&cg->ndot, &ndot, 1, MPI_INT64_T, MPI_SUM, root, comm);
    ndot /= commsize;
    MPI_Reduce(&cg->nnrm2, &nnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm);
    nnrm2 /= commsize;
    MPI_Reduce(&cg->naxpy, &naxpy, 1, MPI_INT64_T, MPI_SUM, root, comm);
    naxpy /= commsize;
    MPI_Reduce(&cg->ncopy, &ncopy, 1, MPI_INT64_T, MPI_SUM, root, comm);
    ncopy /= commsize;
    MPI_Reduce(&cg->nallreduce, &nallreduce, 1, MPI_INT64_T, MPI_SUM, root, comm);
    nallreduce /= commsize;
    MPI_Reduce(&cg->nprecond, &nprecond, 1, MPI_INT64_T, MPI_SUM, root, comm);
    nprecond /= commsize;
    MPI_Reduce(&cg->halo->npack, &nhalopack, 1, MPI_INT64_T, MPI_SUM, root, comm);
    nhalopack /= commsize;
    MPI_Reduce(&cg->halo->nunpack, &nhalounpack, 1, MPI_INT64_T, MPI_SUM, root, comm);
    nhalounpack /= commsize;
    MPI_Reduce(&cg->halo->nmpiirecv, &nhalompiirecv, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->halo->nmpisend, &nhalompisend, 1, MPI_INT64_T, MPI_SUM, root, comm);
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
    MPI_Reduce(&cg->Ballreduce, &Ballreduce, 1, MPI_INT64_T, MPI_SUM, root, comm);
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
        fprintf(f, "total flop rate: %'.3f Gflop/s\n", tsolve > 0 ? 1.0e-9 * nflops / tsolve : 0);
        findent(f, indent);
        fprintf(f, "total solver time: %'.6f seconds\n", tsolve);
        findent(f, indent);
        fprintf(f, "performance breakdown:\n");
        findent(f, indent);
        fprintf(f, "  gemv: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc\n",
                tgemv, ngemv, Bgemv, tgemv > 0 ? 1.0e-9 * Bgemv / tgemv : 0.0);
        findent(f, indent);
        fprintf(f, "  dot: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc\n",
                tdot, ndot, Bdot, tdot > 0 ? 1.0e-9 * Bdot / tdot : 0.0);
        findent(f, indent);
        fprintf(f, "  nrm2: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc\n",
                tnrm2, nnrm2, Bnrm2, tnrm2 > 0 ? 1.0e-9 * Bnrm2 / tnrm2 : 0.0);
        findent(f, indent);
        fprintf(f, "  axpy: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc\n",
                taxpy, naxpy, Baxpy, taxpy > 0 ? 1.0e-9 * Baxpy / taxpy : 0.0);
        fprintf(f, "  preconditioner: %'.6f seconds/proc %'" PRId64 " times/proc\n",
                tprecond, nprecond);
        findent(f, indent);
        fprintf(f, "  copy: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc\n",
                tcopy, ncopy, Bcopy, tcopy > 0 ? 1.0e-9 * Bcopy / tcopy : 0.0);
        findent(f, indent);
        fprintf(f, "  allreduce: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc %'.3f µs/op/proc\n",
                tallreduce, nallreduce, Ballreduce, tallreduce > 0 ? 1.0e-9 * Ballreduce / tallreduce : 0.0,
                nallreduce > 0 ? 1.0e6 * tallreduce / nallreduce : 0.0);
        findent(f, indent);
        fprintf(f, "  haloexchange: %'.6f seconds/proc %'" PRId64 " times/proc %'" PRId64 " B/proc %'.3f GB/s/proc %'.1f msg/proc %'.3f µs/msg/proc\n",
                thalo, nhalo, Bhalo, thalo > 0 ? 1.0e-9 * Bhalo / thalo : 0.0,
                ((double)nhalomsgs) / commsize, nhalomsgs > 0 ? 1.0e6 * thalo / nhalomsgs / commsize : 0.0);
    }

    int *pnrecipients = rank == root ? malloc(commsize * sizeof(*pnrecipients)) : NULL;
    MPI_Gather(&cg->halo->nrecipients, 1, MPI_INT, pnrecipients, 1, MPI_INT, root, comm);
    int *pnsenders = rank == root ? malloc(commsize * sizeof(*pnsenders)) : NULL;
    MPI_Gather(&cg->halo->nsenders, 1, MPI_INT, pnsenders, 1, MPI_INT, root, comm);
    int *psendsize = rank == root ? malloc(commsize * sizeof(*psendsize)) : NULL;
    MPI_Gather(&cg->halo->sendsize, 1, MPI_INT, psendsize, 1, MPI_INT, root, comm);
    int *precvsize = rank == root ? malloc(commsize * sizeof(*precvsize)) : NULL;
    MPI_Gather(&cg->halo->recvsize, 1, MPI_INT, precvsize, 1, MPI_INT, root, comm);
    int *pmaxsendcount = rank == root ? malloc(commsize * sizeof(*pmaxsendcount)) : NULL;
    int maxsendcount = 0;
    for (int q = 0; q < cg->halo->nrecipients; q++)
        maxsendcount = maxsendcount > cg->halo->sendcounts[q] ? maxsendcount : cg->halo->sendcounts[q];
    MPI_Gather(&maxsendcount, 1, MPI_INT, pmaxsendcount, 1, MPI_INT, root, comm);
    int *pmaxrecvcount = rank == root ? malloc(commsize * sizeof(*pmaxrecvcount)) : NULL;
    int maxrecvcount = 0;
    for (int q = 0; q < cg->halo->nsenders; q++)
        maxrecvcount = maxrecvcount > cg->halo->recvcounts[q] ? maxrecvcount : cg->halo->recvcounts[q];
    MPI_Gather(&maxrecvcount, 1, MPI_INT, pmaxrecvcount, 1, MPI_INT, root, comm);
    if (rank == root)
    {
        for (int p = 0; p < commsize; p++)
        {
            findent(f, indent);
            fprintf(f, "    rank %'2d sends %'" PRId64 " B %'lu B/it in %'lu msg %'d msg/it max %'lu B/msg\n",
                    p, (int64_t)nhalo * psendsize[p] * sizeof(double),
                    psendsize[p] * sizeof(double),
                    nhalo * pnrecipients[p], pnrecipients[p],
                    pmaxsendcount[p] * sizeof(double));
            findent(f, indent);
            fprintf(f, "    rank %'2d receives %'" PRId64 " B %'lu B/it in %'lu msg %'d msg/it max %'lu B/msg\n",
                    p, (int64_t)nhalo * precvsize[p] * sizeof(double),
                    precvsize[p] * sizeof(double),
                    nhalo * pnrecipients[p], pnrecipients[p],
                    pmaxrecvcount[p] * sizeof(double));
        }
    }

    const struct acghaloexchange *haloexchange = cg->haloexchange;
    int maxevents = haloexchange->maxevents, nevents = 0;
    double *texchange = malloc(maxevents * sizeof(*texchange));
    double *tpack = malloc(maxevents * sizeof(*tpack));
    double *tsendrecv = malloc(maxevents * sizeof(*tsendrecv));
    double *tunpack = malloc(maxevents * sizeof(*tunpack));
    int err = acghaloexchange_profile(
        haloexchange, maxevents, &nevents,
        texchange, tpack, tsendrecv, tunpack);
    if (err)
        return err;
    double *ptexchange = rank == root ? malloc(commsize * sizeof(*ptexchange)) : NULL;
    double *ptpack = rank == root ? malloc(commsize * sizeof(*ptpack)) : NULL;
    double *ptsendrecv = rank == root ? malloc(commsize * sizeof(*ptsendrecv)) : NULL;
    double *ptunpack = rank == root ? malloc(commsize * sizeof(*ptunpack)) : NULL;

    /* sum and mean over all iterations per rank */
    double texchangesum = 0.0, tpacksum = 0.0, tsendrecvsum = 0.0, tunpacksum = 0.0;
    for (int i = 0; i < nevents; i++)
    {
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
    if (rank == root)
    {
        for (int p = 0; p < commsize; p++)
        {
            sendsizeavg += psendsize[p];
            recvsizeavg += precvsize[p];
        }
        sendsizeavg /= commsize;
        recvsizeavg /= commsize;
        double texchangeavg = 0.0, tpackavg = 0.0, tsendrecvavg = 0.0, tunpackavg = 0.0;
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
        fprintf(f, "    summary of %'d most recent iterations per rank:\n", nevents);
        if (nevents > 0)
        {
            fprintf(f, "      mean of %'d ranks:"
                       " %'.6f s %'.6f s/it total"
                       " %'.6f s %'.6f s/it %'5.2f GB/s send %'5.2f GB/s recv"
                       " %'.6f s %'.6f s/it %'5.2f GB/s pack"
                       " %'.6f s %'.6f s/it %'5.2f GB/s unpack\n",
                    commsize, texchangeavg, texchangeavg / (double)nevents,
                    tsendrecvavg, tsendrecvavg / (double)nevents,
                    nevents * sendsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
                    nevents * recvsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
                    tpackavg, tpackavg / (double)nevents, nevents * sendsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tpackavg,
                    tunpackavg, tunpackavg / (double)nevents, nevents * recvsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tunpackavg);
            for (int p = 0; p < commsize; p++)
            {
                findent(f, indent);
                fprintf(f, "      rank %'2d:"
                           " %'.6f s %'.6f s/it total"
                           " %'.6f s %'.6f s/it %'5.2f GB/s send %'5.2f GB/s recv"
                           " %'.6f s %'.6f s/it %'5.2f GB/s pack"
                           " %'.6f s %'.6f s/it %'5.2f GB/s unpack\n",
                        p, ptexchange[p], ptexchange[p] / (double)nevents,
                        ptsendrecv[p], ptsendrecv[p] / (double)nevents,
                        nevents * psendsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
                        nevents * precvsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
                        ptpack[p], ptpack[p] / (double)nevents, nevents * psendsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptpack[p],
                        ptunpack[p], ptunpack[p] / (double)nevents, nevents * precvsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptunpack[p]);
            }
        }
    }

    if (verbose > 0)
    {
        double critpath = 0.0;
        for (int i = 0; i < nevents; i++)
        {
            MPI_Gather(&texchange[i], 1, MPI_DOUBLE, ptexchange, 1, MPI_DOUBLE, root, comm);
            MPI_Gather(&tpack[i], 1, MPI_DOUBLE, ptpack, 1, MPI_DOUBLE, root, comm);
            MPI_Gather(&tsendrecv[i], 1, MPI_DOUBLE, ptsendrecv, 1, MPI_DOUBLE, root, comm);
            MPI_Gather(&tunpack[i], 1, MPI_DOUBLE, ptunpack, 1, MPI_DOUBLE, root, comm);
            if (rank == root)
            {
                double texchangeavg = 0.0, tpackavg = 0.0, tsendrecvavg = 0.0, tunpackavg = 0.0;
                double texchangemax = 0.0;
                for (int p = 0; p < commsize; p++)
                {
                    texchangeavg += ptexchange[p];
                    tpackavg += ptpack[p];
                    tsendrecvavg += ptsendrecv[p];
                    tunpackavg += ptunpack[p];
                    texchangemax = texchangemax > ptexchange[p] ? texchangemax : ptexchange[p];
                }
                texchangeavg /= (double)commsize;
                tpackavg /= (double)commsize;
                tsendrecvavg /= (double)commsize;
                tunpackavg /= (double)commsize;
                critpath += texchangemax;

                findent(f, indent);
                fprintf(f, "    iteration %'4d:\n", cg->halo->nexchanges - i - 1);
                findent(f, indent);
                fprintf(
                    f, "      mean of %'2d ranks: %'.6f s total %'.6f s %'5.2f GB/s send %'5.2f GB/s recv %'.6f s %'5.2f GB/s pack %'.6f s %'5.2f GB/s unpack\n",
                    commsize, texchangeavg, tsendrecvavg,
                    sendsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
                    recvsizeavg * sizeof(double) * 1.0e-9 / tsendrecvavg,
                    tpackavg, sendsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tpackavg,
                    tunpackavg, recvsizeavg * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / tunpackavg);
                for (int p = 0; p < commsize; p++)
                {
                    findent(f, indent);
                    fprintf(f, "      rank %'2d: %'.6f s total %'.6f s %'5.2f GB/s send %'5.2f GB/s recv %'.6f s %'5.2f GB/s pack %'.6f s %'5.2f GB/s unpack\n",
                            p, ptexchange[p], ptsendrecv[p],
                            psendsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
                            precvsize[p] * sizeof(double) * 1.0e-9 / ptsendrecv[p],
                            ptpack[p], psendsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptpack[p],
                            ptunpack[p], precvsize[p] * (2 * sizeof(double) + sizeof(int)) * 1.0e-9 / ptunpack[p]);
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
        fprintf(f, "    tolerance for residual: %.*g\n", DBL_DIG, cg->residualatol);
        findent(f, indent);
        fprintf(f, "    tolerance for relative residual: %.*g\n", DBL_DIG, cg->residualrtol);
        findent(f, indent);
        fprintf(f, "    tolerance for difference in solution iterates: %.*g\n", DBL_DIG, cg->diffatol);
        findent(f, indent);
        fprintf(f, "    tolerance for relative difference in solution iterates: %.*g\n", DBL_DIG, cg->diffrtol);
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
        fprintf(f, "  difference in solution iterates 2-norm: %.*g\n", DBL_DIG, cg->dxnrm2);
        findent(f, indent);
        fprintf(f, "  floating-point exceptions: %s\n", acgerrcodestr(ACG_ERR_FEXCEPT, 0));
    }
    return ACG_SUCCESS;
}
#endif
