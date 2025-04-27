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

#ifndef ACG_CGCUDA_H
#define ACG_CGCUDA_H

#include "acg/config.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef ACG_HAVE_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef ACG_HAVE_CUSPARSE
#include <cusparse.h>
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct acgcomm;
struct acghalo;
struct acgsymcsrmatrix;

/**
 * ‘acgsolvercuda’ is a data structure for use with a conjugate
 * gradient-based iterative solver.
 *
 * It consists of three temporary vectors, r, p and t, which are used
 * throughout the iterative solution procedure.
 */
struct acgsolvercuda
{
    /* vectors */
    struct acgvector r;
    struct acgvector p;
    struct acgvector t;
    struct acgvector * w, * q, * z;
    struct acgvector * dx;

    /* “halo exchange” communication */
    struct acghalo * halo;
    struct acghaloexchange * haloexchange;

    /* stopping criterion */
    int maxits;
    double diffatol;
    double diffrtol;
    double residualatol;
    double residualrtol;

    /* norms of right-hand side, initial guess, residual, and so on,
     * needed for convergence tests and to provide diagnostics */
    double bnrm2;
    double r0nrm2, rnrm2;
    double x0nrm2, dxnrm2;

    /* device-side constants */
    double * d_minus_one, * d_one, * d_zero, * d_inf;

    /* device-side data */
    double * d_bnrm2sqr, * d_r0nrm2sqr, * d_rnrm2sqr, * d_rnrm2sqr_prev;
    double * d_pdott, * d_alpha, * d_minus_alpha, * d_beta;
    int * d_niterations, * d_converged;
    double * d_r, * d_p, * d_t, * d_w, * d_q, * d_z;
    acgidx_t * d_rowptr, * d_orowptr;
    acgidx_t * d_colidx, * d_ocolidx;
    double * d_a, * d_oa;
    int use_nvshmem;

    /* solver statistics, including the number of solves, iterations,
     * floating-point operations, and a performance breakdown of time
     * spent in different parts. */
    int nsolves, ntotaliterations, niterations;
    int64_t nflops;
    double tsolve;
    double tgemv, tdot, tnrm2, taxpy, tcopy, tallreduce, thalo;
    int64_t ngemv, ndot, nnrm2, naxpy, ncopy, nallreduce, nhalo;
    int64_t Bgemv, Bdot, Bnrm2, Baxpy, Bcopy, Ballreduce, Bhalo;
    int64_t nhalomsgs;
};

/*
 * memory management
 */

/**
 * ‘acgsolvercuda_free()’ frees storage allocated for a solver.
 */
ACG_API void acgsolvercuda_free(
    struct acgsolvercuda * cg);

/*
 * initialise a solver
 */

#if defined(ACG_HAVE_CUBLAS) && defined(ACG_HAVE_CUSPARSE)
/**
 * ‘acgsolvercuda_init()’ sets up a conjugate gradient solver for a given
 * symmetric sparse matrix in CSR format.
 *
 * The matrix may be partitioned and distributed.
 */
ACG_API int acgsolvercuda_init(
    struct acgsolvercuda * cg,
    const struct acgsymcsrmatrix * A,
    cublasHandle_t cublas,
    cusparseHandle_t cusparse,
    const struct acgcomm * comm);
#endif

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
ACG_API int acgsolvercuda_solve(
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
ACG_API int acgsolvercuda_solvempi(
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
    cusparseSpMVAlg_t cusparse_spmv_alg);

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
ACG_API int acgsolvercuda_solve_pipelined(
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
    cusparseHandle_t cusparse);
#endif

/*
 * output solver info
 */

/**
 * ‘acgsolvercuda_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolvercuda_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
ACG_API int acgsolvercuda_fwrite(
    FILE * f,
    const struct acgsolvercuda * cg,
    int indent);

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolvercuda_fwritempi()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolvercuda_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
ACG_API int acgsolvercuda_fwritempi(
    FILE * f,
    const struct acgsolvercuda * cg,
    int indent,
    int verbose,
    MPI_Comm comm,
    int root);
#endif

#ifdef __cplusplus
}
#endif

#endif
