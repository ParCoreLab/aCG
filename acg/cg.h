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
 * conjugate gradient (CG) solver
 */

#ifndef ACG_CG_H
#define ACG_CG_H

#include "acg/config.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct acghalo;
struct acgsymcsrmatrix;

/**
 * ‘acgsolver’ is a data structure for use with a conjugate
 * gradient-based iterative solver.
 *
 * It consists of three temporary vectors, r, p and t, which are used
 * throughout the iterative solution procedure.
 */
struct acgsolver
{
    /* vectors */
    struct acgvector r;
    struct acgvector p;
    struct acgvector t;
    struct acgvector * dx;

    /* “halo exchange” communication */
    struct acghalo * halo;
    double * sendbuf;
    double * recvbuf;
    void * sendreqs;
    void * recvreqs;

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

    /* solver statistics, including the number of solves, iterations,
     * floating-point operations, and a performance breakdown of time
     * spent in different parts. */
    int nsolves, ntotaliterations, niterations;
    int64_t nflops;
    double tsolve;
    double tgemv, tdot, tnrm2, taxpy, tcopy, tmpiallreduce, tmpihalo;
    int64_t ngemv, ndot, nnrm2, naxpy, ncopy, nmpiallreduce, nmpihalo;
    int64_t Bgemv, Bdot, Bnrm2, Baxpy, Bcopy, Bmpiallreduce, Bmpihalo;
    int64_t nmpihalomsgs;
};

/*
 * memory management
 */

/**
 * ‘acgsolver_free()’ frees storage allocated for a solver.
 */
ACG_API void acgsolver_free(
    struct acgsolver * cg);

/*
 * initialise a solver
 */

/**
 * ‘acgsolver_init()’ sets up a conjugate gradient solver for a given
 * symmetric sparse matrix in CSR format.
 *
 * The matrix may be partitioned and distributed.
 */
ACG_API int acgsolver_init(
    struct acgsolver * cg,
    const struct acgsymcsrmatrix * A);

/*
 * iterative solution procedure
 */

/**
 * ‘acgsolver_solve()’ solves the given linear system, Ax=b, using the
 * conjugate gradient method.
 *
 * The solver must already have been configured with ‘acgsolver_init()’
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
ACG_API int acgsolver_solve(
    struct acgsolver * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol);

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolver_solvempi()’ solves the given linear system, Ax=b, using
 * the conjugate gradient method. The linear system may be distributed
 * across multiple processes and communication is handled using MPI.
 *
 * The solver must already have been configured with ‘acgsolver_init()’
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
ACG_API int acgsolver_solvempi(
    struct acgsolver * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    MPI_Comm comm,
    int tag,
    int * mpierrcode);
#endif

/*
 * output solver info
 */

/**
 * ‘acgsolver_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolver_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
ACG_API int acgsolver_fwrite(
    FILE * f,
    const struct acgsolver * cg,
    int indent);

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolver_fwritempi()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolver_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
ACG_API int acgsolver_fwritempi(
    FILE * f,
    const struct acgsolver * cg,
    int indent,
    MPI_Comm comm,
    int root);
#endif

#ifdef __cplusplus
}
#endif

#endif
