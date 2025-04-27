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

#include "acg/config.h"
#include "acg/cg.h"
#include "acg/halo.h"
#include "acg/symcsrmatrix.h"
#include "acg/error.h"
#include "acg/time.h"
#include "acg/vector.h"

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * memory management
 */

/**
 * ‘acgsolver_free()’ frees storage allocated for a solver.
 */
void acgsolver_free(
    struct acgsolver * cg)
{
    acgvector_free(&cg->r);
    acgvector_free(&cg->p);
    acgvector_free(&cg->t);
    if (cg->dx) acgvector_free(cg->dx);
    free(cg->dx);
    if (cg->halo) acghalo_free(cg->halo);
    free(cg->halo);
    free(cg->sendbuf);
    free(cg->recvbuf);
    free(cg->sendreqs);
    free(cg->recvreqs);
}

/*
 * initialise a solver
 */

/**
 * ‘acgsolver_init()’ sets up a conjugate gradient solver for a given
 * sparse matrix in CSR format.
 *
 * The matrix may be partitioned and distributed.
 */
int acgsolver_init(
    struct acgsolver * cg,
    const struct acgsymcsrmatrix * A)
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
        free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return err;
    }
    cg->sendbuf = malloc(cg->halo->sendsize*sizeof(*cg->sendbuf));
    if (!cg->sendbuf) {
        acghalo_free(cg->halo); free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0; i < cg->halo->sendsize; i++) cg->sendbuf[i] = 0;
    cg->recvbuf = malloc(cg->halo->recvsize*sizeof(*cg->recvbuf));
    if (!cg->recvbuf) {
        free(cg->sendbuf);
        acghalo_free(cg->halo); free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0; i < cg->halo->recvsize; i++) cg->recvbuf[i] = 0;
#ifdef ACG_HAVE_MPI
    cg->sendreqs = malloc(cg->halo->nrecipients*sizeof(MPI_Request));
    if (!cg->sendreqs) {
        free(cg->recvbuf); free(cg->sendbuf);
        acghalo_free(cg->halo); free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return ACG_ERR_ERRNO;
    }
    cg->recvreqs = malloc(cg->halo->nsenders*sizeof(MPI_Request));
    if (!cg->recvreqs) {
        free(cg->recvbuf); free(cg->sendbuf);
        acghalo_free(cg->halo); free(cg->halo);
        acgvector_free(&cg->t);
        acgvector_free(&cg->p);
        acgvector_free(&cg->r);
        return ACG_ERR_ERRNO;
    }
#else
    cg->sendreqs = NULL;
    cg->recvreqs = NULL;
#endif

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
    cg->tgemv = cg->tdot = cg->tnrm2 = cg->taxpy = cg->tcopy = 0;
    cg->ngemv = cg->ndot = cg->nnrm2 = cg->naxpy = cg->ncopy = cg->nmpiallreduce = cg->nmpihalo = 0;
    cg->Bgemv = cg->Bdot = cg->Bnrm2 = cg->Baxpy = cg->Bcopy = cg->Bmpiallreduce = cg->Bmpihalo = 0;
    cg->nmpihalomsgs = 0;
    return ACG_SUCCESS;
}

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
int acgsolver_solve(
    struct acgsolver * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol)
{
    if (b->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->r.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->p.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->t.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    /* If the stopping criterion is based on the difference in
     * solution from one iteration to the next, then allocate
     * additional storage for storing the difference. */
    if ((diffatol > 0 || diffrtol > 0) && !cg->dx) {
        cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->dx, x); if (err) return err;
    }

    /* set initial state */
    acgtime_t t0, t1;
    acgtime_t tgemv0, tgemv1;
    acgtime_t tdot0, tdot1;
    acgtime_t tnrm20, tnrm21;
    acgtime_t taxpy0, taxpy1;
    acgtime_t tcopy0, tcopy1;
    cg->nsolves++; cg->niterations = 0;
    cg->bnrm2 = INFINITY;
    cg->r0nrm2 = cg->rnrm2 = INFINITY;
    cg->x0nrm2 = cg->dxnrm2 = INFINITY;
    cg->maxits = maxits;
    cg->diffatol = diffatol;
    cg->diffrtol = diffrtol;
    cg->residualatol = residualatol;
    cg->residualrtol = residualrtol;
    gettime(&t0);

    /* compute right-hand side norm */
    gettime(&tnrm20);
    int err = acgvector_dnrm2(b, &cg->bnrm2, &cg->nflops, &cg->Bnrm2);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);

    /* compute norm of initial guess */
    if (diffatol > 0 || diffrtol > 0) {
        gettime(&tnrm20);
        err = acgvector_dnrm2(x, &cg->x0nrm2, &cg->nflops, &cg->Bnrm2);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
        diffrtol *= cg->x0nrm2;
    }

    /* compute initial residual, r₀ = b-A*x₀ */
    gettime(&tcopy0);
    err = acgvector_copy(&cg->r, b, &cg->Bcopy);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tcopy1); cg->ncopy++; cg->tcopy += elapsed(tcopy0,tcopy1);
    gettime(&tgemv0);
    err = acgsymcsrmatrix_dsymv(-1.0, A, x, 1.0, &cg->r, &cg->nflops, &cg->Bgemv);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tgemv1); cg->ngemv++; cg->tgemv += elapsed(tgemv0,tgemv1);

    /* compute initial search direction: p = r₀ */
    gettime(&tcopy0);
    err = acgvector_copy(&cg->p, &cg->r, &cg->Bcopy);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tcopy1); cg->ncopy++; cg->tcopy += elapsed(tcopy0,tcopy1);

    /* compute initial residual norm */
    double rnrm2sqr;
    gettime(&tnrm20);
    err = acgvector_dnrm2sqr(&cg->r, &rnrm2sqr, &cg->nflops, &cg->Bnrm2);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    cg->rnrm2 = cg->r0nrm2 = sqrt(rnrm2sqr);
    gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
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
        gettime(&tgemv0);
        err = acgsymcsrmatrix_dsymv(
            1.0, A, &cg->p, 0.0, &cg->t, &cg->nflops, &cg->Bgemv);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tgemv1); cg->ngemv++; cg->tgemv += elapsed(tgemv0,tgemv1);

        /* compute α = (r,r) / (p,Ap) */
        double alpha;
        gettime(&tdot0);
        err = acgvector_ddot(&cg->p, &cg->t, &alpha, &cg->nflops, &cg->Bdot);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tdot1); cg->ndot++; cg->tdot += elapsed(tdot0,tdot1);
        if (alpha == 0) return ACG_ERR_NOT_CONVERGED_INDEFINITE_MATRIX;
        alpha = rnrm2sqr / alpha;

        /* update solution, x = αp + x */
        if (diffatol > 0 || diffrtol > 0) {
            gettime(&tcopy0);
            err = acgvector_copy(cg->dx, x, &cg->Bcopy);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            gettime(&tcopy1); cg->ncopy++; cg->tcopy += elapsed(tcopy0,tcopy1);
        }
        gettime(&taxpy0);
        err = acgvector_daxpy(alpha, &cg->p, x, &cg->nflops, &cg->Baxpy);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&taxpy1); cg->naxpy++; cg->taxpy += elapsed(taxpy0,taxpy1);
        if (diffatol > 0 || diffrtol > 0) {
            gettime(&taxpy0);
            err = acgvector_daxpy(-1.0, x, cg->dx, &cg->nflops, &cg->Baxpy);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            gettime(&taxpy1); cg->naxpy++; cg->taxpy += elapsed(taxpy0,taxpy1);
            gettime(&tnrm20);
            err = acgvector_dnrm2(cg->dx, &cg->dxnrm2, &cg->nflops, &cg->Bnrm2);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
        }

        /* update residual, r = -αt + r */
        gettime(&taxpy0);
        err = acgvector_daxpy(-alpha, &cg->t, &cg->r, &cg->nflops, &cg->Baxpy);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&taxpy1); cg->naxpy++; cg->taxpy += elapsed(taxpy0,taxpy1);

        /* compute residual norm */
        double rnrm2sqr_prev = rnrm2sqr;
        gettime(&tnrm20);
        err = acgvector_dnrm2sqr(&cg->r, &rnrm2sqr, &cg->nflops, &cg->Bnrm2);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        cg->rnrm2 = sqrt(rnrm2sqr);
        gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);

        /* convergence tests */
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) ||
            (diffrtol > 0 && cg->dxnrm2 < diffrtol) ||
            (residualatol > 0 && cg->rnrm2 < residualatol) ||
            (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            cg->ntotaliterations++; cg->niterations++;
            gettime(&t1); cg->tsolve += elapsed(t0,t1);
            return ACG_SUCCESS;
        }

        /* β = (rₖ₊₁,rₖ₊₁)/(rₖ,rₖ) */
        double beta;
        if (rnrm2sqr_prev == 0) return ACG_ERR_NOT_CONVERGED_INDEFINITE_MATRIX;
        beta = rnrm2sqr / rnrm2sqr_prev;

        /* update search direction, p = βp + r */
        gettime(&taxpy0);
        err = acgvector_daypx(beta, &cg->p, &cg->r, &cg->nflops, &cg->Baxpy);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&taxpy1); cg->naxpy++; cg->taxpy += elapsed(taxpy0,taxpy1);
        cg->ntotaliterations++; cg->niterations++;
    }
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

    /* if the only stopping criteria is a maximum number of
     * iterations, then the solver succeeded */
    if (diffatol == 0 && diffrtol == 0 &&
        residualatol == 0 && residualrtol == 0)
        return ACG_SUCCESS;

    /* otherwise, the solver failed to converge with the given number
     * of maximum iterations */
    return ACG_ERR_NOT_CONVERGED;
}

/*
 * iterative solution procedure in distributed memory using MPI
 */

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
int acgsolver_solvempi(
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
    int * mpierrcode)
{
    if (b->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->r.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->p.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (cg->t.size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    /* If the stopping criterion is based on the difference in
     * solution from one iteration to the next, then allocate
     * additional storage for storing the difference. */
    if ((diffatol > 0 || diffrtol > 0) && !cg->dx) {
        cg->dx = malloc(sizeof(*cg->dx)); if (!cg->dx) return ACG_ERR_ERRNO;
        int err = acgvector_init_copy(cg->dx, x); if (err) return err;
    }

    /* set initial state */
    acgtime_t t0, t1;
    acgtime_t tgemv0, tgemv1;
    acgtime_t tdot0, tdot1;
    acgtime_t tnrm20, tnrm21;
    acgtime_t taxpy0, taxpy1;
    acgtime_t tcopy0, tcopy1;
    acgtime_t tmpiallreduce0, tmpiallreduce1;
    acgtime_t tmpihalo0, tmpihalo1;
    cg->nsolves++; cg->niterations = 0;
    cg->bnrm2 = INFINITY;
    cg->r0nrm2 = cg->rnrm2 = INFINITY;
    cg->x0nrm2 = cg->dxnrm2 = INFINITY;
    cg->maxits = maxits;
    cg->diffatol = diffatol;
    cg->diffrtol = diffrtol;
    cg->residualatol = residualatol;
    cg->residualrtol = residualrtol;
    gettime(&t0);

    /* compute right-hand side norm */
    gettime(&tnrm20);
    double bnrm2sqr;
    int err = acgvector_dnrm2sqr(b, &bnrm2sqr, &cg->nflops, &cg->Bnrm2);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
    gettime(&tmpiallreduce0);
    err = MPI_Allreduce(MPI_IN_PLACE, &bnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *mpierrcode = err; return ACG_ERR_MPI; }
    cg->Bmpiallreduce += sizeof(bnrm2sqr);
    gettime(&tmpiallreduce1); cg->nmpiallreduce++; cg->tmpiallreduce += elapsed(tmpiallreduce0,tmpiallreduce1);
    cg->bnrm2 = sqrt(bnrm2sqr);

    /* compute norm of initial guess */
    if (diffatol > 0 || diffrtol > 0) {
        gettime(&tnrm20);
        double x0nrm2sqr;
        err = acgvector_dnrm2sqr(x, &x0nrm2sqr, &cg->nflops, &cg->Bnrm2);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tnrm21); cg->nnrm2++; cg->tnrm2 += elapsed(tnrm20,tnrm21);
        gettime(&tmpiallreduce0);
        err = MPI_Allreduce(MPI_IN_PLACE, &x0nrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *mpierrcode = err; return ACG_ERR_MPI; }
        cg->Bmpiallreduce += sizeof(x0nrm2sqr);
        gettime(&tmpiallreduce1); cg->nmpiallreduce++; cg->tmpiallreduce += elapsed(tmpiallreduce0,tmpiallreduce1);
        cg->x0nrm2 = sqrt(x0nrm2sqr);
        diffrtol *= cg->x0nrm2;
    }

    /* compute initial residual, r₀ = b-A*x₀ */
    gettime(&tcopy0);
    err = acgvector_copy(&cg->r, b, &cg->Bcopy);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tcopy1); cg->ncopy++; cg->tcopy += elapsed(tcopy0,tcopy1); gettime(&tmpihalo0);
    err = acghalo_exchange(
        cg->halo, x->num_nonzeros, x->x, MPI_DOUBLE,
        x->num_nonzeros, x->x, MPI_DOUBLE,
        cg->halo->sendsize, cg->sendbuf, cg->sendreqs,
        cg->halo->recvsize, cg->recvbuf, cg->recvreqs,
        comm, tag, mpierrcode);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    cg->Bmpihalo += cg->halo->sendsize*sizeof(*x->x);
    cg->nmpihalomsgs += cg->halo->nrecipients;
    gettime(&tmpihalo1); cg->nmpihalo++; cg->tmpihalo += elapsed(tmpihalo0,tmpihalo1);
    gettime(&tgemv0);
    err = acgsymcsrmatrix_dsymv(-1.0, A, x, 1.0, &cg->r, &cg->nflops, &cg->Bgemv);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tgemv1); cg->tgemv += elapsed(tgemv0,tgemv1); cg->ngemv++;

    /* compute initial search direction: p = r₀ */
    gettime(&tcopy0);
    err = acgvector_copy(&cg->p, &cg->r, &cg->Bcopy);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tcopy1); cg->ncopy++; cg->tcopy += elapsed(tcopy0,tcopy1);

    /* compute initial residual norm */
    double rnrm2sqr;
    gettime(&tnrm20);
    err = acgvector_dnrm2sqr(&cg->r, &rnrm2sqr, &cg->nflops, &cg->Bnrm2);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
    gettime(&tnrm21); cg->tnrm2 += elapsed(tnrm20,tnrm21); cg->nnrm2++;
    gettime(&tmpiallreduce0);
    err = MPI_Allreduce(MPI_IN_PLACE, &rnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *mpierrcode = err; return ACG_ERR_MPI; }
    cg->Bmpiallreduce += sizeof(rnrm2sqr);
    gettime(&tmpiallreduce1); cg->tmpiallreduce += elapsed(tmpiallreduce0,tmpiallreduce1); cg->nmpiallreduce++;
    cg->rnrm2 = cg->r0nrm2 = sqrt(rnrm2sqr);
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
        gettime(&tmpihalo0);
        err = acghalo_exchange(
            cg->halo, cg->p.num_nonzeros, cg->p.x, MPI_DOUBLE,
            cg->p.num_nonzeros, cg->p.x, MPI_DOUBLE,
            cg->halo->sendsize, cg->sendbuf, cg->sendreqs,
            cg->halo->recvsize, cg->recvbuf, cg->recvreqs,
            comm, tag, mpierrcode);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        cg->Bmpihalo += cg->halo->sendsize*sizeof(*cg->p.x);
        cg->nmpihalomsgs += cg->halo->nrecipients;
        gettime(&tmpihalo1); cg->tmpihalo += elapsed(tmpihalo0,tmpihalo1); cg->nmpihalo++;
        gettime(&tgemv0);
        err = acgsymcsrmatrix_dsymv(
            1.0, A, &cg->p, 0.0, &cg->t, &cg->nflops, &cg->Bgemv);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tgemv1); cg->tgemv += elapsed(tgemv0,tgemv1); cg->ngemv++;

        /* compute (p,Ap) */
        double pdott;
        gettime(&tdot0);
        err = acgvector_ddot(&cg->p, &cg->t, &pdott, &cg->nflops, &cg->Bdot);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tdot1); cg->tdot += elapsed(tdot0,tdot1); cg->ndot++;
        gettime(&tmpiallreduce0);
        err = MPI_Allreduce(MPI_IN_PLACE, &pdott, 1, MPI_DOUBLE, MPI_SUM, comm);
        cg->Bmpiallreduce += sizeof(pdott);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *mpierrcode = err; return ACG_ERR_MPI; }
        gettime(&tmpiallreduce1); cg->tmpiallreduce += elapsed(tmpiallreduce0,tmpiallreduce1); cg->nmpiallreduce++;

        /* compute α = (r,r) / (p,Ap) */
        double alpha;
        alpha = rnrm2sqr / pdott;

        /* update solution, x = αp + x */
        if (diffatol > 0 || diffrtol > 0) {
            gettime(&tcopy0);
            err = acgvector_copy(cg->dx, x, &cg->Bcopy);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            gettime(&tcopy1); cg->tcopy += elapsed(tcopy0,tcopy1); cg->ncopy++;
        }
        gettime(&taxpy0);
        err = acgvector_daxpy(alpha, &cg->p, x, &cg->nflops, &cg->Baxpy);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&taxpy1); cg->taxpy += elapsed(taxpy0,taxpy1); cg->naxpy++;
        if (diffatol > 0 || diffrtol > 0) {
            gettime(&taxpy0);
            err = acgvector_daxpy(-1.0, x, cg->dx, &cg->nflops, &cg->Baxpy);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            gettime(&taxpy1); cg->taxpy += elapsed(taxpy0,taxpy1); cg->naxpy++;
            gettime(&tnrm20);
            double dxnrm2sqr;
            err = acgvector_dnrm2sqr(cg->dx, &dxnrm2sqr, &cg->nflops, &cg->Bnrm2);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
            gettime(&tnrm21); cg->tnrm2 += elapsed(tnrm20,tnrm21); cg->nnrm2++;
            gettime(&tmpiallreduce0);
            err = MPI_Allreduce(MPI_IN_PLACE, &dxnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm);
            if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *mpierrcode = err; return ACG_ERR_MPI; }
            cg->Bmpiallreduce += sizeof(dxnrm2sqr);
            gettime(&tmpiallreduce1); cg->tmpiallreduce += elapsed(tmpiallreduce0,tmpiallreduce1); cg->nmpiallreduce++;
            cg->dxnrm2 = sqrt(dxnrm2sqr);
        }

        /* update residual, r = -αt + r */
        gettime(&taxpy0);
        err = acgvector_daxpy(-alpha, &cg->t, &cg->r, &cg->nflops, &cg->Baxpy);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&taxpy1); cg->taxpy += elapsed(taxpy0,taxpy1); cg->naxpy++;

        /* compute residual norm */
        double rnrm2sqr_prev = rnrm2sqr;
        gettime(&tnrm20);
        err = acgvector_dnrm2sqr(&cg->r, &rnrm2sqr, &cg->nflops, &cg->Bnrm2);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&tnrm21); cg->tnrm2 += elapsed(tnrm20,tnrm21); cg->nnrm2++;
        gettime(&tmpiallreduce0);
        err = MPI_Allreduce(MPI_IN_PLACE, &rnrm2sqr, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); *mpierrcode = err; return ACG_ERR_MPI; }
        cg->Bmpiallreduce += sizeof(rnrm2sqr);
        gettime(&tmpiallreduce1); cg->tmpiallreduce += elapsed(tmpiallreduce0,tmpiallreduce1); cg->nmpiallreduce++;
        cg->rnrm2 = sqrt(rnrm2sqr);

        /* convergence tests */
        if ((diffatol > 0 && cg->dxnrm2 < diffatol) ||
            (diffrtol > 0 && cg->dxnrm2 < diffrtol) ||
            (residualatol > 0 && cg->rnrm2 < residualatol) ||
            (residualrtol > 0 && cg->rnrm2 < residualrtol))
        {
            cg->ntotaliterations++; cg->niterations++;
            gettime(&t1); cg->tsolve += elapsed(t0,t1);
            return ACG_SUCCESS;
        }

        /* β = (rₖ₊₁,rₖ₊₁)/(rₖ,rₖ) */
        double beta = rnrm2sqr / rnrm2sqr_prev;

        /* update search direction, p = βp + r */
        gettime(&taxpy0);
        err = acgvector_daypx(beta, &cg->p, &cg->r, &cg->nflops, &cg->Baxpy);
        if (err) { gettime(&t1); cg->tsolve += elapsed(t0,t1); return err; }
        gettime(&taxpy1); cg->taxpy += elapsed(taxpy0,taxpy1); cg->naxpy++;
        cg->ntotaliterations++; cg->niterations++;
    }
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

    /* if the only stopping criteria is a maximum number of
     * iterations, then the solver succeeded */
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
 * ‘acgsolver_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolver_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolver_fwrite(
    FILE * f,
    const struct acgsolver * cg,
    int indent)
{
    double tother = cg->tsolve -
        (cg->tgemv+cg->tdot+cg->tnrm2+cg->taxpy+cg->tcopy
         +cg->tmpiallreduce+cg->tmpihalo);
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
                               cg->tmpiallreduce, cg->nmpiallreduce, cg->Bmpiallreduce, cg->tmpiallreduce>0 ? 1.0e-9*cg->Bmpiallreduce/cg->tmpiallreduce : 0.0);
    findent(f,indent); fprintf(f, "  MPI_HaloExchange: %'.6f seconds %'"PRId64" times %'"PRId64" B %'.3f GB/s\n",
                               cg->tmpihalo, cg->nmpihalo, cg->Bmpihalo, cg->tmpihalo>0 ? 1.0e-9*cg->Bmpihalo/cg->tmpihalo : 0.0);
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
 * ‘acgsolver_fwritempi()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolver_solvempi()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolver_fwritempi(
    FILE * f,
    const struct acgsolver * cg,
    int indent,
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
    double tmpiallreduce = cg->tmpiallreduce;
    double tmpihalo = cg->tmpihalo;
    double tmpihalopack = cg->halo->tpack;
    double tmpihalounpack = cg->halo->tunpack;
    double tmpihalompiirecv = cg->halo->tmpiirecv;
    double tmpihalompisend = cg->halo->tmpisend;
    double tmpihalompiwaitall = cg->halo->tmpiwaitall;
    double tother = tsolve-(tgemv+tdot+tnrm2+taxpy+tcopy+tmpiallreduce+tmpihalo);
    int64_t ngemv = cg->ngemv, Bgemv = cg->Bgemv;
    int64_t ndot = cg->ndot, Bdot = cg->Bdot;
    int64_t nnrm2 = cg->nnrm2, Bnrm2 = cg->Bnrm2;
    int64_t naxpy = cg->naxpy, Baxpy = cg->Baxpy;
    int64_t ncopy = cg->ncopy, Bcopy = cg->Bcopy;
    int64_t nmpiallreduce = cg->nmpiallreduce, Bmpiallreduce = cg->Bmpiallreduce;
    int64_t nmpihalo = cg->nmpihalo, Bmpihalo = cg->Bmpihalo;
    int64_t nmpihalopack = cg->halo->npack, Bmpihalopack = cg->halo->Bpack;
    int64_t nmpihalounpack = cg->halo->nunpack, Bmpihalounpack = cg->halo->Bunpack;
    int64_t nmpihalompiirecv = cg->halo->nmpiirecv, Bmpihalompiirecv = cg->halo->Bmpiirecv;
    int64_t nmpihalompisend = cg->halo->nmpisend, Bmpihalompisend = cg->halo->Bmpisend;
    int64_t nmpihalomsgs = cg->nmpihalomsgs;
    MPI_Reduce(&cg->nflops, &nflops, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->tsolve, &tsolve, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    MPI_Reduce(&cg->tgemv, &tgemv, 1, MPI_DOUBLE, MPI_SUM, root, comm); tgemv /= commsize;
    MPI_Reduce(&cg->tdot, &tdot, 1, MPI_DOUBLE, MPI_SUM, root, comm); tdot /= commsize;
    MPI_Reduce(&cg->tnrm2, &tnrm2, 1, MPI_DOUBLE, MPI_SUM, root, comm); tnrm2 /= commsize;
    MPI_Reduce(&cg->taxpy, &taxpy, 1, MPI_DOUBLE, MPI_SUM, root, comm); taxpy /= commsize;
    MPI_Reduce(&cg->tcopy, &tcopy, 1, MPI_DOUBLE, MPI_SUM, root, comm); tcopy /= commsize;
    MPI_Reduce(&cg->tmpiallreduce, &tmpiallreduce, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpiallreduce /= commsize;
    MPI_Reduce(&cg->tmpihalo, &tmpihalo, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpihalo /= commsize;
    MPI_Reduce(&cg->halo->tpack, &tmpihalopack, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpihalopack /= commsize;
    MPI_Reduce(&cg->halo->tunpack, &tmpihalounpack, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpihalounpack /= commsize;
    MPI_Reduce(&cg->halo->tmpiirecv, &tmpihalompiirecv, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpihalompiirecv /= commsize;
    MPI_Reduce(&cg->halo->tmpisend, &tmpihalompisend, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpihalompisend /= commsize;
    MPI_Reduce(&cg->halo->tmpiwaitall, &tmpihalompiwaitall, 1, MPI_DOUBLE, MPI_SUM, root, comm); tmpihalompiwaitall /= commsize;
    MPI_Reduce(rank == root ? MPI_IN_PLACE : &tother, &tother, 1, MPI_DOUBLE, MPI_SUM, root, comm); tother /= commsize;
    MPI_Reduce(&cg->ngemv, &ngemv, 1, MPI_INT64_T, MPI_SUM, root, comm); ngemv /= commsize;
    MPI_Reduce(&cg->ndot, &ndot, 1, MPI_INT64_T, MPI_SUM, root, comm); ndot /= commsize;
    MPI_Reduce(&cg->nnrm2, &nnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm); nnrm2 /= commsize;
    MPI_Reduce(&cg->naxpy, &naxpy, 1, MPI_INT64_T, MPI_SUM, root, comm); naxpy /= commsize;
    MPI_Reduce(&cg->ncopy, &ncopy, 1, MPI_INT64_T, MPI_SUM, root, comm); ncopy /= commsize;
    MPI_Reduce(&cg->nmpiallreduce, &nmpiallreduce, 1, MPI_INT64_T, MPI_SUM, root, comm); nmpiallreduce /= commsize;
    MPI_Reduce(&cg->nmpihalo, &nmpihalo, 1, MPI_INT64_T, MPI_SUM, root, comm); nmpihalo /= commsize;
    MPI_Reduce(&cg->halo->npack, &nmpihalopack, 1, MPI_INT64_T, MPI_SUM, root, comm); nmpihalopack /= commsize;
    MPI_Reduce(&cg->halo->nunpack, &nmpihalounpack, 1, MPI_INT64_T, MPI_SUM, root, comm); nmpihalounpack /= commsize;
    MPI_Reduce(&cg->halo->nmpiirecv, &nmpihalompiirecv, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->halo->nmpisend, &nmpihalompisend, 1, MPI_INT64_T, MPI_SUM, root, comm);
    MPI_Reduce(&cg->Bgemv, &Bgemv, 1, MPI_INT64_T, MPI_SUM, root, comm); Bgemv /= commsize;
    MPI_Reduce(&cg->Bdot, &Bdot, 1, MPI_INT64_T, MPI_SUM, root, comm); Bdot /= commsize;
    MPI_Reduce(&cg->Bnrm2, &Bnrm2, 1, MPI_INT64_T, MPI_SUM, root, comm); Bnrm2 /= commsize;
    MPI_Reduce(&cg->Baxpy, &Baxpy, 1, MPI_INT64_T, MPI_SUM, root, comm); Baxpy /= commsize;
    MPI_Reduce(&cg->Bcopy, &Bcopy, 1, MPI_INT64_T, MPI_SUM, root, comm); Bcopy /= commsize;
    MPI_Reduce(&cg->Bmpiallreduce, &Bmpiallreduce, 1, MPI_INT64_T, MPI_SUM, root, comm); Bmpiallreduce /= commsize;
    MPI_Reduce(&cg->Bmpihalo, &Bmpihalo, 1, MPI_INT64_T, MPI_SUM, root, comm); Bmpihalo /= commsize;
    MPI_Reduce(&cg->halo->Bpack, &Bmpihalopack, 1, MPI_INT64_T, MPI_SUM, root, comm); Bmpihalopack /= commsize;
    MPI_Reduce(&cg->halo->Bunpack, &Bmpihalounpack, 1, MPI_INT64_T, MPI_SUM, root, comm); Bmpihalounpack /= commsize;
    MPI_Reduce(&cg->halo->Bmpiirecv, &Bmpihalompiirecv, 1, MPI_INT64_T, MPI_SUM, root, comm); Bmpihalompiirecv /= commsize;
    MPI_Reduce(&cg->halo->Bmpisend, &Bmpihalompisend, 1, MPI_INT64_T, MPI_SUM, root, comm); Bmpihalompisend /= commsize;
    MPI_Reduce(&cg->nmpihalomsgs, &nmpihalomsgs, 1, MPI_INT64_T, MPI_SUM, root, comm);
    if (rank == root) {
        findent(f,indent); fprintf(f, "unknowns: %'"PRIdx"\n", cg->p.size);
        findent(f,indent); fprintf(f, "solves: %'d\n", cg->nsolves);
        findent(f,indent); fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
        findent(f,indent); fprintf(f, "total flops: %'.3f Gflop\n", 1.0e-9*nflops);
        findent(f,indent); fprintf(f, "total flop rate: %'.3f Gflop/s\n", tsolve > 0 ? 1.0e-9*nflops/tsolve : 0);
        findent(f,indent); fprintf(f, "total solver time: %'.6f seconds\n", tsolve);
        findent(f,indent); fprintf(f, "performance breakdown:\n");
        findent(f,indent); fprintf(f, "  gemv: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tgemv, ngemv, Bgemv, tgemv>0 ? 1.0e-9*Bgemv/tgemv : 0.0);
        findent(f,indent); fprintf(f, "  dot: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tdot, ndot, Bdot, tdot>0 ? 1.0e-9*Bdot/tdot : 0.0);
        findent(f,indent); fprintf(f, "  nrm2: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tnrm2, nnrm2, Bnrm2, tnrm2>0 ? 1.0e-9*Bnrm2/tnrm2 : 0.0);
        findent(f,indent); fprintf(f, "  axpy: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   taxpy, naxpy, Baxpy, taxpy>0 ? 1.0e-9*Baxpy/taxpy : 0.0);
        findent(f,indent); fprintf(f, "  copy: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tcopy, ncopy, Bcopy, tcopy>0 ? 1.0e-9*Bcopy/tcopy : 0.0);
        findent(f,indent); fprintf(f, "  MPI_Allreduce: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc %'.3f µs/op/proc\n",
                                   tmpiallreduce, nmpiallreduce, Bmpiallreduce, tmpiallreduce>0 ? 1.0e-9*Bmpiallreduce/tmpiallreduce : 0.0,
                                   nmpiallreduce>0 ? 1.0e6*tmpiallreduce/nmpiallreduce : 0.0);
        findent(f,indent); fprintf(f, "  MPI_HaloExchange: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc %'.1f msg/proc %'.3f µs/msg/proc\n",
                                   tmpihalo, nmpihalo, Bmpihalo, tmpihalo>0 ? 1.0e-9*Bmpihalo/tmpihalo : 0.0,
                                   ((double) nmpihalomsgs)/commsize, nmpihalomsgs>0 ? 1.0e6*tmpihalo/nmpihalomsgs/commsize : 0.0);
        findent(f,indent); fprintf(f, "    pack: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tmpihalopack, nmpihalopack, Bmpihalopack, tmpihalopack>0 ? 1.0e-9*Bmpihalopack/tmpihalopack : 0.0);
        findent(f,indent); fprintf(f, "    unpack: %'.6f seconds/proc %'"PRId64" times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tmpihalounpack, nmpihalounpack, Bmpihalounpack, tmpihalounpack>0 ? 1.0e-9*Bmpihalounpack/tmpihalounpack : 0.0);
        findent(f,indent); fprintf(f, "    MPI_Recv: %'.6f seconds/proc %'.1f times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tmpihalompiirecv, (double)nmpihalompiirecv/commsize, Bmpihalompiirecv, tmpihalompiirecv>0 ? 1.0e-9*Bmpihalompiirecv/tmpihalompiirecv : 0.0);
        findent(f,indent); fprintf(f, "    MPI_Send: %'.6f seconds/proc %'.1f times/proc %'"PRId64" B/proc %'.3f GB/s/proc\n",
                                   tmpihalompisend, (double)nmpihalompisend/commsize, Bmpihalompisend, tmpihalompisend>0 ? 1.0e-9*Bmpihalompisend/tmpihalompisend : 0.0);
        findent(f,indent); fprintf(f, "    MPI_Waitall: %'.6f seconds/proc %'"PRId64" times/proc\n", tmpihalompiwaitall, nmpihalo);
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
