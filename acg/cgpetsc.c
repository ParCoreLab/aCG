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
 * conjugate gradient (CG) solver based on PETSc
 */

#include "acg/config.h"
#include "acg/cgpetsc.h"
#include "acg/comm.h"
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

#ifdef ACG_HAVE_PETSC
#include <petsc.h>
#endif

/*
 * PETSc data structures
 */

#ifdef ACG_HAVE_PETSC
struct acgpetsc
{
    Mat A;
    Vec b, x;
    KSP ksp;
    PC pc;
};

static int acgpetsc_init(
    struct acgpetsc * petsc,
    const struct acgsymcsrmatrix * A,
    PetscDeviceType devicetype,
    KSPType ksptype,
    MPI_Comm comm)
{
    int err;
    if (sizeof(PetscInt) != sizeof(acgidx_t)) return ACG_ERR_NOT_SUPPORTED;

    /* convert the Matrix to PETSc format */
    err = MatCreate(PETSC_COMM_WORLD, &petsc->A); CHKERRQ(err);
    err = MatSetSizes(
        petsc->A, A->nownedrows, A->nownedrows, A->nrows, A->nrows); CHKERRQ(err);
    /* err = MatSetFromOptions(petsc->A); */

    /* determine and set matrix/vector type based on device type */
    MatType mattype; VecType vectype;
    if (devicetype == PETSC_DEVICE_HOST) {
        mattype = MATMPIAIJ; vectype = VECSTANDARD;
    } else if (devicetype == PETSC_DEVICE_CUDA) {
        mattype = MATMPIAIJCUSPARSE; vectype = VECCUDA;
    } else if (devicetype == PETSC_DEVICE_HIP) {
        mattype = MATMPIAIJHIPSPARSE; vectype = VECHIP;
    } else { return ACG_ERR_INVALID_VALUE; }
    err = MatSetType(petsc->A, mattype); CHKERRQ(err);

    int commsize, rank;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    /* set up halo communication pattern */
    struct acghalo halo;
    err = acgsymcsrmatrix_halo(A, &halo);
    if (err) return err;

    /* map matrix rows to PETSc's global numbering */
    acgidx_t rowoffset = 0;
    MPI_Exscan(&A->nownedrows, &rowoffset, 1, MPI_ACGIDX_T, MPI_SUM, comm);
    PetscInt * rowmap = malloc(A->nprows*sizeof(*rowmap));
    if (!rowmap) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < A->nownedrows; i++)
        rowmap[i] = rowoffset + i;
    int mpierrcode;
    err = acghalo_exchange(
        &halo, A->nprows, rowmap, MPIU_INT,
        A->nprows, rowmap, MPIU_INT,
        0, NULL, NULL, 0, NULL, NULL,
        comm, 22, &mpierrcode);
    if (err) return err;
    acghalo_free(&halo);
    ISLocalToGlobalMapping rowis;
    err = ISLocalToGlobalMappingCreate(
        comm, 1, A->nprows, rowmap, PETSC_COPY_VALUES, &rowis); CHKERRQ(err);
    err = MatSetLocalToGlobalMapping(petsc->A, rowis, rowis); CHKERRQ(err);

    /* preallocate storage for nonzeros */
    PetscInt * rowcountd = malloc(A->nownedrows*sizeof(*rowcountd));
    if (!rowcountd) return ACG_ERR_ERRNO;
    PetscInt * rowcounto = malloc(A->nownedrows*sizeof(*rowcounto));
    if (!rowcounto) return ACG_ERR_ERRNO;
    PetscInt nghostnzs = 0;
    for (PetscInt i = 0; i < A->nownedrows; i++) {
        rowcountd[i] = A->frowptr[i+1]-A->frowptr[i];
        rowcounto[i] = 0;
    }
    for (PetscInt i = 0; i < A->nborderrows; i++)
        rowcounto[A->borderrowoffset+i] = A->orowptr[i+1]-A->orowptr[i];
    err = MatSeqAIJSetPreallocation(petsc->A, 0, rowcountd); CHKERRQ(err);
    err = MatMPIAIJSetPreallocation(
        petsc->A, 0, rowcountd, 0, rowcounto); CHKERRQ(err);

    /* calculate global column indices for off-diagonal entries */
    PetscInt * colidx = malloc(A->onpnzs*sizeof(*colidx));
    if (!colidx) return ACG_ERR_ERRNO;
    for (PetscInt i = 0; i < A->nborderrows; i++) {
        for (int64_t k = A->orowptr[i]; k < A->orowptr[i+1]; k++) {
            acgidx_t j = A->borderrowoffset+A->ocolidx[k];
            colidx[k] = rowmap[j];
        }
    }

    /* insert nonzeros */
    err = MatAssemblyBegin(petsc->A, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
    for (PetscInt i = 0; i < A->nownedrows; i++) {
        if (rowcountd[i] > 0) {
            err = MatSetValuesLocal(
                petsc->A, 1, &i, rowcountd[i],
                &A->fcolidx[A->frowptr[i]], &A->fa[A->frowptr[i]],
                INSERT_VALUES);
            CHKERRQ(err);
        }
    }
    for (PetscInt i = 0; i < A->nborderrows; i++) {
        if (rowcounto[A->borderrowoffset+i] > 0) {
            err = MatSetValues(
                petsc->A, 1, &rowmap[A->borderrowoffset+i], rowcounto[A->borderrowoffset+i],
                &colidx[A->orowptr[i]], &A->oa[A->orowptr[i]],
                INSERT_VALUES);
            CHKERRQ(err);
        }
    }
    err = MatAssemblyEnd(petsc->A, MAT_FINAL_ASSEMBLY); CHKERRQ(err);
    err = ISLocalToGlobalMappingDestroy(&rowis); CHKERRQ(err);
    free(colidx); free(rowmap);

    /* allocate compatible right-hand side and solution vectors */
    err = MatCreateVecs(petsc->A, &petsc->x, &petsc->b); CHKERRQ(err);
    err = VecSetType(petsc->x, vectype); CHKERRQ(err);
    err = VecSetType(petsc->b, vectype); CHKERRQ(err);

    /* set up solver */
    err = KSPCreate(comm, &petsc->ksp); CHKERRQ(err);

    err = KSPSetType(petsc->ksp, ksptype); CHKERRQ(err);
    err = KSPSetNormType(petsc->ksp, KSP_NORM_UNPRECONDITIONED); CHKERRQ(err);
    err = KSPConvergedDefaultSetUIRNorm(petsc->ksp); CHKERRQ(err);
    err = KSPSetOperators(petsc->ksp, petsc->A, petsc->A); CHKERRQ(err);
    err = KSPSetUp(petsc->ksp); CHKERRQ(err);
    /* err = KSPSetFromOptions(petsc->ksp); */

    /* set up preconditioner */
    err = PCCreate(comm, &petsc->pc); CHKERRQ(err);
    err = PCSetType(petsc->pc, PCNONE); CHKERRQ(err);
    err = PCSetOperators(petsc->pc, petsc->A, petsc->A);
    err = KSPSetPC(petsc->ksp, petsc->pc); CHKERRQ(err);
    err = PCSetUp(petsc->pc); CHKERRQ(err);
    return ACG_SUCCESS;
}

void acgpetsc_free(
    struct acgpetsc * petsc)
{
    PCDestroy(&petsc->pc);
    KSPDestroy(&petsc->ksp);
    VecDestroy(&petsc->x);
    VecDestroy(&petsc->b);
    MatDestroy(&petsc->A);
}
#endif

/*
 * memory management
 */

/**
 * ‘acgsolverpetsc_free()’ frees storage allocated for a solver.
 */
void acgsolverpetsc_free(
    struct acgsolverpetsc * cg)
{
#ifdef ACG_HAVE_PETSC
    acgpetsc_free(cg->petsc);
#endif
}

/*
 * initialise a solver
 */

/**
 * ‘acgsolverpetsc_init()’ sets up a conjugate gradient solver for a
 * given sparse matrix in CSR format.
 *
 * The matrix may be partitioned and distributed.
 */
int acgsolverpetsc_init(
    struct acgsolverpetsc * cg,
    const struct acgsymcsrmatrix * A,
    enum acgdevicetype acgdevicetype,
    enum acgpetscksptype acgpetscksptype,
    const struct acgcomm * comm)
{
#ifndef ACG_HAVE_PETSC
    return ACG_ERR_PETSC_NOT_SUPPORTED;
#else
    int err;
    PetscDeviceType petscdevicetype;
    if (acgdevicetype == ACG_DEVICE_HOST) petscdevicetype = PETSC_DEVICE_HOST;
    else if (acgdevicetype == ACG_DEVICE_CUDA) petscdevicetype = PETSC_DEVICE_CUDA;
    else if (acgdevicetype == ACG_DEVICE_HIP) petscdevicetype = PETSC_DEVICE_HIP;
    else return ACG_ERR_INVALID_VALUE;
    if (comm->type != acgcomm_mpi) return ACG_ERR_INVALID_VALUE;
    KSPType petscksptype;
    if (acgpetscksptype == PETSC_KSPCG) petscksptype = KSPCG;
    else if (acgpetscksptype == PETSC_KSPPIPECG) petscksptype = KSPPIPECG;
    else return ACG_ERR_INVALID_VALUE;
    cg->petsc = malloc(sizeof(*cg->petsc));
    if (!cg->petsc) return ACG_ERR_ERRNO;
    err = acgpetsc_init(cg->petsc, A, petscdevicetype, petscksptype, comm->mpicomm);
    if (err) { free(cg->petsc); return err; }
    cg->ksptype = acgpetscksptype;
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
    return ACG_SUCCESS;
#endif
}

/*
 * iterative solution procedure
 */

/**
 * ‘acgsolverpetsc_solve()’ solves the given linear system, Ax=b, using the
 * conjugate gradient method.
 *
 * The solver must already have been configured with ‘acgsolverpetsc_init()’
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
int acgsolverpetsc_solve(
    struct acgsolverpetsc * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup)
{
#ifndef ACG_HAVE_PETSC
    return ACG_ERR_PETSC_NOT_SUPPORTED;
#else
    int err;
    if (b->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (x->size < A->nrows) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (diffatol > 0 || diffrtol > 0) return ACG_ERR_NOT_SUPPORTED;

    /* copy right-hand side values to PETSc vector */
    struct acgpetsc * petsc = cg->petsc;
    {
        double * bval;
        PetscInt bsize;
        err = VecGetArray(petsc->b, &bval); CHKERRQ(err);
        err = VecGetLocalSize(petsc->b, &bsize); CHKERRQ(err);
        for (PetscInt i = 0; i < bsize; i++) bval[i] = b->x[i];
        err = VecRestoreArray(petsc->b, &bval); CHKERRQ(err);
    }

    /* 10 warmup iterations */
    if (warmup > 0) {
        err = KSPSetTolerances(petsc->ksp, 0.0, 0.0, 1e10, warmup); CHKERRQ(err);
        err = KSPSetMinimumIterations(petsc->ksp, warmup); CHKERRQ(err);
        err = KSPSetErrorIfNotConverged(petsc->ksp, PETSC_FALSE); CHKERRQ(err);
        err = KSPSetInitialGuessNonzero(petsc->ksp, PETSC_FALSE);
        err = KSPSolve(petsc->ksp, petsc->b, petsc->x); CHKERRQ(err);
        err = KSPSetErrorIfNotConverged(petsc->ksp, PETSC_TRUE); CHKERRQ(err);
        err = KSPSetMinimumIterations(petsc->ksp, 0); CHKERRQ(err);
    }

    /* copy initial guess to PETSc vector */
    err = KSPSetInitialGuessNonzero(petsc->ksp, PETSC_TRUE);
    {
        double * xval;
        PetscInt xsize;
        err = VecGetArray(petsc->x, &xval); CHKERRQ(err);
        err = VecGetLocalSize(petsc->x, &xsize); CHKERRQ(err);
        for (PetscInt i = 0; i < xsize; i++) xval[i] = x->x[i];
        err = VecRestoreArray(petsc->x, &xval); CHKERRQ(err);
    }

    /* set stopping criteria */
    err = KSPSetTolerances(
        petsc->ksp, residualrtol, residualatol, 1e10, maxits); CHKERRQ(err);
    if (residualatol == 0 && residualrtol == 0) {
        err = KSPSetMinimumIterations(petsc->ksp, maxits); CHKERRQ(err);
        err = KSPSetErrorIfNotConverged(petsc->ksp, PETSC_FALSE); CHKERRQ(err);
    }

    /* set initial state */
    acgtime_t t0, t1;
    cg->nsolves++; cg->niterations = 0;
    cg->bnrm2 = INFINITY;
    cg->r0nrm2 = cg->rnrm2 = INFINITY;
    cg->x0nrm2 = cg->dxnrm2 = INFINITY;
    cg->maxits = maxits;
    cg->diffatol = diffatol;
    cg->diffrtol = diffrtol;
    cg->residualatol = residualatol;
    cg->residualrtol = residualrtol;

    /* run the solver */
    err = KSPView(petsc->ksp, PETSC_VIEWER_DEFAULT); CHKERRQ(err);
    gettime(&t0);
    err = KSPSolve(petsc->ksp, petsc->b, petsc->x); CHKERRQ(err);
    gettime(&t1); cg->tsolve += elapsed(t0,t1);

    /* copy solution back from PETSc vector */
    {
        double * xval;
        PetscInt xsize;
        err = VecGetArray(petsc->x, &xval); CHKERRQ(err);
        err = VecGetLocalSize(petsc->x, &xsize); CHKERRQ(err);
        for (PetscInt i = 0; i < xsize; i++) x->x[i] = xval[i];
        err = VecRestoreArray(petsc->x, &xval); CHKERRQ(err);
    }

    PetscInt its;
    err = KSPGetIterationNumber(petsc->ksp, &its); CHKERRQ(err);
    cg->ntotaliterations += its; cg->niterations += its;
    err = KSPGetResidualNorm(petsc->ksp, &cg->rnrm2); CHKERRQ(err);
    KSPConvergedReason reason;
    err = KSPGetConvergedReason(petsc->ksp, &reason); CHKERRQ(err);
    if (residualatol == 0 && residualrtol == 0 && its == maxits) return ACG_SUCCESS;
    if (reason < 0) return ACG_ERR_NOT_CONVERGED;
    return ACG_SUCCESS;
#endif
}

/*
 * iterative solution procedure in distributed memory using MPI
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolverpetsc_solvempi()’ solves the given linear system, Ax=b, using
 * the conjugate gradient method. The linear system may be distributed
 * across multiple processes and communication is handled using MPI.
 *
 * The solver must already have been configured with ‘acgsolverpetsc_init()’
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
int acgsolverpetsc_solvempi(
    struct acgsolverpetsc * cg,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * b,
    struct acgvector * x,
    int maxits,
    double diffatol,
    double diffrtol,
    double residualatol,
    double residualrtol,
    int warmup,
    MPI_Comm comm,
    int tag,
    int * mpierrcode)
{
    return acgsolverpetsc_solve(
        cg, A, b, x, maxits, diffatol, diffrtol, residualatol, residualrtol, warmup);
}
#endif

/*
 * output solver info
 */

static void findent(FILE * f, int indent) { fprintf(f, "%*c", indent, ' '); }

/**
 * ‘acgsolverpetsc_fwrite()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolverpetsc_solve()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolverpetsc_fwrite(
    FILE * f,
    const struct acgsolverpetsc * cg,
    int indent)
{
#ifndef ACG_HAVE_PETSC
    return ACG_ERR_PETSC_NOT_SUPPORTED;
#else
    const struct acgpetsc * petsc = cg->petsc;
    PetscInt size;
    int err = VecGetSize(petsc->x, &size); CHKERRQ(err);
    findent(f,indent); fprintf(f, "unknowns: %'"PRIdx"\n", size);
    findent(f,indent); fprintf(f, "solves: %'d\n", cg->nsolves);
    findent(f,indent); fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
    findent(f,indent); fprintf(f, "total flops: N/A\n");
    findent(f,indent); fprintf(f, "total flop rate: N/A\n");
    findent(f,indent); fprintf(f, "total solver time: %'.6f seconds\n", cg->tsolve);
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
#endif
}

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsolverpetsc_fwritempi()’ outputs the status of a solver.
 *
 * This is normally used after calling ‘acgsolverpetsc_solvempi()’ to print a
 * message to report the status of the solver together with various
 * useful statistics.
 */
int acgsolverpetsc_fwritempi(
    FILE * f,
    const struct acgsolverpetsc * cg,
    int indent,
    int verbose,
    MPI_Comm comm,
    int root)
{
#ifndef ACG_HAVE_PETSC
    return ACG_ERR_PETSC_NOT_SUPPORTED;
#else
    int commsize, rank;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);
    const struct acgpetsc * petsc = cg->petsc;
    double tsolve = cg->tsolve;
    MPI_Reduce(&cg->tsolve, &tsolve, 1, MPI_DOUBLE, MPI_MAX, root, comm);
    if (rank == root) {
        PetscInt size;
        int err = VecGetSize(petsc->x, &size); CHKERRQ(err);
        findent(f,indent); fprintf(f, "unknowns: %'"PRIdx"\n", size);
        findent(f,indent); fprintf(f, "solves: %'d\n", cg->nsolves);
        findent(f,indent); fprintf(f, "total iterations: %'d\n", cg->ntotaliterations);
        findent(f,indent); fprintf(f, "total flops: N/A\n");
        findent(f,indent); fprintf(f, "total flop rate: N/A\n");
        findent(f,indent); fprintf(f, "total solver time: %'.6f seconds\n", tsolve);
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
#endif
}
#endif
