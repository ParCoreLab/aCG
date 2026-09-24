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

#ifndef ACG_CG_KERNELS_CUDA_H
#define ACG_CG_KERNELS_CUDA_H

#include "acg/config.h"

#ifdef ACG_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    int getNumberOfSMs(int *numOfSMs);

    int acgsolvercuda_init_constants(
        double **d_minus_one,
        double **d_one,
        double **d_zero);

    int acgsolvercuda_alpha(
        double *alpha,
        double *minus_alpha,
        const double *rnrm2sqr,
        const double *pdott);

    int acgsolvercuda_beta(
        double *beta,
        const double *rnrm2sqr,
        const double *rnrm2sqr_prev);

    int acgsolvercuda_daxpy_alpha(
        int n,
        const double *d_rnrm2sqr,
        const double *d_pdott,
        const double *d_x,
        double *d_y);

    int acgsolvercuda_daxpy_minus_alpha(
        int n,
        const double *d_rnrm2sqr,
        const double *d_pdott,
        const double *d_x,
        double *d_y);

#ifdef ACG_HAVE_CUDA

    int acgsolvercuda_compute_alpha_beta(
        double *d_alpha,
        double *d_minus_alpha,
        double *d_beta,
        double *d_gamma,
        double *d_delta,
        double *d_gamma_prev,
        int k,
        cudaStream_t stream);

    int acgsolvercuda_apply_jacobi_preconditioner(
        int n,
        const double *M_inv,
        const double *r,
        double *d_z,
        int numSMs,
        cudaStream_t stream);

    int acgsolvercuda_jacobi_preconditioner(
        int n,
        double *d_M_inv,
        const int *d_rowptr,
        const int *d_colidx,
        const double *d_a,
        cudaStream_t stream);

    int acgsolvercuda_pipelined_daxpy_fused(
        int n,
        const double *d_gamma,
        double *d_gamma_prev,
        const double *d_delta,
        const double *d_q,
        double *d_p,
        double *d_r,
        double *d_t,
        double *d_x,
        double *d_z,
        double *d_w,
        double *d_alpha_prev,
        cudaStream_t stream);

    int acgsolvercuda_preconditioned_pipelined_daxpy_fused(
        int num,
        int k,
        const double *d_gamma,
        double *d_gamma_prev,
        const double *d_delta,
        double *d_p,
        double *d_r,
        double *d_t,
        double *d_x,
        double *d_z,
        double *d_w,
        double *d_q,
        double *d_n,
        double *d_m,
        double *d_u,
        double *d_alpha_prev,
        cudaStream_t stream);

    int acgsolvercuda_preconditioned_daxpy_fused(
        int num,
        const double *d_gamma,
        double *d_gamma_prev,
        const double *d_delta,
        double *d_p,
        double *d_r,
        double *d_t,
        double *d_x,
        cudaStream_t stream);

    /*
     * Preconditioned BiCGStab vector updates (see cg-kernels-cuda.cu).
     */
    int acgsolvercuda_bicgstab_p_update(
        int n,
        int k,
        const double *d_rho,
        const double *d_rho_prev,
        const double *d_alpha,
        const double *d_omega,
        double *d_p,
        const double *d_r,
        const double *d_v,
        cudaStream_t stream);

    int acgsolvercuda_bicgstab_s_update(
        int n,
        double *d_alpha,
        const double *d_rho,
        const double *d_rhat_v,
        double *d_s,
        const double *d_r,
        const double *d_v,
        cudaStream_t stream);

    int acgsolvercuda_bicgstab_xr_update(
        int n,
        double *d_omega,
        double *d_rho_prev,
        const double *d_num,
        const double *d_den,
        const double *d_rho,
        const double *d_alpha,
        double *d_x,
        const double *d_y,
        const double *d_z,
        double *d_r,
        const double *d_s,
        const double *d_t,
        cudaStream_t stream);

    int acgsolvercuda_bicgstab_x_halfstep(
        int n,
        const double *d_alpha,
        double *d_x,
        const double *d_y,
        cudaStream_t stream);

    /*
     * Pipelined (communication-hiding) BiCGStab vector updates
     * (see cg-kernels-cuda.cu).
     */
    int acgsolvercuda_pipelined_bicgstab_psz_update(
        int n,
        const double *d_beta,
        const double *d_omega,
        const double *d_r,
        const double *d_w,
        const double *d_t,
        double *d_p,
        double *d_s,
        double *d_z,
        const double *d_v,
        cudaStream_t stream);

    int acgsolvercuda_pipelined_bicgstab_qy_update(
        int n,
        const double *d_alpha,
        const double *d_r,
        const double *d_w,
        const double *d_s,
        const double *d_z,
        double *d_q,
        double *d_y,
        cudaStream_t stream);

    int acgsolvercuda_pipelined_bicgstab_xrw_update(
        int n,
        double *d_omega,
        const double *d_g1,
        const double *d_g2,
        const double *d_alpha,
        const double *d_p,
        const double *d_q,
        const double *d_y,
        const double *d_t,
        const double *d_v,
        double *d_x,
        double *d_r,
        double *d_w,
        cudaStream_t stream);

    int acgsolvercuda_pipelined_bicgstab_scalars(
        double *d_beta,
        double *d_alpha,
        double *d_rho,
        const double *d_omega,
        const double *d_d1,
        const double *d_d2,
        const double *d_d3,
        const double *d_d4,
        cudaStream_t stream);
#endif

    int acgsolvercuda_daypx_beta(
        int n,
        const double *d_rnrm2sqr,
        const double *d_rnrm2sqr_prev,
        double *d_y,
        const double *d_x);

#ifdef ACG_HAVE_CUDA
    /**
     * 'acgsolvercuda_dzero()' zeros out a device vector.
     * y[i] = 0 for i = 0, ..., n-1
     */
    int acgsolvercuda_dzero(
        acgidx_t n,
        double *d_x,
        cudaStream_t stream);

    /**
     * 'acgsolvercuda_dcopy()' copies a device vector.
     * y[i] = x[i] for i = 0, ..., n-1
     */
    int acgsolvercuda_dcopy(
        acgidx_t n,
        double *d_y,
        const double *d_x,
        cudaStream_t stream);

    /**
     * 'acgsolvercuda_ddot()' computes the dot product of two device vectors.
     * result = sum(x[i] * y[i]) for i = 0, ..., n-1
     * Note: result is zeroed internally before accumulation.
     */
    int acgsolvercuda_ddot(
        acgidx_t n,
        const double *d_x,
        const double *d_y,
        double *d_result,
        cudaStream_t stream);

    /**
     * 'acgsolvercuda_csrgemv()' computes sparse matrix-vector product.
     * y = beta * y + alpha * A * x
     * where A is in CSR format.
     */
    int acgsolvercuda_csrgemv(
        acgidx_t n,
        double *d_y,
        const double *d_x,
        const acgidx_t *d_rowptr,
        const acgidx_t *d_colidx,
        const double *d_a,
        double alpha,
        double beta,
        cudaStream_t stream);

    /**
     * 'acgsolvercuda_csrgemv_merge_nstartrows()' computes the number of
     * startrows entries needed for merge-based SpMV.
     */
    acgidx_t acgsolvercuda_csrgemv_merge_nstartrows(acgidx_t n, acgidx_t nnz);

    /**
     * 'acgsolvercuda_csrgemv_merge_init()' precomputes the starting rows
     * for merge-based SpMV. Must be called once before using
     * acgsolvercuda_csrgemv_merge.
     */
    int acgsolvercuda_csrgemv_merge_init(
        acgidx_t n,
        const acgidx_t *d_rowptr,
        acgidx_t nstartrows,
        acgidx_t *d_startrows,
        cudaStream_t stream);

    /**
     * 'acgsolvercuda_csrgemv_merge()' computes sparse matrix-vector product
     * using merge-based algorithm.
     * y = beta * y + alpha * A * x
     * where A is in CSR format.
     * Note: beta must be 0 or 1 (0 zeros output first, 1 accumulates).
     */
    int acgsolvercuda_csrgemv_merge(
        acgidx_t n,
        double *d_y,
        const double *d_x,
        const acgidx_t *d_rowptr,
        const acgidx_t *d_colidx,
        const double *d_a,
        double alpha,
        double beta,
        acgidx_t nstartrows,
        const acgidx_t *d_startrows,
        int numSMs,
        cudaStream_t stream);

    int csrgemv_host(
        acgidx_t n,
        double *d_y,
        const double *d_x,
        const acgidx_t *d_rowptr,
        const acgidx_t *d_colidx,
        const double *d_a,
        double alpha,
        int numSMs,
        cudaStream_t stream);
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
     *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ <
     * γₐ
     *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖
     * < γᵣ
     *  - ‘residualatol’, an absolute tolerance for the residual, ‖b-Ax‖ < εₐ
     *  - ‘residualrtol’, a relative tolerance for the residual, ‖b-Ax‖/‖b-Ax₀‖
     * < εᵣ
     *
     * The iterative solver converges if
     *
     *   ‖δx‖ < γₐ, ‖δx‖ < γᵣ‖x₀‖, ‖b-Ax‖ < εₐ or ‖b-Ax‖ < εᵣ‖b-Ax₀‖.
     *
     * To skip the convergence test for any one of the above stopping
     * criterion, the associated tolerance may be set to zero.
     */
    ACG_API int acgsolvercuda_solve_device(
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
        int *errcode);

    /**
     * ‘acgsolvercuda_solve_preconditioned_device()’ solves the given linear system,
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
     *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ <
     * γₐ
     *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖
     * < γᵣ
     *  - ‘residualatol’, an absolute tolerance for the residual, ‖b-Ax‖ < εₐ
     *  - ‘residualrtol’, a relative tolerance for the residual, ‖b-Ax‖/‖b-Ax₀‖
     * < εᵣ
     *
     * The iterative solver converges if
     *
     *   ‖δx‖ < γₐ, ‖δx‖ < γᵣ‖x₀‖, ‖b-Ax‖ < εₐ or ‖b-Ax‖ < εᵣ‖b-Ax₀‖.
     *
     * To skip the convergence test for any one of the above stopping
     * criterion, the associated tolerance may be set to zero.
     */
    ACG_API int acgsolvercuda_solve_preconditioned_device(
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
        int *errcode,
        int preconditioner);

    /**
     * ‘acgsolvercuda_solve_pipelined_preconditioned_device()’ solves the given linear system,
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
     *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ <
     * γₐ
     *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖
     * < γᵣ
     *  - ‘residualatol’, an absolute tolerance for the residual, ‖b-Ax‖ < εₐ
     *  - ‘residualrtol’, a relative tolerance for the residual, ‖b-Ax‖/‖b-Ax₀‖
     * < εᵣ
     *
     * The iterative solver converges if
     *
     *   ‖δx‖ < γₐ, ‖δx‖ < γᵣ‖x₀‖, ‖b-Ax‖ < εₐ or ‖b-Ax‖ < εᵣ‖b-Ax₀‖.
     *
     * To skip the convergence test for any one of the above stopping
     * criterion, the associated tolerance may be set to zero.
     */
    ACG_API int acgsolvercuda_solve_pipelined_preconditioned_device(
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
        int *errcode,
        int preconditioner);

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
     *  - ‘diffatol’, an absolute tolerance for the change in solution, ‖δx‖ <
     * γₐ
     *  - ‘diffrtol’, a relative tolerance for the change in solution, ‖δx‖/‖x₀‖
     * < γᵣ
     *  - ‘residualatol’, an absolute tolerance for the residual, ‖b-Ax‖ < εₐ
     *  - ‘residualrtol’, a relative tolerance for the residual, ‖b-Ax‖/‖b-Ax₀‖
     * < εᵣ
     *
     * The iterative solver converges if
     *
     *   ‖δx‖ < γₐ, ‖δx‖ < γᵣ‖x₀‖, ‖b-Ax‖ < εₐ or ‖b-Ax‖ < εᵣ‖b-Ax₀‖.
     *
     * To skip the convergence test for any one of the above stopping
     * criterion, the associated tolerance may be set to zero.
     */
    ACG_API int acgsolvercuda_solve_device_pipelined(
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
        int *errcode);

#ifdef __cplusplus
}
#endif

#endif
