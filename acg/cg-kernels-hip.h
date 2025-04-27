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

#ifndef ACG_CG_KERNELS_HIP_H
#define ACG_CG_KERNELS_HIP_H

#include "acg/config.h"

#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int acgsolverhip_init_constants(
    double ** d_minus_one,
    double ** d_one,
    double ** d_zero);

int acgsolverhip_alpha(
    double * alpha,
    double * minus_alpha,
    const double * rnrm2sqr,
    const double * pdott);

int acgsolverhip_beta(
    double * beta,
    const double * rnrm2sqr,
    const double * rnrm2sqr_prev);

int acgsolverhip_daxpy_alpha(
    int n,
    const double * d_rnrm2sqr,
    const double * d_pdott,
    const double * d_x,
    double * d_y);

int acgsolverhip_daxpy_minus_alpha(
    int n,
    const double * d_rnrm2sqr,
    const double * d_pdott,
    const double * d_x,
    double * d_y);

#ifdef ACG_HAVE_HIP
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
    hipStream_t stream);
#endif

int acgsolverhip_daypx_beta(
    int n,
    const double * d_rnrm2sqr,
    const double * d_rnrm2sqr_prev,
    double * d_y,
    const double * d_x);

int acgsolverhip_csrgemv_merge_startrows(
    acgidx_t n,
    const acgidx_t * __restrict d_rowptr,
    acgidx_t nstartrows,
    acgidx_t * d_startrows);

int acgsolverhip_csrgemv_merge(
    acgidx_t n,
    double * __restrict d_y,
    const double * __restrict d_x,
    const acgidx_t * __restrict d_rowptr,
    const acgidx_t * __restrict d_colidx,
    const double * __restrict d_a,
    double alpha,
    acgidx_t nstartrows,
    const acgidx_t * __restrict d_startrows);

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
ACG_API int acgsolverhip_solve_device(
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
    int * errcode);

#ifdef __cplusplus
}
#endif

#endif
