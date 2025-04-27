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
 * prefix sums
 */

#include <acg/error.h>
#include <acg/prefixsum.h>

#ifdef ACG_HAVE_OPENMP
#include <omp.h>
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

/**
 * ‘acgprefixsum_inplace_int32_t()’ computes an in-place prefix sum
 * of an array of 32-bit signed integers.
 *
 * The array ‘x’ must be of length equal to ‘size’, and it is used
 * both to provide the input and to contain the output.
 *
 * If ‘inclusive’ is true, then the prefix sum is inclusive,
 *
 *   yᵢ = Σⱼ xⱼ for i = 0,1,2,... and j = 0,1,...,i.
 *
 * Otherwise, the prefix sum is exclusive, i.e., y₀ = 0, and
 *
 *   yᵢ = Σⱼ xⱼ for i = 1,2,... and j = 0,1,...,i-1.
 *
 * If OpenMP is enabled, the prefix sum is computed in parallel.
 */
int acgprefixsum_inplace_int32_t(
    size_t size,
    int32_t * x,
    bool inclusive)
{
    if (size < 2) return ACG_SUCCESS;
#ifndef ACG_HAVE_OPENMP
    for (size_t i = 1; i < size; i++) x[i] += x[i-1];
    if (!inclusive) {
        for (size_t i = size-1; i > 0; i--) x[i] = x[i-1];
        x[0] = 0;
    }
#else
    #pragma omp parallel
    {
        int P = omp_get_num_threads(), p = omp_get_thread_num();
        size_t n = size/P, r = size%P;
        size_t a = p<r ? n*p+p : (p<size ? n*p+r : size);
        size_t b = (p+1)<r ? n*(p+1)+(p+1) : ((p+1)<size ? n*(p+1)+r : size);
        for (size_t i = a+1; i < b; i++) x[i] += x[i-1];
        #pragma omp barrier
        size_t offset = 0;
        for (int q = 0; q < p; q++) {
            size_t bq = (q+1)<r ? n*(q+1)+(q+1) : ((q+1)<size ? n*(q+1)+r : size);
            offset += x[bq-1];
        }
        #pragma omp barrier
        for (size_t i = a; i < b; i++) x[i] += offset;
        if (!inclusive) {
            #pragma omp barrier
            int32_t xa = a > 0 && a < size ? x[a-1] : 0;
            #pragma omp barrier
            for (size_t i = b-1; i > a; i--) x[i] = x[i-1];
            #pragma omp barrier
            if (a < size) x[a] = xa;
        }
    }
#endif
    return ACG_SUCCESS;
}

/**
 * ‘acgprefixsum_inplace_int64_t()’ computes an in-place prefix sum
 * of an array of 64-bit signed integers.
 *
 * The array ‘x’ must be of length equal to ‘size’, and it is used
 * both to provide the input and to contain the output.
 *
 * If ‘inclusive’ is true, then the prefix sum is inclusive,
 *
 *   yᵢ = Σⱼ xⱼ for i = 0,1,2,... and j = 0,1,...,i.
 *
 * Otherwise, the prefix sum is exclusive, i.e., y₀ = 0, and
 *
 *   yᵢ = Σⱼ xⱼ for i = 1,2,... and j = 0,1,...,i-1.
 *
 * If OpenMP is enabled, the prefix sum is computed in parallel.
 */
int acgprefixsum_inplace_int64_t(
    size_t size,
    int64_t * x,
    bool inclusive)
{
    if (size < 2) return ACG_SUCCESS;
#ifndef ACG_HAVE_OPENMP
    for (size_t i = 1; i < size; i++) x[i] += x[i-1];
    if (!inclusive) {
        for (size_t i = size-1; i > 0; i--) x[i] = x[i-1];
        x[0] = 0;
    }
#else
    #pragma omp parallel
    {
        int P = omp_get_num_threads(), p = omp_get_thread_num();
        size_t n = size/P, r = size%P;
        size_t a = p<r ? n*p+p : (p<size ? n*p+r : size);
        size_t b = (p+1)<r ? n*(p+1)+(p+1) : ((p+1)<size ? n*(p+1)+r : size);
        for (size_t i = a+1; i < b; i++) x[i] += x[i-1];
        #pragma omp barrier
        size_t offset = 0;
        for (int q = 0; q < p; q++) {
            size_t bq = (q+1)<r ? n*(q+1)+(q+1) : ((q+1)<size ? n*(q+1)+r : size);
            offset += x[bq-1];
        }
        #pragma omp barrier
        for (size_t i = a; i < b; i++) x[i] += offset;
        if (!inclusive) {
            #pragma omp barrier
            int64_t xa = a > 0 && a < size ? x[a-1] : 0;
            #pragma omp barrier
            for (size_t i = b-1; i > a; i--) x[i] = x[i-1];
            #pragma omp barrier
            if (a < size) x[a] = xa;
        }
    }
#endif
    return ACG_SUCCESS;
}

/**
 * ‘acgprefixsum_inplace_int()’ computes an in-place prefix sum
 * of an array signed integers.
 *
 * The array ‘x’ must be of length equal to ‘size’, and it is used
 * both to provide the input and to contain the output.
 *
 * If ‘inclusive’ is true, then the prefix sum is inclusive,
 *
 *   yᵢ = Σⱼ xⱼ for i = 0,1,2,... and j = 0,1,...,i.
 *
 * Otherwise, the prefix sum is exclusive, i.e., y₀ = 0, and
 *
 *   yᵢ = Σⱼ xⱼ for i = 1,2,... and j = 0,1,...,i-1.
 *
 * If OpenMP is enabled, the prefix sum is computed in parallel.
 */
int acgprefixsum_inplace_int(
    size_t size,
    int * x,
    bool inclusive)
{
    if (sizeof(int) == sizeof(int32_t)) {
        return acgprefixsum_inplace_int32_t(size, (int32_t *) x, inclusive);
    } else if (sizeof(int) == sizeof(int64_t)) {
        return acgprefixsum_inplace_int64_t(size, (int64_t *) x, inclusive);
    } else { return ACG_ERR_NOT_SUPPORTED; }
}


/*
 * TODO: the example below is an incomplete version of an out-of-place
 * prefix sum
 */

/* /\** */
/*  * ‘acgprefixsum_int64_t()’ computes the exclusive prefix sum of an */
/*  * array of 64-bit signed integers, i.e., yᵢ = Σⱼ xⱼ for i = 1,2,..., */
/*  * where the sum ranges over j = 0,1,...,i-1. */
/*  * */
/*  * The input array, ‘x’, and output array, ‘y’, must contain ‘size’ */
/*  * elements, and the arrays may not overlap. */
/*  *\/ */
/* int acgprefixsum_int64_t( */
/*     size_t size, */
/*     const int64_t * __restrict x, */
/*     int64_t * __restrict y) */
/* { */
/* #ifndef ACG_HAVE_OPENMP */
/*     y[0] = 0; */
/*     for (size_t i = 1; i < size; i++) */
/*         y[i] = y[i-1] + x[i-1]; */
/* #else */
/*     #pragma omp parallel */
/*     { */
/*         int P = omp_get_num_threads(); */
/*         int p = omp_get_thread_num(); */
/*         size_t n = (size+P-1)/P; */
/*         size_t a = p*n < size ? p*n : size; */
/*         size_t b = (p+1)*n < size ? (p+1)*n : size; */
/*         if (a == 0 && a < size) y[a] = 0; */
/*         else if (a > 0 && a < size) y[a] = x[a-1]; */
/*         for (size_t i = a+1; i < b; i++) y[i] = y[i-1] + x[i-1]; */
/*         #pragma omp barrier */
/*         size_t offset = 0; */
/*         for (int q = 0; q < p; q++) offset += y[(q+1)*n-1]; */
/*         #pragma omp barrier */
/*         for (size_t i = a; i < b; i++) y[i] += offset; */
/*     } */
/* #endif */
/*     return ACG_SUCCESS; */
/* } */
