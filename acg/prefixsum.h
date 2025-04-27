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

#ifndef ACG_PREFIXSUM_H
#define ACG_PREFIXSUM_H

#include "acg/config.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef ACG_IDX_SIZE
#define acgprefixsum_inplace_idx_t acgprefixsum_inplace_int
#elif ACG_IDX_SIZE == 32
#define acgprefixsum_inplace_idx_t acgprefixsum_inplace_int32_t
#elif ACG_IDX_SIZE == 64
#define acgprefixsum_inplace_idx_t acgprefixsum_inplace_int64_t
#else
#error "invalid ACG_IDX_SIZE; expected 32 or 64"
#endif

/*
 * prefix sums
 */

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
    bool inclusive);

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
    bool inclusive);

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
    bool inclusive);

#endif
