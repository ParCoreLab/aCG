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
 * sorting
 */

#ifndef ACG_SORT_H
#define ACG_SORT_H

#include "acg/config.h"

#include <stdint.h>

#ifndef ACG_IDX_SIZE
#define acgradixsort_idx_t acgradixsort_int
#define acgradixsortpair_idx_t acgradixsortpair_int
#elif ACG_IDX_SIZE == 32
#define acgradixsort_idx_t acgradixsort_int32_t
#define acgradixsortpair_idx_t acgradixsortpair_int32_t
#elif ACG_IDX_SIZE == 64
#define acgradixsort_idx_t acgradixsort_int64_t
#define acgradixsortpair_idx_t acgradixsortpair_int64_t
#else
#error "invalid ACG_IDX_SIZE; expected 32 or 64"
#endif

/*
 * radix sort for unsigned integers
 */

/**
 * ‘acgradixsort_uint32_t()’ sorts an array of 32-bit unsigned
 * integers in ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array.
 */
int acgradixsort_uint32_t(
    int64_t size,
    int stride,
    uint32_t * keys,
    int64_t * perm,
    int64_t * invperm);

/**
 * ‘acgradixsort_uint64_t()’ sorts an array of 64-bit unsigned
 * integers in ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array.
 */
int acgradixsort_uint64_t(
    int64_t size,
    int stride,
    uint64_t * keys,
    int64_t * perm,
    int64_t * invperm);

/*
 * radix sort for signed integers
 */

/**
 * ‘acgradixsort_int32_t()’ sorts an array of 32-bit (signed)
 * integers in ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array.
 */
int acgradixsort_int32_t(
    int64_t size,
    int stride,
    int32_t * keys,
    int64_t * perm,
    int64_t * invperm);

/**
 * ‘acgradixsort_int64_t()’ sorts an array of 64-bit (signed)
 * integers in ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array.
 */
int acgradixsort_int64_t(
    int64_t size,
    int stride,
    int64_t * keys,
    int64_t * perm,
    int64_t * invperm);

/**
 * ‘acgradixsort_int()’ sorts an array of (signed) integers in
 * ascending order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the array ‘keys’. On success, the same
 * array will contain the keys in sorted order.
 *
 * The value ‘stride’ is used to specify a stride (in bytes), which is
 * used when accessing elements of the ‘keys’ array. This is useful
 * for cases where the keys are not necessarily stored contiguously in
 * memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array.
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array.
 */
int acgradixsort_int(
    int64_t size,
    int stride,
    int * keys,
    int64_t * perm,
    int64_t * invperm);

/*
 * radix sort for unsigned integer pairs
 */

/**
 * ‘acgradixsortpair_uint32_t()’ sorts pairs of 32-bit unsigned
 * integers in ascending, lexicographic order using a radix sort
 * algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘keys1’ and ‘keys2’, where the
 * former contains the first element of each tuple and the latter
 * contains the second element of each tuple. On success, the same
 * arrays will contain the keys in sorted order.
 *
 * The values ‘stride1’ and ‘stride2’ are used to specify a stride (in
 * bytes), which is used when accessing elements of the ‘keys1’ and
 * ‘keys2’ arrays, respectively. This is useful for cases where the
 * keys are stored in a strided form in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array. In other words, the value of ‘keys1[i]’ before
 * sorting is equal to ‘keys1[perm[i]]’ after sorting (and similarly
 * for ‘keys2’).
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array. In other words, the value of ‘keys1[i]’
 * after sorting is equal to ‘keys1[invperm[i]]’ before sorting (and
 * similarly for ‘keys2’).
 */
int acgradixsortpair_uint32_t(
    int64_t size,
    int stride1,
    uint32_t * keys1,
    int stride2,
    uint32_t * keys2,
    int64_t * perm,
    int64_t * invperm);

/**
 * ‘acgradixsortpair_uint64_t()’ sorts pairs of 64-bit unsigned
 * integers in ascending, lexicographic order using a radix sort
 * algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘keys1’ and ‘keys2’, where the
 * former contains the first element of each tuple and the latter
 * contains the second element of each tuple. On success, the same
 * arrays will contain the keys in sorted order.
 *
 * The values ‘stride1’ and ‘stride2’ are used to specify a stride (in
 * bytes), which is used when accessing elements of the ‘keys1’ and
 * ‘keys2’ arrays, respectively. This is useful for cases where the
 * keys are stored in a strided form in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array. In other words, the value of ‘keys1[i]’ before
 * sorting is equal to ‘keys1[perm[i]]’ after sorting (and similarly
 * for ‘keys2’).
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array. In other words, the value of ‘keys1[i]’
 * after sorting is equal to ‘keys1[invperm[i]]’ before sorting (and
 * similarly for ‘keys2’).
 */
int acgradixsortpair_uint64_t(
    int64_t size,
    int stride1,
    uint64_t * keys1,
    int stride2,
    uint64_t * keys2,
    int64_t * perm,
    int64_t * invperm);

/*
 * radix sort for signed integer pairs
 */

/**
 * ‘acgradixsortpair_int32_t()’ sorts pairs of 32-bit signed integers
 * in ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘keys1’ and ‘keys2’, where the
 * former contains the first element of each tuple and the latter
 * contains the second element of each tuple. On success, the same
 * arrays will contain the keys in sorted order.
 *
 * The values ‘stride1’ and ‘stride2’ are used to specify a stride (in
 * bytes), which is used when accessing elements of the ‘keys1’ and
 * ‘keys2’ arrays, respectively. This is useful for cases where the
 * keys are stored in a strided form in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array. In other words, the value of ‘keys1[i]’ before
 * sorting is equal to ‘keys1[perm[i]]’ after sorting (and similarly
 * for ‘keys2’).
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array. In other words, the value of ‘keys1[i]’
 * after sorting is equal to ‘keys1[invperm[i]]’ before sorting (and
 * similarly for ‘keys2’).
 */
int acgradixsortpair_int32_t(
    int64_t size,
    int stride1,
    int32_t * keys1,
    int stride2,
    int32_t * keys2,
    int64_t * perm,
    int64_t * invperm);

/**
 * ‘acgradixsortpair_int64_t()’ sorts pairs of 64-bit signed
 * integers in ascending, lexicographic order using a radix sort
 * algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘keys1’ and ‘keys2’, where the
 * former contains the first element of each tuple and the latter
 * contains the second element of each tuple. On success, the same
 * arrays will contain the keys in sorted order.
 *
 * The values ‘stride1’ and ‘stride2’ are used to specify a stride (in
 * bytes), which is used when accessing elements of the ‘keys1’ and
 * ‘keys2’ arrays, respectively. This is useful for cases where the
 * keys are stored in a strided form in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array. In other words, the value of ‘keys1[i]’ before
 * sorting is equal to ‘keys1[perm[i]]’ after sorting (and similarly
 * for ‘keys2’).
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array. In other words, the value of ‘keys1[i]’
 * after sorting is equal to ‘keys1[invperm[i]]’ before sorting (and
 * similarly for ‘keys2’).
 */
int acgradixsortpair_int64_t(
    int64_t size,
    int stride1,
    int64_t * keys1,
    int stride2,
    int64_t * keys2,
    int64_t * perm,
    int64_t * invperm);

/**
 * ‘acgradixsortpair_int()’ sorts pairs of signed integers in
 * ascending, lexicographic order using a radix sort algorithm.
 *
 * The number of keys to sort is given by ‘size’, and the unsorted,
 * integer keys are given in the arrays ‘keys1’ and ‘keys2’, where the
 * former contains the first element of each tuple and the latter
 * contains the second element of each tuple. On success, the same
 * arrays will contain the keys in sorted order.
 *
 * The values ‘stride1’ and ‘stride2’ are used to specify a stride (in
 * bytes), which is used when accessing elements of the ‘keys1’ and
 * ‘keys2’ arrays, respectively. This is useful for cases where the
 * keys are stored in a strided form in memory.
 *
 * If ‘perm’ is ‘NULL’, then this argument is ignored and a sorting
 * permutation is not computed. Otherwise, it must point to an array
 * that holds enough storage for ‘size’ values of type ‘int64_t’. On
 * success, this array will contain the sorting permutation, mapping
 * the locations of the original, unsorted keys to their new locations
 * in the sorted array. In other words, the value of ‘keys1[i]’ before
 * sorting is equal to ‘keys1[perm[i]]’ after sorting (and similarly
 * for ‘keys2’).
 *
 * Similarly, if ‘invperm’ is not ‘NULL’, it must point to an array
 * ‘size’ values of type ‘int64_t’. On success, this array will
 * contain the inverse sorting permutation, mapping the locations of
 * the keys in the sorted array to the corresponding keys in the
 * original, unsorted array. In other words, the value of ‘keys1[i]’
 * after sorting is equal to ‘keys1[invperm[i]]’ before sorting (and
 * similarly for ‘keys2’).
 */
int acgradixsortpair_int(
    int64_t size,
    int stride1,
    int * keys1,
    int stride2,
    int * keys2,
    int64_t * perm,
    int64_t * invperm);

#endif
