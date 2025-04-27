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

#include "acg/config.h"
#include <acg/error.h>
#include <acg/sort.h>

#include <errno.h>

#ifdef ACG_HAVE_OPENMP
#include <omp.h>
#endif

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    int64_t * invperm)
{
    /*
     * Sort using 11 binary digits in each round, which yields 3
     * rounds for 32-bit integers and 6 rounds for 64-bit integers.
     * The choice of using 11 bits in each round is described in the
     * article "Radix Tricks" by Michael Herf, published online in
     * December 2001 at http://stereopsis.com/radix.html.
     *
     * Double-buffering is used to move the keys into the correct
     * position in every round. As a result, at the end of every
     * even-numbered round (counting from zero), keys will be stored
     * in the auxiliary buffer. If the sorting ends after completing
     * an even number of rounds, then a final copy is performed at the
     * end to ensure that the sorted keys end up in the original array
     * that was provided as input.
     *
     * During the first round, the maximum key is found, and it is
     * used to exit early, if possible.
     *
     * If OpenMP is available, sorting will be performed in parallel.
     */
    const int bitsperplace = 11;
    const int radix = 1 << bitsperplace;
    const int nbits = CHAR_BIT*sizeof(*keys);
    const int places = (nbits+bitsperplace-1) / bitsperplace;
    int nthreads = 1;
#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    /* allocate storage for double-buffering keys */
    uint32_t * tmpkeys = malloc(size*sizeof(*tmpkeys));
    if (!tmpkeys) return ACG_ERR_ERRNO;

    /*
     * Allocate storage for each thread to count occurrences of digits
     * using the given base/radix.
     *
     * The number of occurrences in the i-th digit for the j-th thread
     * will be stored in bucketptr[j*radix+i].
     *
     * NOTE: If we pad the array with one additional entry, then we
     * can avoid the second pass over bucketptr to transform the
     * inclusive scan to an exclusive scan.
     */
    int64_t * bucketptr = malloc(nthreads*radix*sizeof(*bucketptr));
    if (!bucketptr) { free(tmpkeys); return ACG_ERR_ERRNO; }

    /* allocate storage for permutation and inverse permutation */
    bool freeperm = false;
    bool freeinvperm = false;
    if (!perm && invperm) {
        perm = malloc(size*sizeof(*perm));
        if (!perm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeperm = true;
    } else if (perm && !invperm) {
        invperm = malloc(size*sizeof(*invperm));
        if (!invperm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeinvperm = true;
    }

    uint32_t maxkey = 0;
#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef ACG_HAVE_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        int nrounds = 0;
        for (int k = 0; k < places; k++) {

            /* reset per-thread counters for each digit, and then
             * count occurrences of each digit */
            for (int j = 0; j < radix; j++) bucketptr[tid*radix+j] = 0;
            if (k == 0) {
                #pragma omp for reduction(max:maxkey)
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x = *(const uint32_t *) ((const char *) keys+i*stride);
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                    maxkey = maxkey >= x ? maxkey : x;
                }
            } else if (k % 2 == 0) {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x = *(const uint32_t *) ((const char *) keys+i*stride);
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                }
            } else {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x = tmpkeys[i];
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                }
            }

            /* skip subsequent rounds, if the digits in the
             * corresponding places are all zero */
            if (maxkey < (1 << (bitsperplace*k))) break;

            /* compute offset to first key for each digit and thread */
            #pragma omp master
            {
                for (int j = 1; j < nthreads; j++)
                    bucketptr[j*radix+0] += bucketptr[(j-1)*radix+0];
                for (int i = 1; i < radix; i++) {
                    bucketptr[0*radix+i] += bucketptr[(nthreads-1)*radix+i-1];
                    for (int j = 1; j < nthreads; j++)
                        bucketptr[j*radix+i] += bucketptr[(j-1)*radix+i];
                }

                for (int i = radix-1; i > 0; i--) {
                    for (int j = nthreads-1; j > 0; j--)
                        bucketptr[j*radix+i] = bucketptr[(j-1)*radix+i];
                    bucketptr[0*radix+i] = bucketptr[(nthreads-1)*radix+i-1];
                }
                for (int j = nthreads-1; j > 0; j--)
                    bucketptr[j*radix+0] = bucketptr[(j-1)*radix+0];
                bucketptr[0*radix+0] = 0;
            }
            #pragma omp barrier

            /*
             * Permute keys to sort them by digits in the kth place.
             * Keys are moved to the auxiliary array (tmpkeys) in even
             * rounds and back to the original array in odd rounds.
             * If the sorting permutation is needed, then the inverse
             * permutation will be stored alternately in 'invperm' and
             * 'perm' in even- and odd-numbered rounds, respectively.
             */
            if (k == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x = *(const uint32_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x; invperm[dst] = i;
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x = *(const uint32_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x;
                    }
                }
            } else if (k % 2 == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x = *(const uint32_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x; invperm[dst] = perm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x = *(const uint32_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x;
                    }
                }
            } else if (k % 2 == 1) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x = tmpkeys[i];
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        *(uint32_t *) ((char *) keys+dst*stride) = x;
                        perm[dst] = invperm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x = tmpkeys[i];
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        *(uint32_t *) ((char *) keys+dst*stride) = x;
                    }
                }
            }
            nrounds++;
        }

        /* post-processing to ensure output is returned in-place */
        if (nrounds == 0) {
            /* if all keys are zero, return identity permutation */
            if (perm && !freeperm) { for (int64_t i = 0; i < size; i++) perm[i] = i; }
            if (invperm && !freeinvperm) { for (int64_t i = 0; i < size; i++) invperm[i] = i; }
        } else if (nrounds % 2 == 1) {
            /* If nrounds is odd, then we stopped after completing an even
             * number of rounds, and we must perform a final swap of
             * the input and temporary arrays. The inverse permutation
             * is currently stored in 'invperm', so we use it to
             * compute the permutation 'perm', if needed. */
            #pragma omp for
            for (int64_t i = 0; i < size; i++)
                *((uint32_t *) ((char *) keys+i*stride)) = tmpkeys[i];
            if (perm && invperm && !freeperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
            }
        } else if (nrounds % 2 == 0) {
            /* If nrounds is even, then we stopped after completing an
             * odd number of rounds, and the sorted keys are already
             * stored in the original array. The inverse permutation
             * is currently stored in 'perm', so we swap the 'perm'
             * and 'invperm', and then use 'invperm' to compute the
             * permutation 'perm', if needed. */
            if (perm && invperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) invperm[j] = perm[j];
                if (!freeperm) {
                    #pragma omp for
                    for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
                }
            }
        }
    }

    if (freeperm) free(perm);
    if (freeinvperm) free(invperm);
    free(bucketptr); free(tmpkeys);
    return ACG_SUCCESS;
}

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
    int64_t * invperm)
{
    /*
     * Sort using 11 binary digits in each round, which yields 3
     * rounds for 32-bit integers and 6 rounds for 64-bit integers.
     * The choice of using 11 bits in each round is described in the
     * article "Radix Tricks" by Michael Herf, published online in
     * December 2001 at http://stereopsis.com/radix.html.
     *
     * Double-buffering is used to move the keys into the correct
     * position in every round. As a result, at the end of every
     * even-numbered round (counting from zero), keys will be stored
     * in the auxiliary buffer. If the sorting ends after completing
     * an even number of rounds, then a final copy is performed at the
     * end to ensure that the sorted keys end up in the original array
     * that was provided as input.
     *
     * During the first round, the maximum key is found, and it is
     * used to exit early, if possible.
     *
     * If OpenMP is available, sorting will be performed in parallel.
     */
    const int bitsperplace = 11;
    const int radix = 1 << bitsperplace;
    const int nbits = CHAR_BIT*sizeof(*keys);
    const int places = (nbits+bitsperplace-1) / bitsperplace;
    int nthreads = 1;
#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    /* allocate storage for double-buffering keys */
    uint64_t * tmpkeys = malloc(size*sizeof(*tmpkeys));
    if (!tmpkeys) return ACG_ERR_ERRNO;

    /*
     * Allocate storage for each thread to count occurrences of digits
     * using the given base/radix.
     *
     * The number of occurrences in the i-th digit for the j-th thread
     * will be stored in bucketptr[j*radix+i].
     *
     * NOTE: If we pad the array with one additional entry, then we
     * can avoid the second pass over bucketptr to transform the
     * inclusive scan to an exclusive scan.
     */
    int64_t * bucketptr = malloc(nthreads*radix*sizeof(*bucketptr));
    if (!bucketptr) { free(tmpkeys); return ACG_ERR_ERRNO; }

    /* allocate storage for permutation and inverse permutation */
    bool freeperm = false;
    bool freeinvperm = false;
    if (!perm && invperm) {
        perm = malloc(size*sizeof(*perm));
        if (!perm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeperm = true;
    } else if (perm && !invperm) {
        invperm = malloc(size*sizeof(*invperm));
        if (!invperm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeinvperm = true;
    }

    uint64_t maxkey = 0;
#if defined(ACG_HAVE_OPENMP) && defined(__NVCOMPILER)
    uint64_t * maxkeys = malloc(nthreads*sizeof(*maxkeys));
    if (!maxkeys) return ACG_ERR_ERRNO;
    for (int i = 0; i < nthreads; i++) maxkeys[i] = 0;
#endif

#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef ACG_HAVE_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        int nrounds = 0;
        for (int k = 0; k < places; k++) {

            /* reset per-thread counters for each digit, and then
             * count occurrences of each digit */
            for (int j = 0; j < radix; j++) bucketptr[tid*radix+j] = 0;
            if (k == 0) {
#ifdef __NVCOMPILER
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                    maxkeys[tid] = maxkeys[tid] >= x ? maxkeys[tid] : x;
                }
                #pragma omp single
                for (int i = 0; i < nthreads; i++)
                    maxkey = maxkeys[i] = maxkeys[i] > maxkey ? maxkeys[i] : maxkey;
#else
                #pragma omp for reduction(max:maxkey)
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                    maxkey = maxkey >= x ? maxkey : x;
                }
#endif
            } else if (k % 2 == 0) {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                }
            } else {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x = tmpkeys[i];
                    bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                }
            }

            /* skip subsequent rounds, if the digits in the
             * corresponding places are all zero */
            if (maxkey < (1 << (bitsperplace*k))) break;

            /* compute offset to first key for each digit and thread */
            #pragma omp master
            {
                for (int j = 1; j < nthreads; j++)
                    bucketptr[j*radix+0] += bucketptr[(j-1)*radix+0];
                for (int i = 1; i < radix; i++) {
                    bucketptr[0*radix+i] += bucketptr[(nthreads-1)*radix+i-1];
                    for (int j = 1; j < nthreads; j++)
                        bucketptr[j*radix+i] += bucketptr[(j-1)*radix+i];
                }

                for (int i = radix-1; i > 0; i--) {
                    for (int j = nthreads-1; j > 0; j--)
                        bucketptr[j*radix+i] = bucketptr[(j-1)*radix+i];
                    bucketptr[0*radix+i] = bucketptr[(nthreads-1)*radix+i-1];
                }
                for (int j = nthreads-1; j > 0; j--)
                    bucketptr[j*radix+0] = bucketptr[(j-1)*radix+0];
                bucketptr[0*radix+0] = 0;
            }
            #pragma omp barrier

            /*
             * Permute keys to sort them by digits in the kth place.
             * Keys are moved to the auxiliary array (tmpkeys) in even
             * rounds and back to the original array in odd rounds.
             * If the sorting permutation is needed, then the inverse
             * permutation will be stored alternately in 'invperm' and
             * 'perm' in even- and odd-numbered rounds, respectively.
             */
            if (k == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x; invperm[dst] = i;
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x;
                    }
                }
            } else if (k % 2 == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x; invperm[dst] = perm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x = *(const uint64_t *) ((const char *) keys+i*stride);
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        tmpkeys[dst] = x;
                    }
                }
            } else if (k % 2 == 1) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x = tmpkeys[i];
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        *(uint64_t *) ((char *) keys+dst*stride) = x;
                        perm[dst] = invperm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x = tmpkeys[i];
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*k))&(radix-1))]++;
                        *(uint64_t *) ((char *) keys+dst*stride) = x;
                    }
                }
            }
            nrounds++;
        }

        /* post-processing to ensure output is returned in-place */
        if (nrounds == 0) {
            /* if all keys are zero, return identity permutation */
            if (perm && !freeperm) { for (int64_t i = 0; i < size; i++) perm[i] = i; }
            if (invperm && !freeinvperm) { for (int64_t i = 0; i < size; i++) invperm[i] = i; }
        } else if (nrounds % 2 == 1) {
            /* If nrounds is odd, then we stopped after completing an even
             * number of rounds, and we must perform a final swap of
             * the input and temporary arrays. The inverse permutation
             * is currently stored in 'invperm', so we use it to
             * compute the permutation 'perm', if needed. */
            #pragma omp for
            for (int64_t i = 0; i < size; i++)
                *((uint64_t *) ((char *) keys+i*stride)) = tmpkeys[i];
            if (perm && invperm && !freeperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
            }
        } else if (nrounds % 2 == 0) {
            /* If nrounds is even, then we stopped after completing an
             * odd number of rounds, and the sorted keys are already
             * stored in the original array. The inverse permutation
             * is currently stored in 'perm', so we swap the 'perm'
             * and 'invperm', and then use 'invperm' to compute the
             * permutation 'perm', if needed. */
            if (perm && invperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) invperm[j] = perm[j];
                if (!freeperm) {
                    #pragma omp for
                    for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
                }
            }
        }
    }

#if defined(ACG_HAVE_OPENMP) && defined(__NVCOMPILER)
    free(maxkeys);
#endif
    if (freeperm) free(perm);
    if (freeinvperm) free(invperm);
    free(bucketptr); free(tmpkeys);
    return ACG_SUCCESS;
}

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
    int64_t * invperm)
{
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++)
        *(int32_t *)((char *) keys+i*stride) ^= INT32_MIN;
    int err = acgradixsort_uint32_t(
        size, stride, (uint32_t *) keys, perm, invperm);
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++)
        *(int32_t *)((char *) keys+i*stride) ^= INT32_MIN;
    return err;
}

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
    int64_t * invperm)
{
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++)
        *(int64_t *)((char *) keys+i*stride) ^= INT64_MIN;
    int err = acgradixsort_uint64_t(
        size, stride, (uint64_t *) keys, perm, invperm);
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++)
        *(int64_t *)((char *) keys+i*stride) ^= INT64_MIN;
    return err;
}

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
    int64_t * invperm)
{
    if (sizeof(int) == sizeof(int32_t)) {
        return acgradixsort_int32_t(
            size, stride, (int32_t *) keys, perm, invperm);
    } else if (sizeof(int) == sizeof(int64_t)) {
        return acgradixsort_int64_t(
            size, stride, (int64_t *) keys, perm, invperm);
    } else { return ENOTSUP; }
}

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
    int64_t * invperm)
{
    /*
     * Sort using 11 binary digits in each round, which yields 6
     * rounds for sorting pairs of 32-bit integers.
     *
     * The choice of using 11 bits in each round is described in the
     * article "Radix Tricks" by Michael Herf, published online in
     * December 2001 at http://stereopsis.com/radix.html.
     *
     * Double-buffering is used to move the keys into the correct
     * position in every round. As a result, at the end of every
     * even-numbered round (counting from zero), keys will be stored
     * in the auxiliary buffer. If the sorting ends after completing
     * an even number of rounds, then a final copy is performed at the
     * end to ensure that the sorted keys end up in the original array
     * that was provided as input.
     *
     * During the first round, the maximum key is found, and it is
     * used to exit early, if possible.
     *
     * If OpenMP is available, sorting will be performed in parallel.
     */
    const int bitsperplace = 11;
    const int radix = 1 << bitsperplace;
    const int nbits = CHAR_BIT*(sizeof(*keys1)+sizeof(*keys2));
    const int places = (nbits+bitsperplace-1) / bitsperplace;
    const int placesperitem = places / 2;
    int nthreads = 1;
#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    /* allocate storage for double-buffering keys */
    uint32_t * tmpkeys = malloc(2*size*sizeof(*tmpkeys));
    if (!tmpkeys) return ACG_ERR_ERRNO;

    /*
     * Allocate storage for each thread to count occurrences of digits
     * using the given base/radix.
     *
     * The number of occurrences in the i-th digit for the j-th thread
     * will be stored in bucketptr[j*radix+i].
     *
     * NOTE: If we pad the array with one additional entry, then we
     * can avoid the second pass over bucketptr to transform the
     * inclusive scan to an exclusive scan.
     */
    int64_t * bucketptr = malloc(nthreads*radix*sizeof(*bucketptr));
    if (!bucketptr) { free(tmpkeys); return ACG_ERR_ERRNO; }

    /* allocate storage for permutation and inverse permutation */
    bool freeperm = false;
    bool freeinvperm = false;
    if (!perm && invperm) {
        perm = malloc(size*sizeof(*perm));
        if (!perm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeperm = true;
    } else if (perm && !invperm) {
        invperm = malloc(size*sizeof(*invperm));
        if (!invperm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeinvperm = true;
    }

    uint32_t maxkey[2] = {0,0};
#if defined(ACG_HAVE_OPENMP) && defined(__NVCOMPILER)
    uint32_t * maxkeys = malloc(2*nthreads*sizeof(*maxkeys));
    if (!maxkeys) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*nthreads; i++) maxkeys[i] = 0;
#endif

#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef ACG_HAVE_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        int nrounds = 0;
        for (int k = 0; k < places; k++) {

            /* reset per-thread counters for each digit, and then
             * count occurrences of each digit */
            for (int j = 0; j < radix; j++) bucketptr[tid*radix+j] = 0;
            if (k == 0) {
#ifdef __NVCOMPILER
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x1 = (*(const uint32_t *) ((const char *) keys1+i*stride1));
                    uint32_t x2 = (*(const uint32_t *) ((const char *) keys2+i*stride2));
                    bucketptr[tid*radix+((x2>>(bitsperplace*(k%3)))&(radix-1))]++;
                    if (maxkeys[2*tid+0] < x1 || (maxkeys[2*tid+0] == x1 && maxkeys[2*tid+1] < x2)) {
                        maxkeys[2*tid+0] = x1, maxkeys[2*tid+1] = x2;
                    }
                }
                #pragma omp single
                for (int i = 0; i < nthreads; i++) {
                    if (maxkey[0] < maxkeys[2*i+0] || (maxkey[0] == maxkeys[2*i+0] && maxkey[1] < maxkeys[2*i+1])) {
                        maxkey[0] = maxkeys[2*i+0], maxkey[1] = maxkeys[2*i+1];
                    }
                }
#else
                #pragma omp for reduction(max:maxkey)
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x1 = (*(const uint32_t *) ((const char *) keys1+i*stride1));
                    uint32_t x2 = (*(const uint32_t *) ((const char *) keys2+i*stride2));
                    bucketptr[tid*radix+((x2>>(bitsperplace*(k%3)))&(radix-1))]++;
                    if (maxkey[0] < x1 || (maxkey[0] == x1 && maxkey[1] < x2)) {
                        maxkey[0] = x1, maxkey[1] = x2;
                    }
                }
#endif
            } else if (k % 2 == 0) {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x = k >= 3 ? (*(const uint32_t *) ((const char *) keys1+i*stride1))
                        : (*(const uint32_t *) ((const char *) keys2+i*stride2));
                    bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                }
            } else if (k % 2 == 1) {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint32_t x = k >= 3 ? tmpkeys[2*i+0] : tmpkeys[2*i+1];
                    bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                }
            }

            /* skip subsequent rounds, if the digits in the
             * corresponding places are all zero */
            if ((k < 3 && maxkey[1] < (1 << bitsperplace*k) && maxkey[0] == 0) ||
                (k >= 3 && maxkey[0] < (1 << bitsperplace*(k-3))))
                break;

            /* compute offset to first key for each digit and thread */
            #pragma omp master
            {
                for (int j = 1; j < nthreads; j++)
                    bucketptr[j*radix+0] += bucketptr[(j-1)*radix+0];
                for (int i = 1; i < radix; i++) {
                    bucketptr[0*radix+i] += bucketptr[(nthreads-1)*radix+i-1];
                    for (int j = 1; j < nthreads; j++)
                        bucketptr[j*radix+i] += bucketptr[(j-1)*radix+i];
                }

                for (int i = radix-1; i > 0; i--) {
                    for (int j = nthreads-1; j > 0; j--)
                        bucketptr[j*radix+i] = bucketptr[(j-1)*radix+i];
                    bucketptr[0*radix+i] = bucketptr[(nthreads-1)*radix+i-1];
                }
                for (int j = nthreads-1; j > 0; j--)
                    bucketptr[j*radix+0] = bucketptr[(j-1)*radix+0];
                bucketptr[0*radix+0] = 0;
            }
            #pragma omp barrier

            /*
             * Permute keys to sort them by digits in the kth place.
             * Keys are moved to the auxiliary array (tmpkeys) in even
             * rounds and back to the original array in odd rounds.
             * If the sorting permutation is needed, then the inverse
             * permutation will be stored alternately in 'invperm' and
             * 'perm' in even- and odd-numbered rounds, respectively.
             */
            if (k == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x1 = (*(const uint32_t *) ((const char *) keys1+i*stride1));
                        uint32_t x2 = (*(const uint32_t *) ((const char *) keys2+i*stride2));
                        uint32_t x = k >= 3 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2; invperm[dst] = i;
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x1 = (*(const uint32_t *) ((const char *) keys1+i*stride1));
                        uint32_t x2 = (*(const uint32_t *) ((const char *) keys2+i*stride2));
                        uint32_t x = k >= 3 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2;
                    }
                }
            } else if (k % 2 == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x1 = (*(const uint32_t *) ((const char *) keys1+i*stride1));
                        uint32_t x2 = (*(const uint32_t *) ((const char *) keys2+i*stride2));
                        uint32_t x = k >= 3 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2; invperm[dst] = perm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x1 = (*(const uint32_t *) ((const char *) keys1+i*stride1));
                        uint32_t x2 = (*(const uint32_t *) ((const char *) keys2+i*stride2));
                        uint32_t x = k >= 3 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2;
                    }
                }
            } else if (k % 2 == 1) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x1 = tmpkeys[2*i+0];
                        uint32_t x2 = tmpkeys[2*i+1];
                        uint32_t x = k >= 3 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                        *(uint32_t *) ((char *) keys1+dst*stride1) = x1;
                        *(uint32_t *) ((char *) keys2+dst*stride2) = x2;
                        perm[dst] = invperm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint32_t x1 = tmpkeys[2*i+0];
                        uint32_t x2 = tmpkeys[2*i+1];
                        uint32_t x = k >= 3 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%3)))&(radix-1))]++;
                        *(uint32_t *) ((char *) keys1+dst*stride1) = x1;
                        *(uint32_t *) ((char *) keys2+dst*stride2) = x2;
                    }
                }
            }
            nrounds++;
        }

        /* post-processing to ensure output is returned in-place */
        if (nrounds == 0) {
            /* if all keys are zero, return identity permutation */
            if (perm && !freeperm) { for (int64_t i = 0; i < size; i++) perm[i] = i; }
            if (invperm && !freeinvperm) { for (int64_t i = 0; i < size; i++) invperm[i] = i; }
        } else if (nrounds % 2 == 1) {
            /* If nrounds is odd, then we stopped after completing an even
             * number of rounds, and we must perform a final swap of
             * the input and temporary arrays. The inverse permutation
             * is currently stored in 'invperm', so we use it to
             * compute the permutation 'perm', if needed. */
            #pragma omp for
            for (int64_t i = 0; i < size; i++) {
                *((uint32_t *) ((char *) keys1+i*stride1)) = tmpkeys[2*i+0];
                *((uint32_t *) ((char *) keys2+i*stride2)) = tmpkeys[2*i+1];
            }
            if (perm && invperm && !freeperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
            }
        } else if (nrounds % 2 == 0) {
            /* If nrounds is even, then we stopped after completing an
             * odd number of rounds, and the sorted keys are already
             * stored in the original array. The inverse permutation
             * is currently stored in 'perm', so we swap the 'perm'
             * and 'invperm', and then use 'invperm' to compute the
             * permutation 'perm', if needed. */
            if (perm && invperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) invperm[j] = perm[j];
                if (!freeperm) {
                    #pragma omp for
                    for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
                }
            }
        }
    }

#if defined(ACG_HAVE_OPENMP) && defined(__NVCOMPILER)
    free(maxkeys);
#endif
    if (freeperm) free(perm);
    if (freeinvperm) free(invperm);
    free(bucketptr); free(tmpkeys);
    return ACG_SUCCESS;
}

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
    int64_t * invperm)
{
    /*
     * Sort using 11 binary digits in each round, which yields 12
     * rounds for sorting pairs of 64-bit integers.
     *
     * The choice of using 11 bits in each round is described in the
     * article "Radix Tricks" by Michael Herf, published online in
     * December 2001 at http://stereopsis.com/radix.html.
     *
     * Double-buffering is used to move the keys into the correct
     * position in every round. As a result, at the end of every
     * even-numbered round (counting from zero), keys will be stored
     * in the auxiliary buffer. If the sorting ends after completing
     * an even number of rounds, then a final copy is performed at the
     * end to ensure that the sorted keys end up in the original array
     * that was provided as input.
     *
     * During the first round, the maximum key is found, and it is
     * used to exit early, if possible.
     *
     * If OpenMP is available, sorting will be performed in parallel.
     */
    const int bitsperplace = 11;
    const int radix = 1 << bitsperplace;
    const int nbits = CHAR_BIT*(sizeof(*keys1)+sizeof(*keys2));
    const int places = (nbits+bitsperplace-1) / bitsperplace;
    const int placesperitem = places / 2;
    int nthreads = 1;
#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        nthreads = omp_get_num_threads();
    }
#endif

    /* allocate storage for double-buffering keys */
    uint64_t * tmpkeys = malloc(2*size*sizeof(*tmpkeys));
    if (!tmpkeys) return ACG_ERR_ERRNO;

    /*
     * Allocate storage for each thread to count occurrences of digits
     * using the given base/radix.
     *
     * The number of occurrences in the i-th digit for the j-th thread
     * will be stored in bucketptr[j*radix+i].
     *
     * NOTE: If we pad the array with one additional entry, then we
     * can avoid the second pass over bucketptr to transform the
     * inclusive scan to an exclusive scan.
     */
    int64_t * bucketptr = malloc(nthreads*radix*sizeof(*bucketptr));
    if (!bucketptr) { free(tmpkeys); return ACG_ERR_ERRNO; }

    /* allocate storage for permutation and inverse permutation */
    bool freeperm = false;
    bool freeinvperm = false;
    if (!perm && invperm) {
        perm = malloc(size*sizeof(*perm));
        if (!perm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeperm = true;
    } else if (perm && !invperm) {
        invperm = malloc(size*sizeof(*invperm));
        if (!invperm) { free(bucketptr); free(tmpkeys); return ACG_ERR_ERRNO; }
        freeinvperm = true;
    }

    uint64_t maxkey[2] = {0,0};
#if defined(ACG_HAVE_OPENMP) && defined(__NVCOMPILER)
    uint64_t * maxkeys = malloc(2*nthreads*sizeof(*maxkeys));
    if (!maxkeys) return ACG_ERR_ERRNO;
    for (int i = 0; i < 2*nthreads; i++) maxkeys[i] = 0;
#endif
#ifdef ACG_HAVE_OPENMP
    #pragma omp parallel
#endif
    {
#ifdef ACG_HAVE_OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        int nrounds = 0;
        for (int k = 0; k < places; k++) {

            /* reset per-thread counters for each digit, and then
             * count occurrences of each digit */
            for (int j = 0; j < radix; j++) bucketptr[tid*radix+j] = 0;
            if (k == 0) {
#ifdef __NVCOMPILER
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x1 = (*(const uint64_t *) ((const char *) keys1+i*stride1));
                    uint64_t x2 = (*(const uint64_t *) ((const char *) keys2+i*stride2));
                    bucketptr[tid*radix+((x2>>(bitsperplace*(k%6)))&(radix-1))]++;
                    if (maxkeys[2*tid+0] < x1 || (maxkeys[2*tid+0] == x1 && maxkeys[2*tid+1] < x2)) {
                        maxkeys[2*tid+0] = x1, maxkeys[2*tid+1] = x2;
                    }
                }
                #pragma omp single
                for (int i = 0; i < nthreads; i++) {
                    if (maxkey[0] < maxkeys[2*i+0] || (maxkey[0] == maxkeys[2*i+0] && maxkey[1] < maxkeys[2*i+1])) {
                        maxkey[0] = maxkeys[2*i+0], maxkey[1] = maxkeys[2*i+1];
                    }
                }
#else
                #pragma omp for reduction(max:maxkey)
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x1 = (*(const uint64_t *) ((const char *) keys1+i*stride1));
                    uint64_t x2 = (*(const uint64_t *) ((const char *) keys2+i*stride2));
                    bucketptr[tid*radix+((x2>>(bitsperplace*(k%6)))&(radix-1))]++;
                    if (maxkey[0] < x1 || (maxkey[0] == x1 && maxkey[1] < x2)) {
                        maxkey[0] = x1, maxkey[1] = x2;
                    }
                }
#endif
            } else if (k % 2 == 0) {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x = k >= 6 ? (*(const uint64_t *) ((const char *) keys1+i*stride1))
                        : (*(const uint64_t *) ((const char *) keys2+i*stride2));
                    bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                }
            } else {
                #pragma omp for
                for (int64_t i = 0; i < size; i++) {
                    uint64_t x = k >= 6 ? tmpkeys[2*i+0] : tmpkeys[2*i+1];
                    bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                }
            }

            /* skip subsequent rounds, if the digits in the
             * corresponding places are all zero */
            if ((k < 6 && maxkey[1] < (1 << bitsperplace*k) && maxkey[0] == 0) ||
                (k >= 6 && maxkey[0] < (1 << bitsperplace*(k-6))))
                break;

            /* compute offset to first key for each digit and thread */
            #pragma omp master
            {
                for (int j = 1; j < nthreads; j++)
                    bucketptr[j*radix+0] += bucketptr[(j-1)*radix+0];
                for (int i = 1; i < radix; i++) {
                    bucketptr[0*radix+i] += bucketptr[(nthreads-1)*radix+i-1];
                    for (int j = 1; j < nthreads; j++)
                        bucketptr[j*radix+i] += bucketptr[(j-1)*radix+i];
                }

                for (int i = radix-1; i > 0; i--) {
                    for (int j = nthreads-1; j > 0; j--)
                        bucketptr[j*radix+i] = bucketptr[(j-1)*radix+i];
                    bucketptr[0*radix+i] = bucketptr[(nthreads-1)*radix+i-1];
                }
                for (int j = nthreads-1; j > 0; j--)
                    bucketptr[j*radix+0] = bucketptr[(j-1)*radix+0];
                bucketptr[0*radix+0] = 0;
            }
            #pragma omp barrier

            /*
             * Permute keys to sort them by digits in the kth place.
             * Keys are moved to the auxiliary array (tmpkeys) in even
             * rounds and back to the original array in odd rounds.
             * If the sorting permutation is needed, then the inverse
             * permutation will be stored alternately in 'invperm' and
             * 'perm' in even- and odd-numbered rounds, respectively.
             */
            if (k == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x1 = (*(const uint64_t *) ((const char *) keys1+i*stride1));
                        uint64_t x2 = (*(const uint64_t *) ((const char *) keys2+i*stride2));
                        uint64_t x = k >= 6 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2; invperm[dst] = i;
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x1 = (*(const uint64_t *) ((const char *) keys1+i*stride1));
                        uint64_t x2 = (*(const uint64_t *) ((const char *) keys2+i*stride2));
                        uint64_t x = k >= 6 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2;
                    }
                }
            } else if (k % 2 == 0) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x1 = (*(const uint64_t *) ((const char *) keys1+i*stride1));
                        uint64_t x2 = (*(const uint64_t *) ((const char *) keys2+i*stride2));
                        uint64_t x = k >= 6 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2; invperm[dst] = perm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x1 = (*(const uint64_t *) ((const char *) keys1+i*stride1));
                        uint64_t x2 = (*(const uint64_t *) ((const char *) keys2+i*stride2));
                        uint64_t x = k >= 6 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                        tmpkeys[2*dst+0] = x1; tmpkeys[2*dst+1] = x2;
                    }
                }
            } else if (k % 2 == 1) {
                if (perm && invperm) {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x1 = tmpkeys[2*i+0];
                        uint64_t x2 = tmpkeys[2*i+1];
                        uint64_t x = k >= 6 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                        *(uint64_t *) ((char *) keys1+dst*stride1) = x1;
                        *(uint64_t *) ((char *) keys2+dst*stride2) = x2;
                        perm[dst] = invperm[i];
                    }
                } else {
                    #pragma omp for
                    for (int64_t i = 0; i < size; i++) {
                        uint64_t x1 = tmpkeys[2*i+0];
                        uint64_t x2 = tmpkeys[2*i+1];
                        uint64_t x = k >= 6 ? x1 : x2;
                        int64_t dst = bucketptr[tid*radix+((x>>(bitsperplace*(k%6)))&(radix-1))]++;
                        *(uint64_t *) ((char *) keys1+dst*stride1) = x1;
                        *(uint64_t *) ((char *) keys2+dst*stride2) = x2;
                    }
                }
            }
            nrounds++;
        }

        /* post-processing to ensure output is returned in-place */
        if (nrounds == 0) {
            /* if all keys are zero, return identity permutation */
            if (perm && !freeperm) { for (int64_t i = 0; i < size; i++) perm[i] = i; }
            if (invperm && !freeinvperm) { for (int64_t i = 0; i < size; i++) invperm[i] = i; }
        } else if (nrounds % 2 == 1) {
            /* If nrounds is odd, then we stopped after completing an even
             * number of rounds, and we must perform a final swap of
             * the input and temporary arrays. The inverse permutation
             * is currently stored in 'invperm', so we use it to
             * compute the permutation 'perm', if needed. */
            #pragma omp for
            for (int64_t i = 0; i < size; i++) {
                *((uint64_t *) ((char *) keys1+i*stride1)) = tmpkeys[2*i+0];
                *((uint64_t *) ((char *) keys2+i*stride2)) = tmpkeys[2*i+1];
            }
            if (perm && invperm && !freeperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
            }
        } else if (nrounds % 2 == 0) {
            /* If nrounds is even, then we stopped after completing an
             * odd number of rounds, and the sorted keys are already
             * stored in the original array. The inverse permutation
             * is currently stored in 'perm', so we swap the 'perm'
             * and 'invperm', and then use 'invperm' to compute the
             * permutation 'perm', if needed. */
            if (perm && invperm) {
                #pragma omp for
                for (int64_t j = 0; j < size; j++) invperm[j] = perm[j];
                if (!freeperm) {
                    #pragma omp for
                    for (int64_t j = 0; j < size; j++) perm[invperm[j]] = j;
                }
            }
        }
    }

#if defined(ACG_HAVE_OPENMP) && defined(__NVCOMPILER)
    free(maxkeys);
#endif
    if (freeperm) free(perm);
    if (freeinvperm) free(invperm);
    free(bucketptr); free(tmpkeys);
    return ACG_SUCCESS;
}

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
    int64_t * invperm)
{
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        *(int32_t *)((char *) keys1+i*stride1) ^= INT32_MIN;
        *(int32_t *)((char *) keys2+i*stride2) ^= INT32_MIN;
    }
    int err = acgradixsortpair_uint32_t(
        size, stride1, (uint32_t *) keys1, stride2, (uint32_t *) keys2,
        perm, invperm);
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        *(int32_t *)((char *) keys1+i*stride1) ^= INT32_MIN;
        *(int32_t *)((char *) keys2+i*stride2) ^= INT32_MIN;
    }
    return err;
}

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
    int64_t * invperm)
{
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        *(int64_t *)((char *) keys1+i*stride1) ^= INT64_MIN;
        *(int64_t *)((char *) keys2+i*stride2) ^= INT64_MIN;
    }
    int err = acgradixsortpair_uint64_t(
        size, stride1, (uint64_t *) keys1, stride2, (uint64_t *) keys2,
        perm, invperm);
    #pragma omp parallel for
    for (int64_t i = 0; i < size; i++) {
        *(int64_t *)((char *) keys1+i*stride1) ^= INT64_MIN;
        *(int64_t *)((char *) keys2+i*stride2) ^= INT64_MIN;
    }
    return err;
}

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
    int64_t * invperm)
{
    if (sizeof(int) == sizeof(int32_t)) {
        return acgradixsortpair_int32_t(
            size, stride1, (int32_t *) keys1, stride2, (int32_t *) keys2,
            perm, invperm);
    } else if (sizeof(int) == sizeof(int64_t)) {
        return acgradixsortpair_int64_t(
            size, stride1, (int64_t *) keys1, stride2, (int64_t *) keys2,
            perm, invperm);
    } else { return ENOTSUP; }
}
