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
 */

#ifndef ACG_CONFIG_H
#define ACG_CONFIG_H

#include <inttypes.h>
#include <stdint.h>

/* Define if you have zlib. */
/* #undef ACG_HAVE_LIBZ */

/* Define if you have an MPI library. */
/* #undef ACG_HAVE_MPI */

/* Define if you have METIS */
/* #undef ACG_HAVE_METIS */

/* Define if you have OpenMP. */
/* #undef ACG_HAVE_OPENMP */

/* Define to 1 or 0, depending whether the compiler supports simple visibility
   declarations. */
/* #undef HAVE_VISIBILITY */

/* symbol visibility */
#if HAVE_VISIBILITY && ACG_API_EXPORT
#define ACG_API __attribute__((__visibility__("default")))
#else
#define ACG_API
#endif

/* integer type width */
/* #undef ACG_IDX_SIZE */

#ifndef ACG_IDX_SIZE
typedef int acgidx_t;
#define PRIdx "d"
#define ACGIDX_T_MIN INT_MIN
#define ACGIDX_T_MAX INT_MAX
#ifdef ACG_HAVE_MPI
#define MPI_ACGIDX_T MPI_INT
#define CUSPARSE_IDX_T CUSPARSE_INDEX_32I
#define HIPSPARSE_IDX_T HIPSPARSE_INDEX_32I
#endif
#elif ACG_IDX_SIZE == 32
typedef int32_t acgidx_t;
#define PRIdx PRId32
#define ACGIDX_T_MIN INT32_MIN
#define ACGIDX_T_MAX INT32_MAX
#ifdef ACG_HAVE_MPI
#define MPI_ACGIDX_T MPI_INT32_T
#define CUSPARSE_IDX_T CUSPARSE_INDEX_32I
#define HIPSPARSE_IDX_T HIPSPARSE_INDEX_32I
#endif
#elif ACG_IDX_SIZE == 64
typedef int64_t acgidx_t;
#define PRIdx PRId64
#define ACGIDX_T_MIN INT64_MIN
#define ACGIDX_T_MAX INT64_MAX
#ifdef ACG_HAVE_MPI
#define MPI_ACGIDX_T MPI_INT64_T
#define CUSPARSE_IDX_T CUSPARSE_INDEX_64I
#define HIPSPARSE_IDX_T HIPSPARSE_INDEX_64I
#endif
#else
#error "invalid ACG_IDX_SIZE; expected 32 or 64"
#endif

#endif
