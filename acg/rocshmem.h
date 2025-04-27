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
 * wrapper functions to provide a C API for ROCSHMEM
 */

#ifndef ACG_ROCSHMEM_H
#define ACG_ROCSHMEM_H

#include "acg/config.h"

#if defined(ACG_HAVE_HIP)
#include <hip/hip_runtime_api.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ACG_HAVE_HIP) && defined(ACG_HAVE_ROCSHMEM)
/*
 * library constants
 */

/* ... */

/*
 * library handles
 */

typedef int32_t acg_rocshmem_team_t;
enum {
    ACG_ROCSHMEM_TEAM_INVALID = -1,
    ACG_ROCSHMEM_TEAM_WORLD = 0,
    ACG_ROCSHMEM_TEAM_SHARED = 1,
    ACG_ROCSHMEMX_TEAM_NODE = 2,
};

/*
 * library setup, exit, and query
 */

void acg_rocshmem_init(void);
/* int acg_rocshmemx_init_attr(unsigned int flags, rocshmemx_init_attr_t *attributes); */
int acg_rocshmem_my_pe(void);
int acg_rocshmem_n_pes(void);
void acg_rocshmem_finalize(void);
void acg_rocshmem_info_get_version(int *major, int *minor);
void acg_rocshmem_info_get_name(char *name);
void acg_rocshmemx_vendor_get_version_info(int *major, int *minor, int *patch);

/*
 * memory management
 */

void *acg_rocshmem_malloc(size_t size);
void acg_rocshmem_free(void *ptr);
void *acg_rocshmem_align(size_t alignment, size_t size);
void *acg_rocshmem_calloc(size_t count, size_t size);

/*
 * implicit team collectives
 */

void acg_rocshmem_barrier_all(void);
void acg_rocshmemx_barrier_all_on_stream(hipStream_t stream);
void acg_rocshmem_sync_all(void);
void acg_rocshmemx_sync_all_on_stream(hipStream_t stream);
int acg_rocshmem_double_sum_reduce(acg_rocshmem_team_t team, double *dest, const double *source, size_t nreduce);
int acg_rocshmemx_double_sum_reduce_on_stream(acg_rocshmem_team_t team, double *dest, const double *source, size_t nreduce, hipStream_t stream);
#endif

#ifdef __cplusplus
}
#endif

#endif
