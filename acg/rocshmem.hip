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

#include "acg/config.h"
#include "acg/rocshmem.h"

#if defined(ACG_HAVE_HIP)
#include <hip/hip_runtime_api.h>
#endif

#if defined(ACG_HAVE_ROCSHMEM)
#include <rocshmem.h>
#include <rocshmemx.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ACG_HAVE_HIP) && defined(ACG_HAVE_ROCSHMEM)

/*
 * library handles
 */

rocshmem_team_t acg_rocshmem_team(acg_rocshmem_team_t team)
{
    if (team == ACG_ROCSHMEM_TEAM_INVALID) return ROCSHMEM_TEAM_INVALID;
    else if (team == ACG_ROCSHMEM_TEAM_WORLD) return ROCSHMEM_TEAM_WORLD;
    else if (team == ACG_ROCSHMEM_TEAM_SHARED) return ROCSHMEM_TEAM_SHARED;
    else if (team == ACG_ROCSHMEMX_TEAM_NODE) return ROCSHMEMX_TEAM_NODE;
    else return team;
}

/*
 * library setup, exit, and query
 */

void acg_rocshmem_init(void) { rocshmem_init(); }
int acg_rocshmemx_init_attr(unsigned int flags, rocshmemx_init_attr_t *attributes) { return rocshmemx_init_attr(flags, attributes); }
int acg_rocshmem_my_pe(void) { return rocshmem_my_pe(); }
int acg_rocshmem_n_pes(void) { return rocshmem_n_pes(); }
void acg_rocshmem_finalize(void) { rocshmem_finalize(); }
void acg_rocshmem_info_get_version(int *major, int *minor) { rocshmem_info_get_version(major, minor); }
void acg_rocshmem_info_get_name(char *name) { rocshmem_info_get_name(name); }
void acg_rocshmemx_vendor_get_version_info(int *major, int *minor, int *patch) { rocshmemx_vendor_get_version_info(major, minor, patch); }

/*
 * memory management
 */

void *acg_rocshmem_malloc(size_t size) { return rocshmem_malloc(size); }
void acg_rocshmem_free(void *ptr) { rocshmem_free(ptr); }
void *acg_rocshmem_align(size_t alignment, size_t size) { return rocshmem_align(alignment, size); }
void *acg_rocshmem_calloc(size_t count, size_t size) { return rocshmem_calloc(count, size); }

/*
 * implicit team collectives
 */

void acg_rocshmem_barrier_all(void) { rocshmem_barrier_all(); }
void acg_rocshmemx_barrier_all_on_stream(hipStream_t stream) { rocshmemx_barrier_all_on_stream(stream); }
void acg_rocshmem_sync_all(void) { rocshmem_sync_all(); }
void acg_rocshmemx_sync_all_on_stream(hipStream_t stream) { rocshmemx_sync_all_on_stream(stream); }
int acg_rocshmem_double_sum_reduce(acg_rocshmem_team_t team, double *dest, const double *source, size_t nreduce) { return rocshmem_double_sum_reduce(acg_rocshmem_team(team), dest, source, nreduce); }
int acg_rocshmemx_double_sum_reduce_on_stream(acg_rocshmem_team_t team, double *dest, const double *source, size_t nreduce, cudaStream_t stream) { return rocshmemx_double_sum_reduce_on_stream(acg_rocshmem_team(team), dest, source, nreduce, stream); }
#endif

#ifdef __cplusplus
}
#endif
