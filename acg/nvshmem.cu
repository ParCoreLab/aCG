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
 * wrapper functions to provide a C API for NVSHMEM
 */

#include "acg/config.h"
#include "acg/nvshmem.h"

#if defined(ACG_HAVE_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(ACG_HAVE_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ACG_HAVE_CUDA) && defined(ACG_HAVE_NVSHMEM)

/*
 * library handles
 */

nvshmem_team_t acg_nvshmem_team(acg_nvshmem_team_t team)
{
    if (team == ACG_NVSHMEM_TEAM_INVALID) return NVSHMEM_TEAM_INVALID;
    else if (team == ACG_NVSHMEM_TEAM_WORLD) return NVSHMEM_TEAM_WORLD;
    else if (team == ACG_NVSHMEM_TEAM_SHARED) return NVSHMEM_TEAM_SHARED;
    else if (team == ACG_NVSHMEMX_TEAM_NODE) return NVSHMEMX_TEAM_NODE;
    else return team;
}

/*
 * library setup, exit, and query
 */

void acg_nvshmem_init(void) { nvshmem_init(); }
int acg_nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attributes) { return nvshmemx_init_attr(flags, attributes); }
int acg_nvshmem_my_pe(void) { return nvshmem_my_pe(); }
int acg_nvshmem_n_pes(void) { return nvshmem_n_pes(); }
void acg_nvshmem_finalize(void) { nvshmem_finalize(); }
void acg_nvshmem_info_get_version(int *major, int *minor) { nvshmem_info_get_version(major, minor); }
void acg_nvshmem_info_get_name(char *name) { nvshmem_info_get_name(name); }
#if NVSHMEM_MAJOR_VERSION >= 3
void acg_nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch) { nvshmemx_vendor_get_version_info(major, minor, patch); }
#else
void acg_nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch) { *major = NVSHMEM_VENDOR_MAJOR_VERSION; *minor = NVSHMEM_VENDOR_MINOR_VERSION; *patch = NVSHMEM_VENDOR_PATCH_VERSION; }
#endif

/*
 * memory management
 */

void *acg_nvshmem_malloc(size_t size) { return nvshmem_malloc(size); }
void acg_nvshmem_free(void *ptr) { nvshmem_free(ptr); }
void *acg_nvshmem_align(size_t alignment, size_t size) { return nvshmem_align(alignment, size); }
void *acg_nvshmem_calloc(size_t count, size_t size) { return nvshmem_calloc(count, size); }

/*
 * implicit team collectives
 */

void acg_nvshmem_barrier_all(void) { nvshmem_barrier_all(); }
void acg_nvshmemx_barrier_all_on_stream(cudaStream_t stream) { nvshmemx_barrier_all_on_stream(stream); }
void acg_nvshmem_sync_all(void) { nvshmem_sync_all(); }
void acg_nvshmemx_sync_all_on_stream(cudaStream_t stream) { nvshmemx_sync_all_on_stream(stream); }
int acg_nvshmem_double_sum_reduce(acg_nvshmem_team_t team, double *dest, const double *source, size_t nreduce) { return nvshmem_double_sum_reduce(acg_nvshmem_team(team), dest, source, nreduce); }
int acg_nvshmemx_double_sum_reduce_on_stream(acg_nvshmem_team_t team, double *dest, const double *source, size_t nreduce, cudaStream_t stream) { return nvshmemx_double_sum_reduce_on_stream(acg_nvshmem_team(team), dest, source, nreduce, stream); }
#endif

#ifdef __cplusplus
}
#endif
