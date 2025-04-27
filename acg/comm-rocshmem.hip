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
 * ROCSHMEM inter-process communication
 */

#include "acg/config.h"
#include "acg/error.h"
#include "acg/comm.h"

#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif
#ifdef ACG_HAVE_ROCSHMEM
#include <rocshmem.h>
#endif

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ‘acgcomm_rocshmem_version()’ prints version information for ROCSHMEM.
 */
int acgcomm_rocshmem_version(
    FILE * f)
{
#if defined(ACG_HAVE_ROCSHMEM)
    int major, minor, patch;
    rocshmemx_vendor_get_version_info(&major, &minor, &patch);
    char name[ROCSHMEM_MAX_NAME_LEN];
    rocshmem_info_get_name(name);
    fprintf(f, "ROCSHMEM version %d.%d.%d (%s)\n", major, minor, patch, name);
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_rocshmem_init()’ initialise ROCSHMEM library.
 */
int acgcomm_rocshmem_init(
    MPI_Comm mpicomm,
    int root,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
#ifdef ROCSHMEM_MPI_SUPPORT
    rocshmemx_init_attr_t attr = ROCSHMEMX_INIT_ATTR_INITIALIZER;
    attr.mpi_comm = &mpicomm;
    int err = rocshmemx_init_attr(ROCSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_ROCSHMEM; }
#else
    int commsize, rank;
    MPI_Comm_size(mpicomm, &commsize);
    MPI_Comm_rank(mpicomm, &rank);
    rocshmemx_uniqueid_t rocshmemuid;
    if (rank == root) rocshmemx_get_uniqueid(&rocshmemuid);
    MPI_Bcast(&rocshmemuid, sizeof(rocshmemuid), MPI_BYTE, root, mpicomm);
    rocshmemx_init_attr_t attr = ROCSHMEMX_INIT_ATTR_INITIALIZER;
    rocshmemx_set_attr_uniqueid_args(rank, commsize, &rocshmemuid, &attr);
    int err = rocshmemx_init_attr(ROCSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_ROCSHMEM; }
#endif
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}
#endif

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_init_rocshmem()’ creates a communicator from a given
 * ROCSHMEM communicator.
 */
int acgcomm_init_rocshmem(
    struct acgcomm * comm,
    MPI_Comm mpicomm,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
    int err = MPI_Comm_dup(mpicomm, &comm->mpicomm);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_MPI; }
    comm->type = acgcomm_rocshmem;
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}
#endif

/**
 * ‘acgcomm_free_rocshmem()’ frees resources associated with a communicator.
 */
void acgcomm_free_rocshmem(
    struct acgcomm * comm)
{
#if defined(ACG_HAVE_ROCSHMEM)
    cudaDeviceSynchronize();
    if (comm->type == acgcomm_rocshmem) MPI_Comm_free(&comm->mpicomm);
#endif
}

/**
 * ‘acgcomm_size_rocshmem()’ size of a communicator (i.e., number of processes).
 */
int acgcomm_size_rocshmem(
    const struct acgcomm * comm,
    int * commsize)
{
#if defined(ACG_HAVE_ROCSHMEM)
    if (comm->type == acgcomm_rocshmem) {
        *commsize = rocshmem_n_pes();
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_rank_rocshmem()’ rank of the current process in a
 * communicator.
 */
int acgcomm_rank_rocshmem(
    const struct acgcomm * comm,
    int * rank)
{
#if defined(ACG_HAVE_ROCSHMEM)
    if (comm->type == acgcomm_rocshmem) {
        *rank = rocshmem_my_pe();
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_rocshmem_malloc()’ allocates storage on the symmetric heap
 * for use with ROCSHMEM.
 */
int acgcomm_rocshmem_malloc(
    void ** ptr,
    size_t size,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
    *ptr = rocshmem_malloc(size);
    if (size > 0 && !*ptr) return ACG_ERR_ROCSHMEM;
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}

/**
 * ‘acgcomm_rocshmem_calloc()’ allocates storage on the symmetric heap
 * for use with ROCSHMEM.
 */
int acgcomm_rocshmem_calloc(
    void ** ptr,
    size_t count,
    size_t size,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
    *ptr = rocshmem_calloc(count, size);
    if (size > 0 && !*ptr) return ACG_ERR_ROCSHMEM;
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}

/**
 * ‘acgcomm_rocshmem_free()’ frees storage allocated for ROCSHMEM.
 */
void acgcomm_rocshmem_free(
    void * ptr)
{
#if defined(ACG_HAVE_ROCSHMEM)
    rocshmem_free(ptr);
#endif
}

/**
 * ‘acgcomm_rocshmem_register_buffer()’ registers a buffer for use with
 * ROCSHMEM.
 */
int acgcomm_rocshmem_register_buffer(
    const struct acgcomm * comm,
    void * addr,
    size_t length,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
    if (comm->type == acgcomm_rocshmem) {
        if (length == 0) return ACG_SUCCESS;
        int err = rocshmemx_buffer_register(addr, length);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_ROCSHMEM; }
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_rocshmem_unregister_buffer()’ unregisters a buffer for use
 * with ROCSHMEM.
 */
int acgcomm_rocshmem_unregister_buffer(
    void * addr,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
    int err = rocshmemx_buffer_unregister(addr);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_ROCSHMEM; }
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}

#ifdef ACG_HAVE_HIP
/**
 * ‘acgcomm_rocshmem_allreduce()’ performs an all-reduce operation on a
 * double precision floating point value.
 */
int acgcomm_rocshmem_allreduce(
    const struct acgcomm * comm,
    double * dest,
    const double * source,
    int nreduce,
    hipStream_t stream,
    int * errcode)
{
#if defined(ACG_HAVE_ROCSHMEM)
    if (comm->type == acgcomm_rocshmem) {
        int err = rocshmemx_double_sum_reduce_on_stream(ROCSHMEM_TEAM_WORLD, dest, source, nreduce, stream);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_ROCSHMEM; }
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}
#endif

/**
 * ‘acgcomm_rocshmem_barrier_all()’ performs barrier synchronization.
 */
ACG_API int acgcomm_rocshmem_barrier_all(void)
{
#if defined(ACG_HAVE_ROCSHMEM)
    rocshmem_barrier_all();
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}

#ifdef __cplusplus
}
#endif
