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
 * NVSHMEM inter-process communication
 */

#include "acg/config.h"
#include "acg/error.h"
#include "acg/comm.h"

#ifdef ACG_HAVE_NVSHMEM
#include <nvshmem.h>
#endif

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ‘acgcomm_nvshmem_version()’ prints version information for NVSHMEM.
 */
int acgcomm_nvshmem_version(
    FILE * f)
{
#if defined(ACG_HAVE_NVSHMEM)
#if NVSHMEM_MAJOR_VERSION >= 3
    int major, minor, patch;
    nvshmemx_vendor_get_version_info(&major, &minor, &patch);
    char name[NVSHMEM_MAX_NAME_LEN];
    nvshmem_info_get_name(name);
    fprintf(f, "NVSHMEM version %d.%d.%d (%s)\n", major, minor, patch, name);
#else
    fprintf(f, "NVSHMEM version %d.%d\n", NVSHMEM_MAJOR_VERSION, NVSHMEM_MINOR_VERSION);
#endif
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_nvshmem_init()’ initialise NVSHMEM library.
 */
int acgcomm_nvshmem_init(
    MPI_Comm mpicomm,
    int root,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
#ifdef NVSHMEM_MPI_SUPPORT
#if NVSHMEM_MAJOR_VERSION >= 3
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
#else
    nvshmemx_init_attr_t attr;
#endif
    attr.mpi_comm = &mpicomm;
    int err = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
#else
    int commsize, rank;
    MPI_Comm_size(mpicomm, &commsize);
    MPI_Comm_rank(mpicomm, &rank);
    nvshmemx_uniqueid_t nvshmemuid;
    if (rank == root) nvshmemx_get_uniqueid(&nvshmemuid);
    MPI_Bcast(&nvshmemuid, sizeof(nvshmemuid), MPI_BYTE, root, mpicomm);
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, commsize, &nvshmemuid, &attr);
    int err = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
#endif
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}
#endif

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_init_nvshmem()’ creates a communicator from a given
 * NVSHMEM communicator.
 */
int acgcomm_init_nvshmem(
    struct acgcomm * comm,
    MPI_Comm mpicomm,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
    int err = MPI_Comm_dup(mpicomm, &comm->mpicomm);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_MPI; }
    comm->type = acgcomm_nvshmem;
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}
#endif

/**
 * ‘acgcomm_free_nvshmem()’ frees resources associated with a communicator.
 */
void acgcomm_free_nvshmem(
    struct acgcomm * comm)
{
#if defined(ACG_HAVE_NVSHMEM)
    cudaDeviceSynchronize();
    if (comm->type == acgcomm_nvshmem) MPI_Comm_free(&comm->mpicomm);
#endif
}

/**
 * ‘acgcomm_size_nvshmem()’ size of a communicator (i.e., number of processes).
 */
int acgcomm_size_nvshmem(
    const struct acgcomm * comm,
    int * commsize)
{
#if defined(ACG_HAVE_NVSHMEM)
    if (comm->type == acgcomm_nvshmem) {
        *commsize = nvshmem_n_pes();
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_rank_nvshmem()’ rank of the current process in a
 * communicator.
 */
int acgcomm_rank_nvshmem(
    const struct acgcomm * comm,
    int * rank)
{
#if defined(ACG_HAVE_NVSHMEM)
    if (comm->type == acgcomm_nvshmem) {
        *rank = nvshmem_my_pe();
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_nvshmem_malloc()’ allocates storage on the symmetric heap
 * for use with NVSHMEM.
 */
int acgcomm_nvshmem_malloc(
    void ** ptr,
    size_t size,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
    *ptr = nvshmem_malloc(size);
    if (size > 0 && !*ptr) return ACG_ERR_NVSHMEM;
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}

/**
 * ‘acgcomm_nvshmem_calloc()’ allocates storage on the symmetric heap
 * for use with NVSHMEM.
 */
int acgcomm_nvshmem_calloc(
    void ** ptr,
    size_t count,
    size_t size,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
    *ptr = nvshmem_calloc(count, size);
    if (size > 0 && !*ptr) return ACG_ERR_NVSHMEM;
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}

/**
 * ‘acgcomm_nvshmem_free()’ frees storage allocated for NVSHMEM.
 */
void acgcomm_nvshmem_free(
    void * ptr)
{
#if defined(ACG_HAVE_NVSHMEM)
    nvshmem_free(ptr);
#endif
}

/**
 * ‘acgcomm_nvshmem_register_buffer()’ registers a buffer for use with
 * NVSHMEM.
 */
int acgcomm_nvshmem_register_buffer(
    const struct acgcomm * comm,
    void * addr,
    size_t length,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
    if (comm->type == acgcomm_nvshmem) {
        if (length == 0) return ACG_SUCCESS;
        int err = nvshmemx_buffer_register(addr, length);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_nvshmem_unregister_buffer()’ unregisters a buffer for use
 * with NVSHMEM.
 */
int acgcomm_nvshmem_unregister_buffer(
    void * addr,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
    int err = nvshmemx_buffer_unregister(addr);
    if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}

/**
 * ‘acgcomm_nvshmem_allreduce()’ performs an all-reduce operation on a
 * double precision floating point value.
 */
int acgcomm_nvshmem_allreduce(
    const struct acgcomm * comm,
    double * dest,
    const double * source,
    int nreduce,
    cudaStream_t stream,
    int * errcode)
{
#if defined(ACG_HAVE_NVSHMEM)
    if (comm->type == acgcomm_nvshmem) {
        int err = nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, dest, source, nreduce, stream);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
        return ACG_SUCCESS;
    }
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_nvshmem_barrier_all()’ performs barrier synchronization.
 */
ACG_API int acgcomm_nvshmem_barrier_all(void)
{
#if defined(ACG_HAVE_NVSHMEM)
    nvshmem_barrier_all();
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}

#ifdef __cplusplus
}
#endif
