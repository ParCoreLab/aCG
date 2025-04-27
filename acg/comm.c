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
 * inter-process communication
 */

#include "acg/config.h"
#include "acg/error.h"
#include "acg/comm.h"
#include "acg/nvshmem.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif
#ifdef ACG_HAVE_NCCL
#include <nccl.h>
#endif
#ifdef ACG_HAVE_RCCL
#include <rccl/rccl.h>
#endif

#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* default communicators */
struct acgcomm ACG_COMM_NULL = { acgcomm_null };
#ifdef ACG_HAVE_MPI
struct acgcomm ACG_COMM_WORLD = { acgcomm_mpi, MPI_COMM_WORLD };
#endif

/**
 * ‘acgcommtypestr()’ returns a string for a communicator type.
 */
const char * acgcommtypestr(enum acgcommtype commtype)
{
    if (commtype == acgcomm_null) { return "null"; }
    else if (commtype == acgcomm_mpi) { return "mpi"; }
    else if (commtype == acgcomm_nccl) { return "nccl"; }
    else if (commtype == acgcomm_nvshmem) { return "nvshmem"; }
    else { return "unknown"; }
}

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_init_mpi()’ creates a communicator from a given MPI
 * communicator.
 */
int acgcomm_init_mpi(
    struct acgcomm * comm,
    MPI_Comm mpicomm,
    int * mpierrcode)
{
    int err = MPI_Comm_dup(mpicomm, &comm->mpicomm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    comm->type = acgcomm_mpi;
    return ACG_SUCCESS;
}
#endif

#if defined(ACG_HAVE_NCCL)
/**
 * ‘acgcomm_init_nccl()’ creates a communicator from a given NCCL
 * communicator.
 */
int acgcomm_init_nccl(
    struct acgcomm * comm,
    ncclComm_t ncclcomm,
    int * ncclerrcode)
{
    /*
     * disable the call to ncclCommSplit for now due to occasional
     * failures in large-scale testing. As a result, the caller must
     * not free 'ncclcomm' until after calling acgcomm_free.
     */

    /* ncclResult_t err = ncclCommSplit(ncclcomm, 0, 0, &comm->ncclcomm, NULL); */
    /* if (err != ncclSuccess) { if (ncclerrcode) *ncclerrcode = err; return ACG_ERR_NCCL; } */
    comm->ncclcomm = ncclcomm;
    comm->type = acgcomm_nccl;
    return ACG_SUCCESS;
}
#endif

#if defined(ACG_HAVE_RCCL)
/**
 * ‘acgcomm_init_rccl()’ creates a communicator from a given RCCL
 * communicator.
 */
int acgcomm_init_rccl(
    struct acgcomm * comm,
    ncclComm_t ncclcomm,
    int * rcclerrcode)
{
    /* int err = ncclCommSplit(ncclcomm, 0, 0, &comm->ncclcomm, NULL); */
    /* if (err != ncclSuccess) { if (rcclerrcode) *rcclerrcode = err; return ACG_ERR_RCCL; } */
    comm->ncclcomm = ncclcomm;
    comm->type = acgcomm_rccl;
    return ACG_SUCCESS;
}
#endif

/**
 * ‘acgcomm_free()’ frees resources associated with a communicator.
 */
void acgcomm_free(
    struct acgcomm * comm)
{
#if defined(ACG_HAVE_MPI)
    if (comm->type == acgcomm_mpi) MPI_Comm_free(&comm->mpicomm);
#endif
#if defined(ACG_HAVE_NCCL)
    /* if (comm->type == acgcomm_nccl) ncclCommDestroy(comm->ncclcomm); */
#endif
#if defined(ACG_HAVE_RCCL)
    /* if (comm->type == acgcomm_rccl) ncclCommDestroy(comm->ncclcomm); */
#endif
#if defined(ACG_HAVE_NVSHMEM)
    if (comm->type == acgcomm_nvshmem) acgcomm_free_nvshmem(comm);
#endif
}

/**
 * ‘acgcomm_size()’ size of a communicator (i.e., number of processes).
 */
int acgcomm_size(
    const struct acgcomm * comm,
    int * commsize)
{
    if (comm->type == acgcomm_null) return 1;
#if defined(ACG_HAVE_MPI)
    else if (comm->type == acgcomm_mpi) {
        int err = MPI_Comm_size(comm->mpicomm, commsize);
        if (err) return ACG_ERR_MPI;
        return ACG_SUCCESS;
    }
#endif
#if defined(ACG_HAVE_NCCL)
    else if (comm->type == acgcomm_nccl) {
        int err = ncclCommCount(comm->ncclcomm, commsize);
        if (err != ncclSuccess) return ACG_ERR_NCCL;
        return ACG_SUCCESS;
    }
#endif
#if defined(ACG_HAVE_RCCL)
    else if (comm->type == acgcomm_rccl) {
        int err = ncclCommCount(comm->ncclcomm, commsize);
        if (err != ncclSuccess) return ACG_ERR_RCCL;
        return ACG_SUCCESS;
    }
#endif
#if defined(ACG_HAVE_NVSHMEM)
    else if (comm->type == acgcomm_nvshmem) {
        return acgcomm_size_nvshmem(comm, commsize);
    }
#endif
    return ACG_ERR_INVALID_VALUE;
}

/**
 * ‘acgcomm_rank()’ rank of the current process in a communicator.
 */
int acgcomm_rank(
    const struct acgcomm * comm,
    int * rank)
{
    if (comm->type == acgcomm_null) return 0;
#if defined(ACG_HAVE_MPI)
    else if (comm->type == acgcomm_mpi) {
        int err = MPI_Comm_rank(comm->mpicomm, rank);
        if (err) return ACG_ERR_MPI;
        return ACG_SUCCESS;
    }
#endif
#if defined(ACG_HAVE_NCCL)
    else if (comm->type == acgcomm_nccl) {
        int err = ncclCommUserRank(comm->ncclcomm, rank);
        if (err != ncclSuccess) return ACG_ERR_NCCL;
        return ACG_SUCCESS;
    }
#endif
#if defined(ACG_HAVE_RCCL)
    else if (comm->type == acgcomm_rccl) {
        int err = ncclCommUserRank(comm->ncclcomm, rank);
        if (err != ncclSuccess) return ACG_ERR_RCCL;
        return ACG_SUCCESS;
    }
#endif
#if defined(ACG_HAVE_NVSHMEM)
    else if (comm->type == acgcomm_nvshmem) {
        return acgcomm_rank_nvshmem(comm, rank);
    }
#endif
    return ACG_ERR_INVALID_VALUE;
}

/*
 * data types
 */

/**
 * ‘acgdatatypestr()’ returns a string for a data type.
 */
const char * acgdatatypestr(enum acgdatatype datatype)
{
    if (datatype == ACG_DOUBLE) { return "double"; }
    else { return "unknown"; }
}

/**
 * ‘acgdatatype_size()’ returns the size (in bytes) of a data type.
 */
int acgdatatype_size(enum acgdatatype datatype, int * size)
{
    if (datatype == ACG_DOUBLE) *size = sizeof(double);
    else return ACG_ERR_INVALID_VALUE;
    return ACG_SUCCESS;
}

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgdatatype_mpi()’ returns a corresponding MPI_Datatype for the
 * given data type.
 */
MPI_Datatype acgdatatype_mpi(enum acgdatatype datatype)
{
    if (datatype == ACG_DOUBLE) return MPI_DOUBLE;
    return MPI_DATATYPE_NULL;
}
#endif

#if defined(ACG_HAVE_NCCL) || defined(ACG_HAVE_RCCL)
/**
 * ‘acgdatatype_nccl()’ returns a corresponding NCCL_Datatype for the
 * given data type.
 */
ncclDataType_t acgdatatype_nccl(enum acgdatatype datatype)
{
    if (datatype == ACG_DOUBLE) return ncclDouble;
    return -1;
}
#endif

/*
 * operations
 */

/**
 * ‘acgopstr()’ returns a string for an operation.
 */
const char * acgopstr(enum acgop op)
{
    if (op == ACG_SUM) { return "sum"; }
    else { return "unknown"; }
}

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgop_mpi()’ returns a corresponding MPI_Op.
 */
MPI_Op acgop_mpi(enum acgop op)
{
    if (op == ACG_SUM) return MPI_SUM;
    return MPI_OP_NULL;
}
#endif

#if defined(ACG_HAVE_NCCL) || defined(ACG_HAVE_RCCL)
/**
 * ‘acgop_nccl()’ returns a corresponding ncclRedOp_t.
 */
ncclRedOp_t acgop_nccl(enum acgop op)
{
    if (op == ACG_SUM) return ncclSum;
    return -1;
}
#endif

/*
 * collective communication
 */

#ifdef ACG_HAVE_CUDA
/**
 * ‘acgcomm_barrier()’ performs barrier synchronisation.
 */
int acgcomm_barrier(
    cudaStream_t stream,
    const struct acgcomm * comm,
    int * errcode)
{
    int err;
    if (comm->type == acgcomm_null) return ACG_SUCCESS;
    else if (comm->type == acgcomm_mpi) {
#if defined(ACG_HAVE_MPI)
        cudaStreamSynchronize(stream);
        err = MPI_Barrier(comm->mpicomm);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_MPI; }
#else
        return ACG_ERR_MPI_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nccl) {
#if defined(ACG_HAVE_NCCL)
        err = ncclAllReduce(NULL, NULL, 0, ncclInt, ncclSum, comm->ncclcomm, stream);
        if (err != ncclSuccess) { if (errcode) *errcode = err; return ACG_ERR_NCCL; }
#else
        return ACG_ERR_NCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nvshmem) {
#if defined(ACG_HAVE_NVSHMEM)
        acg_nvshmemx_barrier_all_on_stream(stream);
#else
        return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}

/**
 * ‘acgcomm_allreduce()’ performs an all-reduce operation on a double
 * precision floating point value.
 */
int acgcomm_allreduce(
    const void * src,
    void * dst,
    int count,
    enum acgdatatype datatype,
    enum acgop op,
    cudaStream_t stream,
    const struct acgcomm * comm,
    int * errcode)
{
    int err;
    if (comm->type == acgcomm_null) return ACG_SUCCESS;
    else if (comm->type == acgcomm_mpi) {
#if defined(ACG_HAVE_MPI)
        cudaStreamSynchronize(stream);
        err = MPI_Allreduce(
            src, dst, count, acgdatatype_mpi(datatype), acgop_mpi(op), comm->mpicomm);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_MPI; }
#else
        return ACG_ERR_MPI_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nccl) {
#if defined(ACG_HAVE_NCCL)
        err = ncclAllReduce(
            src == ACG_IN_PLACE ? dst : src, dst,
            count, acgdatatype_nccl(datatype), acgop_nccl(op),
            comm->ncclcomm, stream);
        if (err != ncclSuccess) { if (errcode) *errcode = err; return ACG_ERR_NCCL; }
#else
        return ACG_ERR_NCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nvshmem) {
#if defined(ACG_HAVE_NVSHMEM)
        if (op == ACG_SUM) {
            err = acg_nvshmemx_double_sum_reduce_on_stream(
                ACG_NVSHMEM_TEAM_WORLD, dst, src == ACG_IN_PLACE ? dst : src,
                count, stream);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_NVSHMEM; }
        } else return ACG_ERR_INVALID_VALUE;
#else
        return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}
#endif

#ifdef ACG_HAVE_HIP
/**
 * ‘acgcomm_barrier_hip()’ performs barrier synchronisation.
 */
int acgcomm_barrier_hip(
    hipStream_t stream,
    const struct acgcomm * comm,
    int * errcode)
{
    int err;
    if (comm->type == acgcomm_null) return ACG_SUCCESS;
    else if (comm->type == acgcomm_mpi) {
#if defined(ACG_HAVE_MPI)
        hipStreamSynchronize(stream);
        err = MPI_Barrier(comm->mpicomm);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_MPI; }
#else
        return ACG_ERR_MPI_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_rccl) {
#if defined(ACG_HAVE_RCCL)
        err = ncclAllReduce(NULL, NULL, 0, ncclInt, ncclSum, comm->ncclcomm, stream);
        if (err != ncclSuccess) { if (errcode) *errcode = err; return ACG_ERR_RCCL; }
#else
        return ACG_ERR_RCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_rocshmem) {
#if defined(ACG_HAVE_ROCSHMEM)
        acg_rocshmemx_barrier_all_on_stream(stream);
#else
        return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}

/**
 * ‘acgcomm_allreduce_hip()’ performs an all-reduce operation on a double
 * precision floating point value.
 */
int acgcomm_allreduce_hip(
    const void * src,
    void * dst,
    int count,
    enum acgdatatype datatype,
    enum acgop op,
    hipStream_t stream,
    const struct acgcomm * comm,
    int * errcode)
{
    int err;
    if (comm->type == acgcomm_null) return ACG_SUCCESS;
    else if (comm->type == acgcomm_mpi) {
#if defined(ACG_HAVE_MPI)
        hipStreamSynchronize(stream);
        err = MPI_Allreduce(
            src, dst, count, acgdatatype_mpi(datatype), acgop_mpi(op), comm->mpicomm);
        if (err) { if (errcode) *errcode = err; return ACG_ERR_MPI; }
#else
        return ACG_ERR_MPI_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_rccl) {
#if defined(ACG_HAVE_RCCL)
        err = ncclAllReduce(
            src == ACG_IN_PLACE ? dst : src, dst,
            count, acgdatatype_nccl(datatype), acgop_nccl(op),
            comm->ncclcomm, stream);
        if (err != ncclSuccess) { if (errcode) *errcode = err; return ACG_ERR_RCCL; }
#else
        return ACG_ERR_RCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_rocshmem) {
#if defined(ACG_HAVE_ROCSHMEM)
        if (op == ACG_SUM) {
            err = acg_rocshmemx_double_sum_reduce_on_stream(
                ACG_ROCSHMEM_TEAM_WORLD, dst, src == ACG_IN_PLACE ? dst : src,
                count, stream);
            if (err) { if (errcode) *errcode = err; return ACG_ERR_ROCSHMEM; }
        } else return ACG_ERR_INVALID_VALUE;
#else
        return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}
#endif
