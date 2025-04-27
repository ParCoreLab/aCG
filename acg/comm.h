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

#ifndef ACG_COMM_H
#define ACG_COMM_H

#include "acg/config.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif
#ifdef ACG_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef ACG_HAVE_NCCL
#include <nccl.h>
#endif
#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif
#ifdef ACG_HAVE_RCCL
#include <rccl/rccl.h>
#endif

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * default communicators
 */

/* default communicators */
struct acgcomm;
extern struct acgcomm ACG_COMM_NULL;
#ifdef ACG_HAVE_MPI
extern struct acgcomm ACG_COMM_WORLD;
#endif

#if defined(ACG_HAVE_MPI)
#define ACG_IN_PLACE MPI_IN_PLACE
#else
#define ACG_IN_PLACE ((void *) 1)
#endif

/*
 * communicators
 */

/**
 * ‘acgcommtype’ is a type for enumerating different kinds of
 * communicators.
 */
enum acgcommtype
{
    acgcomm_null,     /* null communicator */
    acgcomm_mpi,      /* MPI communicator */
    acgcomm_nccl,     /* NCCL communicator */
    acgcomm_rccl,     /* RCCL communicator */
    acgcomm_nvshmem,  /* NVSHMEM communicator */
    acgcomm_rocshmem,  /* rocSHMEM communicator */
};

/**
 * ‘acgcommtypestr()’ returns a string for a communicator type.
 */
ACG_API const char * acgcommtypestr(enum acgcommtype commtype);

/**
 * ‘acgcomm’ is a data structure used to represent a communicator for
 * inter-process communication using, e.g., MPI, NNCL or RCCL.
 */
struct acgcomm
{
    /**
     * ‘type’ specifies the type of the underlying communicator.
     */
    enum acgcommtype type;

#if defined(ACG_HAVE_MPI)
    MPI_Comm mpicomm;
#endif

#if defined(ACG_HAVE_NCCL) || defined(ACG_HAVE_RCCL)
    ncclComm_t ncclcomm;
#endif
};

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_init_mpi()’ creates a communicator from a given MPI
 * communicator.
 */
ACG_API int acgcomm_init_mpi(
    struct acgcomm * comm,
    MPI_Comm mpicomm,
    int * mpierrcode);
#endif

#if defined(ACG_HAVE_NCCL)
/**
 * ‘acgcomm_init_nccl()’ creates a communicator from a given NCCL
 * communicator.
 */
ACG_API int acgcomm_init_nccl(
    struct acgcomm * comm,
    ncclComm_t ncclcomm,
    int * ncclerrcode);
#endif

#if defined(ACG_HAVE_RCCL)
/**
 * ‘acgcomm_init_rccl()’ creates a communicator from a given RCCL
 * communicator.
 */
ACG_API int acgcomm_init_rccl(
    struct acgcomm * comm,
    ncclComm_t ncclcomm,
    int * rcclerrcode);
#endif

/**
 * ‘acgcomm_free()’ frees resources associated with a communicator.
 */
ACG_API void acgcomm_free(
    struct acgcomm * comm);

/**
 * ‘acgcomm_size()’ size of a communicator (i.e., number of processes).
 */
ACG_API int acgcomm_size(
    const struct acgcomm * comm,
    int * commsize);

/**
 * ‘acgcomm_rank()’ rank of the current process in a communicator.
 */
ACG_API int acgcomm_rank(
    const struct acgcomm * comm,
    int * rank);

/*
 * data types
 */

/*
 * ‘acgdatatype’ is a type for enumerating different data types to be
 * used in communication routines.
 */
enum acgdatatype
{
    ACG_DOUBLE
};

/**
 * ‘acgdatatypestr()’ returns a string for a data type.
 */
ACG_API const char * acgdatatypestr(enum acgdatatype datatype);

/**
 * ‘acgdatatype_size()’ returns the size (in bytes) of a data type.
 */
ACG_API int acgdatatype_size(enum acgdatatype datatype, int * size);

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgdatatype_mpi()’ returns a corresponding MPI_Datatype for the
 * given data type.
 */
ACG_API MPI_Datatype acgdatatype_mpi(enum acgdatatype datatype);
#endif

#if defined(ACG_HAVE_NCCL) || defined(ACG_HAVE_RCCL)
/**
 * ‘acgdatatype_nccl()’ returns a corresponding NCCL_Datatype for the
 * given data type.
 */
ACG_API ncclDataType_t acgdatatype_nccl(enum acgdatatype datatype);
#endif

/*
 * operations
 */

/*
 * ‘acgop’ is a type for enumerating different different operations
 * that can be used for reduction-type collective communication.
 */
enum acgop
{
    ACG_SUM
};

/**
 * ‘acgopstr()’ returns a string for an operation.
 */
ACG_API const char * acgopstr(enum acgop op);

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgop_mpi()’ returns a corresponding MPI_Op.
 */
ACG_API MPI_Op acgop_mpi(enum acgop op);
#endif

#if defined(ACG_HAVE_NCCL) || defined(ACG_HAVE_RCCL)
/**
 * ‘acgop_nccl()’ returns a corresponding ncclRedOp_t.
 */
ACG_API ncclRedOp_t acgop_nccl(enum acgop op);
#endif

/*
 * collective communication
 */

#ifdef ACG_HAVE_CUDA
/**
 * ‘acgcomm_barrier()’ performs barrier synchronisation.
 */
ACG_API int acgcomm_barrier(
    cudaStream_t stream,
    const struct acgcomm * comm,
    int * errcode);

/**
 * ‘acgcomm_allreduce()’ performs an all-reduce operation.
 */
ACG_API int acgcomm_allreduce(
    const void * sendbuf,
    void * recvbuf,
    int count,
    enum acgdatatype datatype,
    enum acgop op,
    cudaStream_t stream,
    const struct acgcomm * comm,
    int * errcode);
#endif

#ifdef ACG_HAVE_HIP
/**
 * ‘acgcomm_barrier_hip()’ performs barrier synchronisation.
 */
ACG_API int acgcomm_barrier_hip(
    hipStream_t stream,
    const struct acgcomm * comm,
    int * errcode);

/**
 * ‘acgcomm_allreduce_hip()’ performs an all-reduce operation.
 */
ACG_API int acgcomm_allreduce_hip(
    const void * sendbuf,
    void * recvbuf,
    int count,
    enum acgdatatype datatype,
    enum acgop op,
    hipStream_t stream,
    const struct acgcomm * comm,
    int * errcode);
#endif

/*
 * helper functions for NVSHMEM
 */

/**
 * ‘acgcomm_nvshmem_version()’ prints version information for NVSHMEM.
 */
ACG_API int acgcomm_nvshmem_version(
    FILE * f);

/**
 * ‘acgcomm_nvshmem_init()’ initialise NVSHMEM library.
 */
#if defined(ACG_HAVE_MPI)
ACG_API int acgcomm_nvshmem_init(
    MPI_Comm mpicomm,
    int root,
    int * errcode);

/**
 * ‘acgcomm_init_nvshmem()’ creates an NVSHMEM-type communicator from
 * a given MPI communicator.
 */
ACG_API int acgcomm_init_nvshmem(
    struct acgcomm * comm,
    MPI_Comm mpicomm,
    int * errcode);
#endif

/**
 * ‘acgcomm_free_nvshmem()’ frees resources associated with a communicator.
 */
ACG_API void acgcomm_free_nvshmem(
    struct acgcomm * comm);

/**
 * ‘acgcomm_size_nvshmem()’ size of a communicator (i.e., number of processes).
 */
ACG_API int acgcomm_size_nvshmem(
    const struct acgcomm * comm,
    int * commsize);

/**
 * ‘acgcomm_rank_nvshmem()’ rank of the current process in a communicator.
 */
ACG_API int acgcomm_rank_nvshmem(
    const struct acgcomm * comm,
    int * rank);

/**
 * ‘acgcomm_nvshmem_malloc()’ allocates storage on the symmetric heap
 * for use with NVSHMEM.
 */
ACG_API int acgcomm_nvshmem_malloc(
    void ** addr,
    size_t size,
    int * errcode);

/**
 * ‘acgcomm_nvshmem_calloc()’ allocates storage on the symmetric heap
 * for use with NVSHMEM.
 */
ACG_API int acgcomm_nvshmem_calloc(
    void ** ptr,
    size_t count,
    size_t size,
    int * errcode);

/**
 * ‘acgcomm_nvshmem_free()’ frees storage allocated for NVSHMEM.
 */
ACG_API void acgcomm_nvshmem_free(
    void * addr);

/**
 * ‘acgcomm_nvshmem_register_buffer()’ registers a buffer for use with
 * NVSHMEM.
 */
ACG_API int acgcomm_nvshmem_register_buffer(
    const struct acgcomm * comm,
    void * addr,
    size_t length,
    int * errcode);

/**
 * ‘acgcomm_nvshmem_unregister_buffer()’ unregisters a buffer for use
 * with NVSHMEM.
 */
ACG_API int acgcomm_nvshmem_unregister_buffer(
    void * addr,
    int * errcode);

#ifdef ACG_HAVE_CUDA
/**
 * ‘acgcomm_nvshmem_allreduce()’ performs an all-reduce operation on a
 * double precision floating point value.
 */
ACG_API int acgcomm_nvshmem_allreduce(
    const struct acgcomm * comm,
    double * dest,
    const double * source,
    int nreduce,
    cudaStream_t stream,
    int * errcode);
#endif

/**
 * ‘acgcomm_nvshmem_barrier_all()’ performs barrier synchronization.
 */
ACG_API int acgcomm_nvshmem_barrier_all(void);

/*
 * helper functions for ROCSHMEM
 */

/**
 * ‘acgcomm_rocshmem_version()’ prints version information for ROCSHMEM.
 */
ACG_API int acgcomm_rocshmem_version(
    FILE * f);

#if defined(ACG_HAVE_MPI)
/**
 * ‘acgcomm_rocshmem_init()’ initialise ROCSHMEM library.
 */
ACG_API int acgcomm_rocshmem_init(
    MPI_Comm mpicomm,
    int root,
    int * errcode);

/**
 * ‘acgcomm_init_rocshmem()’ creates an ROCSHMEM-type communicator from
 * a given MPI communicator.
 */
ACG_API int acgcomm_init_rocshmem(
    struct acgcomm * comm,
    MPI_Comm mpicomm,
    int * errcode);
#endif

/**
 * ‘acgcomm_free_rocshmem()’ frees resources associated with a communicator.
 */
ACG_API void acgcomm_free_rocshmem(
    struct acgcomm * comm);

/**
 * ‘acgcomm_size_rocshmem()’ size of a communicator (i.e., number of processes).
 */
ACG_API int acgcomm_size_rocshmem(
    const struct acgcomm * comm,
    int * commsize);

/**
 * ‘acgcomm_rank_rocshmem()’ rank of the current process in a communicator.
 */
ACG_API int acgcomm_rank_rocshmem(
    const struct acgcomm * comm,
    int * rank);

/**
 * ‘acgcomm_rocshmem_malloc()’ allocates storage on the symmetric heap
 * for use with ROCSHMEM.
 */
ACG_API int acgcomm_rocshmem_malloc(
    void ** addr,
    size_t size,
    int * errcode);

/**
 * ‘acgcomm_rocshmem_calloc()’ allocates storage on the symmetric heap
 * for use with ROCSHMEM.
 */
ACG_API int acgcomm_rocshmem_calloc(
    void ** ptr,
    size_t count,
    size_t size,
    int * errcode);

/**
 * ‘acgcomm_rocshmem_free()’ frees storage allocated for ROCSHMEM.
 */
ACG_API void acgcomm_rocshmem_free(
    void * addr);

/**
 * ‘acgcomm_rocshmem_register_buffer()’ registers a buffer for use with
 * ROCSHMEM.
 */
ACG_API int acgcomm_rocshmem_register_buffer(
    const struct acgcomm * comm,
    void * addr,
    size_t length,
    int * errcode);

/**
 * ‘acgcomm_rocshmem_unregister_buffer()’ unregisters a buffer for use
 * with ROCSHMEM.
 */
ACG_API int acgcomm_rocshmem_unregister_buffer(
    void * addr,
    int * errcode);

#ifdef ACG_HAVE_HIP
/**
 * ‘acgcomm_rocshmem_allreduce()’ performs an all-reduce operation on a
 * double precision floating point value.
 */
ACG_API int acgcomm_rocshmem_allreduce(
    const struct acgcomm * comm,
    double * dest,
    const double * source,
    int nreduce,
    hipStream_t stream,
    int * errcode);
#endif

/**
 * ‘acgcomm_rocshmem_barrier_all()’ performs barrier synchronization.
 */
ACG_API int acgcomm_rocshmem_barrier_all(void);

#ifdef __cplusplus
}
#endif

#endif
