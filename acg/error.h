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
 * Error handling
 */

#ifndef ACG_ERROR_H
#define ACG_ERROR_H

#include "acg/config.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ‘acgerrcode’ is a type for enumerating different error codes that
 * are used for error handling.
 */
enum acgerrcode
{
    ACG_SUCCESS = 0,                         /* no error */

    /* system call or external library errors */
    ACG_ERR_ERRNO,                           /* error code provided by errno */
    ACG_ERR_FEXCEPT,                         /* error code provided by fetestexcept() */
    ACG_ERR_MPI,                             /* error code provided by MPI */
    ACG_ERR_CUDA,                            /* error code provided by CUDA */
    ACG_ERR_NCCL,                            /* error code provided by NCCL */
    ACG_ERR_NVSHMEM,                         /* error code provided by NVSHMEM */
    ACG_ERR_CUBLAS,                          /* error code provided by cuBLAS */
    ACG_ERR_CUSPARSE,                        /* error code provided by cuSPARSE */
    ACG_ERR_HIP,                             /* error code provided by HIP */
    ACG_ERR_RCCL,                            /* error code provided by RCCL */
    ACG_ERR_ROCSHMEM,                        /* error code provided by ROCSHMEM */
    ACG_ERR_HIPBLAS,                         /* error code provided by hipBLAS */
    ACG_ERR_HIPSPARSE,                       /* error code provided by hipsparse */
    ACG_ERR_MPI_NOT_SUPPORTED,               /* MPI not supported */
    ACG_ERR_NCCL_NOT_SUPPORTED,              /* NCCL not supported */
    ACG_ERR_NVSHMEM_NOT_SUPPORTED,           /* NVSHMEM not supported */
    ACG_ERR_RCCL_NOT_SUPPORTED,              /* RCCL not supported */
    ACG_ERR_ROCSHMEM_NOT_SUPPORTED,          /* ROCSHMEM not supported */
    ACG_ERR_METIS_NOT_SUPPORTED,             /* METIS not supported */
    ACG_ERR_PETSC_NOT_SUPPORTED,             /* PETSc not supported */
    ACG_ERR_LIBZ_NOT_SUPPORTED,              /* zlib not supported */

    /* METIS errors */
    ACG_ERR_METIS_INPUT,                     /* METIS: input error */
    ACG_ERR_METIS_MEMORY,                    /* METIS: cannot allocate memory */
    ACG_ERR_METIS,                           /* METIS: error */
    ACG_ERR_METIS_EOVERFLOW,                 /* METIS: value too large for defined data type */

    /* generic errors */
    ACG_ERR_NOT_SUPPORTED,                   /* not supported */
    ACG_ERR_EOF,                             /* unexpected end-of-file */
    ACG_ERR_LINE_TOO_LONG,                   /* line exceeds maximum length */
    ACG_ERR_INVALID_VALUE,                   /* invalid value */
    ACG_ERR_OVERFLOW,                        /* value too large to be stored in data type */
    ACG_ERR_INDEX_OUT_OF_BOUNDS,             /* index out of bounds */
    ACG_ERR_NO_BUFFER_SPACE,                 /* not enough space in buffer */

    /* Matrix Market I/O errors */
    ACG_ERR_MTX_INVALID_COMMENT,             /* invalid comment line */
    ACG_ERR_INVALID_FORMAT_SPECIFIER,        /* invalid format specifier */

    /* vector-related errors */
    ACG_ERR_VECTOR_INCOMPATIBLE_SIZE,        /* incompatible vector size */
    ACG_ERR_VECTOR_INCOMPATIBLE_FORMAT,      /* incompatible vector format */
    ACG_ERR_VECTOR_EXPECTED_FULL,            /* expected vector in full storage format */
    ACG_ERR_VECTOR_EXPECTED_PACKED,          /* expected vector in packed storage format */

    /* iterative solver errors */
    ACG_ERR_NOT_CONVERGED,                   /* not converged */
    ACG_ERR_NOT_CONVERGED_INDEFINITE_MATRIX, /* not converged (indefinite matrix) */
};

/**
 * ‘acgerrcodestr()’ is a string describing an error code.
 *
 * The error code ‘err’ must correspond to one of the error codes
 * defined in the ‘acgerrcode’ enum type.
 *
 * If ‘err’ is ‘ACG_ERR_ERRNO’, then ‘acgerrcodestr()’ will use the
 * value of ‘errno’ to obtain a description of the error.
 *
 * If ‘err’ is ‘ACG_ERR_MPI’, then ‘acgerrcodestr()’ will use the
 * value of ‘mpierrcode’ to obtain a description of the error.
 */
const char * acgerrcodestr(
    int err,
    int mpierrcode);

#ifdef ACG_HAVE_MPI
/**
 * ‘acgerrmpi()’ checks if any of the MPI processes in a given
 * communicator encountered an error.
 *
 * This is needed to perform proper error handling, recovery or a
 * graceful exit in cases where one or more MPI processes encounter
 * errors. If this is not handled carefully, some processes are
 * typically left hanging indefinitely in a communication or
 * synchronisation call.
 *
 * This function performs collective communication, and must therefore
 * be called by all processes in the given communicator. More
 * specifically, an all-reduce operation is performed on the error
 * code ‘err’.
 *
 * The return value is ‘ACG_SUCCESS’ if the value of ‘err’ is set to
 * ‘ACG_SUCCESS’ on every process. Otherwise, a nonzero error code is
 * returned corresponding to the value of ‘err’ on the lowest ranking
 * MPI process with a nonzero ‘err’ value.
 *
 * If ‘rank’ is not ‘NULL’, then it is used to store the rank of the
 * lowest ranking MPI process with a nonzero error code. Similarly, if
 * ‘errnocode’ or ‘mpierrcode’ are not ‘NULL’, then they are used to
 * store the corresponding values from the lowest ranking MPI process
 * with a nonzero error code.
 */
int acgerrmpi(
    MPI_Comm comm,
    int err,
    int * rank,
    int * errnocode,
    int * mpierrcode);
#endif

#ifdef __cplusplus
}
#endif

#endif
