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

// #pragma STDC FENV_ACCESS on

#include "acg/config.h"
#include "acg/error.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif
#ifdef ACG_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif
#ifdef ACG_HAVE_NCCL
#include <nccl.h>
#endif
#ifdef ACG_HAVE_RCCL
#include <rccl/rccl.h>
#endif

#include <errno.h>

#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘fexcept_str()’ converts floating-point exceptions to a string.
 */
static const char * fexceptstr(int except) {
    if ((except & FE_DIVBYZERO) && (except & FE_INEXACT) && (except & FE_INVALID) && (except & FE_OVERFLOW) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,inexact,invalid,overflow,underflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INEXACT) && (except & FE_INVALID) && (except & FE_OVERFLOW)) {
        return "divide-by-zero,inexact,invalid,overflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INEXACT) && (except & FE_INVALID) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,inexact,invalid,underflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INVALID) && (except & FE_OVERFLOW) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,invalid,overflow,underflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INEXACT) && (except & FE_INVALID)) {
        return "divide-by-zero,inexact,invalid";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INEXACT) && (except & FE_OVERFLOW)) {
        return "divide-by-zero,inexact,overflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INEXACT) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,inexact,underflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INVALID) && (except & FE_OVERFLOW)) {
        return "divide-by-zero,invalid,overflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INVALID) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,invalid,underflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_OVERFLOW) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,overflow,underflow";
    } else if ((except & FE_INEXACT) && (except & FE_INVALID) && (except & FE_OVERFLOW)) {
        return "inexact,invalid,overflow";
    } else if ((except & FE_INEXACT) && (except & FE_INVALID) && (except & FE_UNDERFLOW)) {
        return "inexact,invalid,underflow";
    } else if ((except & FE_INEXACT) && (except & FE_OVERFLOW) && (except & FE_UNDERFLOW)) {
        return "inexact,overflow,underflow";
    } else if ((except & FE_INVALID) && (except & FE_OVERFLOW) && (except & FE_UNDERFLOW)) {
        return "invalid,overflow,underflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INEXACT)) {
        return "divide-by-zero,inexact";
    } else if ((except & FE_DIVBYZERO) && (except & FE_INVALID)) {
        return "divide-by-zero,invalid";
    } else if ((except & FE_DIVBYZERO) && (except & FE_OVERFLOW)) {
        return "divide-by-zero,overflow";
    } else if ((except & FE_DIVBYZERO) && (except & FE_UNDERFLOW)) {
        return "divide-by-zero,underflow";
    } else if ((except & FE_INEXACT) && (except & FE_INVALID)) {
        return "inexact,invalid";
    } else if ((except & FE_INEXACT) && (except & FE_OVERFLOW)) {
        return "inexact,overflow";
    } else if ((except & FE_INEXACT) && (except & FE_UNDERFLOW)) {
        return "inexact,underflow";
    } else if ((except & FE_INVALID) && (except & FE_OVERFLOW)) {
        return "invalid,overflow";
    } else if ((except & FE_INVALID) && (except & FE_UNDERFLOW)) {
        return "invalid,underflow";
    } else if ((except & FE_OVERFLOW) && (except & FE_UNDERFLOW)) {
        return "overflow,underflow";
    } else if (except & FE_DIVBYZERO) { return "divide-by-zero";
    } else if (except & FE_INEXACT) { return "inexact";
    } else if (except & FE_INVALID) { return "invalid";
    } else if (except & FE_OVERFLOW) { return "overflow";
    } else if (except & FE_UNDERFLOW) { return "underflow";
    } else { return "none"; }
}

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
    int mpierrcode)
{
    switch (err) {
    case ACG_SUCCESS: return "success";

    /* system call or external library errors */
    case ACG_ERR_ERRNO: return strerror(errno);
    case ACG_ERR_FEXCEPT: return fexceptstr(fetestexcept(FE_ALL_EXCEPT));
    case ACG_ERR_MPI:
        {
#ifdef ACG_HAVE_MPI
            static char mpierrstr[MPI_MAX_ERROR_STRING];
            int mpierrstrlen = MPI_MAX_ERROR_STRING;
            MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
            return mpierrstr;
#else
            return "unknown MPI error";
#endif
        }
    case ACG_ERR_CUDA:
        {
#ifdef ACG_HAVE_CUDA
            int cudaerr = cudaGetLastError();
            return cudaGetErrorString(cudaerr);
#else
            return "unknown CUDA error";
#endif
        }
    case ACG_ERR_NCCL:
        {
#ifdef ACG_HAVE_NCCL
            return ncclGetErrorString(mpierrcode);
#else
            return "unknown NCCL error";
#endif
        }
    case ACG_ERR_NVSHMEM:
        {
#ifdef ACG_HAVE_NVSHMEM
            /* NVSHMEM does not seem to have any description of its error codes */
            return "unknown NVSHMEM error";
#else
            return "unknown NVSHMEM error";
#endif
        }
    case ACG_ERR_CUBLAS:
        {
#ifdef ACG_HAVE_CUBLAS
            return "unknown cuBLAS error";
            /* return cublasGetStatusString(cublasstatus); */
#else
            return "unknown cuBLAS error";
#endif
        }
    case ACG_ERR_CUSPARSE:
        {
#ifdef ACG_HAVE_CUSPARSE
            return "unknown cuSPARSE error";
            /* return cusparseGetErrorString(cusparsestatus); */
#else
            return "unknown cuSPARSE error";
#endif
        }
    case ACG_ERR_HIP:
        {
#ifdef ACG_HAVE_HIP
            int hiperr = hipGetLastError();
            return hipGetErrorString(hiperr);
#else
            return "unknown HIP error";
#endif
        }
    case ACG_ERR_HIPBLAS:
        {
#ifdef ACG_HAVE_HIPBLAS
            return "unknown hipBLAS error";
            /* return hipblasGetStatusString(hipblasstatus); */
#else
            return "unknown hipBLAS error";
#endif
        }
    case ACG_ERR_HIPSPARSE:
        {
#ifdef ACG_HAVE_HIPSPARSE
            return "unknown hipSPARSE error";
            /* return hipsparseGetErrorString(hipsparsestatus); */
#else
            return "unknown hipSPARSE error";
#endif
        }
    case ACG_ERR_RCCL:
        {
#ifdef ACG_HAVE_RCCL
            return ncclGetErrorString(mpierrcode);
#else
            return "unknown RCCL error";
#endif
        }
    case ACG_ERR_ROCSHMEM:
        {
#ifdef ACG_HAVE_ROCSHMEM
            /* does ROCSHMEM have any description of its error codes? */
            return "unknown ROCSHMEM error";
#else
            return "unknown ROCSHMEM error";
#endif
        }
    case ACG_ERR_MPI_NOT_SUPPORTED: return "MPI is disabled; please rebuild with MPI support";
    case ACG_ERR_NCCL_NOT_SUPPORTED: return "NCCL is disabled; please rebuild with NCCL support";
    case ACG_ERR_NVSHMEM_NOT_SUPPORTED: return "NVSHMEM is disabled; please rebuild with NVSHMEM support";
    case ACG_ERR_RCCL_NOT_SUPPORTED: return "RCCL is disabled; please rebuild with RCCL support";
    case ACG_ERR_ROCSHMEM_NOT_SUPPORTED: return "ROCSHMEM is disabled; please rebuild with ROCSHMEM support";
    case ACG_ERR_METIS_NOT_SUPPORTED: return "METIS is disabled; please rebuild with METIS support";
    case ACG_ERR_LIBZ_NOT_SUPPORTED: return "zlib is disabled; please rebuild with zlib support";

    /* METIS errors */
    case ACG_ERR_METIS_INPUT: return "METIS: input error";
    case ACG_ERR_METIS_MEMORY: return "METIS: cannot allocate memory";
    case ACG_ERR_METIS: return "METIS: error";
    case ACG_ERR_METIS_EOVERFLOW:
#if ACG_HAVE_METIS && IDXTYPEWIDTH < 64
        return "METIS: value too large for defined data type; "
            "please rebuild METIS with support for 64-bit integer types";
#else
        return "METIS: value too large for defined data type";
#endif

    /* generic errors */
    case ACG_ERR_NOT_SUPPORTED: return "not supported";
    case ACG_ERR_EOF: return errno ? strerror(errno) : "unexpected end-of-file";
    case ACG_ERR_LINE_TOO_LONG: return "maximum line length exceeded";
    case ACG_ERR_INVALID_VALUE: return "invalid value";
    case ACG_ERR_OVERFLOW: return "value too large to be stored in data type";
    case ACG_ERR_INDEX_OUT_OF_BOUNDS: return "index out of bounds";
    case ACG_ERR_NO_BUFFER_SPACE: return "not enough space in buffer";

    /* Matrix Market I/O errors */
    case ACG_ERR_MTX_INVALID_COMMENT: return "invalid comment line";
    case ACG_ERR_INVALID_FORMAT_SPECIFIER: return "invalid format specifier";

    /* vector-related errors */
    case ACG_ERR_VECTOR_INCOMPATIBLE_SIZE: return "incompatible vector size";
    case ACG_ERR_VECTOR_INCOMPATIBLE_FORMAT: return "incompatible vector format";
    case ACG_ERR_VECTOR_EXPECTED_FULL: return "expected vector in full storage format";
    case ACG_ERR_VECTOR_EXPECTED_PACKED: return "expected vector in packed storage format";

    /* iterative solver errors */
    case ACG_ERR_NOT_CONVERGED: return "not converged";
    case ACG_ERR_NOT_CONVERGED_INDEFINITE_MATRIX: return "not converged (indefinite matrix)";

    default: return "unknown error";
    }
}

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
    int * outrank,
    int * outerrnocode,
    int * outmpierrcode)
{
    int rank;
    int mpierrcode = MPI_Comm_rank(comm, &rank);
    if (mpierrcode) { *outmpierrcode = mpierrcode; return ACG_ERR_MPI; }
    /* if (mpierrcode) { MPI_Abort(comm, mpierrcode); } */

#ifdef ACG_ABORT_ON_ERROR
    if (err) {
        fprintf(stderr, "%s\n",
                acgerrcodestr(err, outmpierrcode ? *outmpierrcode : 0));
        MPI_Abort(comm, err);
    }
#else
    /* find the first rank with a nonzero error code */
    int have_error[2] = { err ? 1 : 0, rank };
    mpierrcode = MPI_Allreduce(MPI_IN_PLACE, &have_error, 1, MPI_2INT, MPI_MAXLOC, comm);
    if (mpierrcode) { *outmpierrcode = mpierrcode; return ACG_ERR_MPI; }
    /* if (mpierrcode) { MPI_Abort(comm, mpierrcode); } */

    /* return if there are no errors */
    if (!have_error[0]) return ACG_SUCCESS;
    int errrank = have_error[1];

    /* broadcast the error from the lowest ranking process with a
     * nonzero error code */
    int buf[3] = {
        rank == errrank ? err : 0,
        rank == errrank && outerrnocode ? *outerrnocode : 0,
        rank == errrank && outmpierrcode ? *outmpierrcode : 0};
    mpierrcode = MPI_Bcast(buf, 3, MPI_INT, errrank, comm);
    if (mpierrcode) { *outmpierrcode = mpierrcode; return ACG_ERR_MPI; }
    /* if (mpierrcode) { MPI_Abort(comm, mpierrcode); } */

    err = buf[0];
    if (outrank) *outrank = errrank;
    if (outerrnocode) *outerrnocode = buf[1];
    if (outmpierrcode) *outmpierrcode = buf[2];
#endif
    return err;
}
#endif
