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
 * double precision floating-point vectors
 */

#ifndef ACG_VECTOR_H
#define ACG_VECTOR_H

#include "acg/config.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ‘acgvector’ represents a vector stored as a contiguous array of
 * elements in full or packed storage format. The vector may also be
 * partitioned and distributed across multiple processes.
 *
 * The vector is represented by a contiguous array of elements. If the
 * vector is stored in packed format, then there is also an array of
 * integers designating the offset of each element.
 */
struct acgvector
{
    /*
     * partitioning information
     */

    /**
     * ‘nparts’ is the total number of partitions of the vector (i.e.,
     * the total number of partitioned subvectors).
     */
    int nparts;

    /**
     * ‘parttag’ is a tag that may be used to identify the subvector.
     */
    int parttag;

    /**
     * ‘nprocs’ is the number of processes among which the partitioned
     * vector is distributed.
     */
    int nprocs;

    /**
     * ‘npparts’ is the number of subvectors assigned to the current
     * process.
     */
    int npparts;

    /**
     * ‘ownerrank’ is the rank of the process that owns the subvector,
     * ranging from ‘0’ up to ‘nprocs-1’.
     */
    int ownerrank;

    /**
     * ‘ownerpart’ is the partition number of the subvector among the
     * subvectors assigned to the current process rank, ranging from
     * ‘0’ up to ‘npparts-1’.
     */
    int ownerpart;

    /*
     * vector elements
     */

    /**
     * ‘size’ is the number of vector elements.
     */
    acgidx_t size;

    /**
     * ‘x’ is an array containing the (nonzero) vector values, which
     * is of length ‘size’ if the vector is stored in full format
     * (i.e., ‘idx’ is ‘NULL’) and of length ‘num_nonzeros’ if the
     * vector is stored in packed format.
     */
    double * x;

    /*
     * auxiliary data for (sparse) vectors in packed storage format
     */

    /**
     * ‘num_nonzeros’ is the number of explicitly stored vector
     * entries (including ghost entries). For a vector in full storage
     * format, ‘num_nonzeros’ must be equal to ‘size’.
     */
    acgidx_t num_nonzeros;

    /**
     * ‘idxbase’ should be set to ‘0’ or ‘1’ if entries of the ‘idx’
     * array use 0-based or 1-based indexing, respectively.
     */
    int idxbase;

    /**
     * ‘idx’ is an array of length ‘num_nonzeros’, containing the
     * offset of each nonzero vector entry. Note that offsets are
     * 0-based, unlike the Matrix Market format, where indices are
     * 1-based.
     *
     * Note that ‘idx’ is set to ‘NULL’ for vectors in full storage
     * format. In this case, ‘size’ and ‘num_nonzeros’ must be equal,
     * and elements of the vector are implicitly numbered from ‘0’ up
     * to ‘size-1’.
     */
    acgidx_t * idx;

    /*
     * auxiliary data for partitioned and distributed vectors
     */

    /**
     * ‘num_ghost_nonzeros’ is the number of additional “ghost”
     * entries that are stored in the vector but are ignored during
     * certain operations (e.g., dot products). The ghost entries are
     * assumed to be placed last, after the regular vector entries.
     *
     * This is a commonly used technique to simplify working with
     * distributed-memory parallel vectors.
     */
    acgidx_t num_ghost_nonzeros;
};

/*
 * memory management
 */

/**
 * ‘acgvector_init_empty()’ initialises an empty vector.
 */
ACG_API void acgvector_init_empty(
    struct acgvector * x);

/**
 * ‘acgvector_free()’ frees storage allocated for a vector.
 */
ACG_API void acgvector_free(
    struct acgvector * x);

/**
 * ‘acgvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
ACG_API int acgvector_alloc_copy(
    struct acgvector * dst,
    const struct acgvector * src);

/**
 * ‘acgvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
ACG_API int acgvector_init_copy(
    struct acgvector * dst,
    const struct acgvector * src);

/*
 * initialise vectors in full storage format
 */

/**
 * ‘acgvector_alloc()’ allocates a vector.
 */
ACG_API int acgvector_alloc(
    struct acgvector * x,
    acgidx_t size);

/**
 * ‘acgvector_init_real_double()’ allocates and initialises a vector
 * with real, double precision coefficients.
 */
ACG_API int acgvector_init_real_double(
    struct acgvector * x,
    acgidx_t size,
    const double * data);

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘acgvector_alloc_packed()’ allocates a vector in packed storage
 * format.
 */
ACG_API int acgvector_alloc_packed(
    struct acgvector * x,
    acgidx_t size,
    acgidx_t num_nonzeros,
    int idxbase,
    const acgidx_t * idx);

/**
 * ‘acgvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
ACG_API int acgvector_init_packed_real_double(
    struct acgvector * x,
    acgidx_t size,
    acgidx_t num_nonzeros,
    int idxbase,
    const acgidx_t * idx,
    const double * data);

/*
 * modifying values
 */

/**
 * ‘acgvector_setzero()’ sets every value of a vector to zero.
 */
ACG_API int acgvector_setzero(
    struct acgvector * x);

/**
 * ‘acgvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
ACG_API int acgvector_set_constant_real_double(
    struct acgvector * x,
    double a);

/**
 * ‘acgvector_set_real_double()’ sets values of a vector based on an
 * array of double precision floating point numbers.
 */
ACG_API int acgvector_set_real_double(
    struct acgvector * x,
    acgidx_t size,
    const double * a,
    bool include_ghosts);

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘acgvector_fwrite()’ writes a Matrix Market file to a stream.
 *
 * If ‘comments’ is not ‘NULL’, then it must either be an empty string
 * or a string of one or more comment lines. Each comment line must
 * begin with '%' and end with a newline character, '\n'.
 *
 * If ‘fmt’ is ‘NULL’, then the format specifier ‘%g’ is used to print
 * floating point numbers with enough digits to ensure correct
 * round-trip conversion from decimal text and back.  Otherwise, the
 * given format string is used to print numerical values.
 *
 * The format string ‘fmt’ follows the conventions of ‘printf’. The
 * format specifiers '%e', '%E', '%f', '%F', '%g' or '%G' may be
 * used. Field width and precision may be specified (e.g., "%3.1f"),
 * but variable field width and precision (e.g., "%*.*f"), as well as
 * length modifiers (e.g., "%Lf") are not allowed.
 *
 * If it is not ‘NULL’, then the number of bytes written to the stream
 * is returned in ‘bytes_written’.
 *
 * The locale is temporarily changed to "C" to ensure that
 * locale-specific settings, such as the type of decimal point, do not
 * affect output.
 */
ACG_API int acgvector_fwrite(
    const struct acgvector * x,
    FILE * f,
    const char * comments,
    const char * fmt,
    int64_t * bytes_written);

/*
 * Level 1 BLAS operations
 */

/**
 * ‘acgvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
ACG_API int acgvector_swap(
    struct acgvector * x,
    struct acgvector * y,
    int64_t * num_bytes);

/**
 * ‘acgvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
ACG_API int acgvector_copy(
    struct acgvector * y,
    const struct acgvector * x,
    int64_t * num_bytes);

/**
 * ‘acgvector_dscal()’ scales a vector by a double precision floating
 * point scalar, ‘x = a*x’.
 */
ACG_API int acgvector_dscal(
    double a,
    struct acgvector * x,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_daxpy()’ adds a vector to another one multiplied by a
 * double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
ACG_API int acgvector_daxpy(
    double a,
    const struct acgvector * x,
    struct acgvector * y,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
ACG_API int acgvector_daypx(
    double a,
    struct acgvector * y,
    const struct acgvector * x,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must be the same size. Furthermore, both
 * vectors are treated as if they are in full storage format.
 */
ACG_API int acgvector_ddot(
    const struct acgvector * x,
    const struct acgvector * y,
    double * dot,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
ACG_API int acgvector_dnrm2(
    const struct acgvector * x,
    double * nrm2,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_dnrm2sqr()’ computes the square of the Euclidean norm
 * of a vector in double precision floating point.
 */
ACG_API int acgvector_dnrm2sqr(
    const struct acgvector * x,
    double * nrm2sqr,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_dasum()’ computes the sum of absolute values (1-norm)
 * of a vector in double precision floating point.
 */
ACG_API int acgvector_dasum(
    const struct acgvector * x,
    double * asum,
    int64_t * num_flops,
    int64_t * num_bytes);

/**
 * ‘acgvector_iamax()’ finds the index of the first element having
 * the maximum absolute value.
 */
ACG_API int acgvector_iamax(
    const struct acgvector * x,
    int * iamax);

/*
 * Level 1 Sparse BLAS operations.
 *
 * See I. Duff, M. Heroux and R. Pozo, “An Overview of the Sparse
 * Basic Linear Algebra Subprograms: The New Standard from the BLAS
 * Technical Forum,” ACM TOMS, Vol. 28, No. 2, June 2002, pp. 239-267.
 */

/**
 * ‘acgvector_usddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size. The vector ‘x’ is
 * a sparse vector in packed form. Repeated indices in the packed
 * vector are not allowed, otherwise the result is undefined.
 */
ACG_API int acgvector_usddot(
    const struct acgvector * x,
    const struct acgvector * y,
    double * dot,
    int64_t * num_flops);

/**
 * ‘acgvector_usdaxpy()’ performs a sparse vector update, multiplying
 * a sparse vector ‘x’ in packed form by a scalar ‘alpha’ and adding
 * the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size size. Repeated
 * indices in the packed vector are allowed.
 */
ACG_API int acgvector_usdaxpy(
    double alpha,
    const struct acgvector * x,
    struct acgvector * y,
    int64_t * num_flops);

/**
 * ‘acgvector_usga()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are allowed.
 */
ACG_API int acgvector_usga(
    struct acgvector * x,
    const struct acgvector * y);

/**
 * ‘acgvector_usgz()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form, while zeroing the values
 * of the source vector ‘y’ that were copied to ‘x’. Repeated indices
 * in the packed vector are allowed.
 */
ACG_API int acgvector_usgz(
    struct acgvector * x,
    struct acgvector * y);

/**
 * ‘acgvector_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
ACG_API int acgvector_ussc(
    struct acgvector * y,
    const struct acgvector * x);

/*
 * distributed-memory level 1 BLAS operations with MPI
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgvector_ddotmpi()’ computes the Euclidean dot product of two
 * distributed vectors in double precision floating point.
 *
 * On each process, the local vectors ‘x’ and ‘y’ must be the same
 * size. Furthermore, both vectors are treated as if they are in full
 * storage format.
 *
 * The distributed vector is assumed to be partitioned among processes
 * in the MPI communicator ‘comm’, meaning that every vector element
 * must belong to exactly one process.
 */
ACG_API int acgvector_ddotmpi(
    const struct acgvector * x,
    const struct acgvector * y,
    double * dot,
    int64_t * num_flops,
    int64_t * num_bytes,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgvector_dnrm2mpi()’ computes the Euclidean norm of a
 * distributed vector in double precision floating point.
 *
 * The distributed vector is assumed to be partitioned among processes
 * in the MPI communicator ‘comm’, meaning that every vector element
 * must belong to exactly one process.
 */
ACG_API int acgvector_dnrm2mpi(
    const struct acgvector * x,
    double * nrm2,
    int64_t * num_flops,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgvector_dasummpi()’ computes the sum of absolute values
 * (1-norm) of a distributed vector in double precision floating
 * point.
 *
 * The distributed vector is assumed to be partitioned among processes
 * in the MPI communicator ‘comm’, meaning that every vector element
 * must belong to exactly one process.
 */
ACG_API int acgvector_dasummpi(
    const struct acgvector * x,
    double * asum,
    int64_t * num_flops,
    MPI_Comm comm,
    int * mpierrcode);
#endif

/*
 * MPI functions
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgvector_send()’ sends a vector to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘acgvector_recv()’.
 */
ACG_API int acgvector_send(
    const struct acgvector * x,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgvector_recv()’ receives a vector from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘acgvector_send()’.
 */
ACG_API int acgvector_recv(
    struct acgvector * x,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgvector_irecv()’ performs a non-blocking receive of a vector
 * from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘acgvector_send()’.
 */
ACG_API int acgvector_irecv(
    struct acgvector * x,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Request * request,
    int * mpierrcode);

/**
 * ‘acgvector_scatter()’ dispatches vectors from a root process to
 * all processes in a communicator.
 *
 * This is analogous to ‘MPI_scatter()’, which is a collective
 * operation, and therefore requires all processes in the communicator
 * to invoke this routine.
 */
ACG_API int acgvector_scatter(
    struct acgvector * sendvectors,
    int sendcount,
    struct acgvector * recvvectors,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode);
#endif

#ifdef __cplusplus
}
#endif

#endif
