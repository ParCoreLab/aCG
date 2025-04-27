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
 * symmetric matrices in compressed sparse row (CSR) format
 */

#ifndef ACG_SYMCSRMATRIX_H
#define ACG_SYMCSRMATRIX_H

#include "acg/config.h"
#include "acg/metis.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

struct acggraph;
struct acgvector;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ‘acgsymcsrmatrix’ is a data structure for square, symmetric sparse
 * matrices stored in compressed sparse row (CSR) format.
 *
 * The storage format consists of three separate arrays: row pointers,
 * column indices and nonzero values. The data structure also provides
 * some auxiliary data and arrays to facilitate distribution of a
 * matrix across multiple processes.
 */
struct acgsymcsrmatrix
{
    /*
     * matrix sparsity pattern, partitioning and distribution
     */

    /**
     * ‘graph’ is the undirected graph that represents the sparsity
     * pattern of the symmetric matrix. It also contains information
     * about the partitioning and distribution of the matrix.
     */
    struct acggraph * graph;

    /*
     * matrix/submatrix rows
     */

    /**
     * ‘nrows’ is the total number of rows/columns in the matrix.
     */
    acgidx_t nrows;

    /**
     * ‘nprows’ is the number of nonzero rows/columns in the
     * submatrix.
     */
    acgidx_t nprows;

    /**
     * ‘nzrows’ is an array of length ‘nprows’, containing the row
     * number of each nonzero matrix row.  Partitioning the matrix
     * causes the rows/columns to be reordered, this array is used to
     * store the original row/column numbers of the non-empty
     * submatrix rows/columns prior to partitioning.
     */
    const acgidx_t * nzrows;

    /*
     * matrix/submatrix nonzeros
     */

    /**
     * ‘nnzs’ is the total number of nonzeros in the matrix.
     */
    int64_t nnzs;

    /**
     * ‘npnzs’ is the number of nonzeros in the submatrix.
     */
    int64_t npnzs;

    /*
     * incidence relation mapping nonzeros to unordered pairs of
     * rows/columns
     */

    /**
     * ‘rowidxbase’ is the base (0 or 1) used for numbering
     * rows/columns in the incidence relation. More specifically, the
     * row/column numbers stored in ‘dstcolidx’ range from
     * ‘rowidxbase’ up to ‘nprows+rowidxbase-1’.
     */
    int rowidxbase;

    /**
     * ‘rownnzs’ is an array of length ‘nprows’ containing the number
     * of nonzeros for each row/column.
     */
    const int64_t * rownnzs;

    /**
     * ‘rowptr’ is an array of length ‘nprows+1’ containing the
     * offsets to the first nonzero of each row. More specifically, if
     * the i-th row has one or more nonzeros, ‘aᵢⱼ₍₁₎’, ‘aᵢⱼ₍₂₎’, ...,
     * then the column number, ‘j(k)’, of the k-th nonzero is stored
     * at ‘colidx[rowptr[i]+k]’.
     */
    const int64_t * rowptr;

    /**
     * ‘rowidx’ is an array of length ‘npnzs’ containing the row
     * number of the row ‘i’ in the pair ‘(i,j)’ for every nonzero of
     * the submatrix.
     */
    const acgidx_t * rowidx;

    /**
     * ‘colidx’ is an array of length ‘npnzs’ containing the column
     * number of the column ‘j’ in the pair ‘(i,j)’ for every nonzero
     * of the submatrix.
     */
    const acgidx_t * colidx;

    /*
     * interior, border and ghost rows/columns
     */

    /**
     * ‘nownedrows’ is the number of rows/columns in the submatrix
     * that are owned by the submatrix according to the row/column
     * partitioning, which is equal to the number of interior and
     * border rows/columns in the submatrix.
     */
    acgidx_t nownedrows;

    /**
     * ‘ninnerrows’ is the number of interior rows/columns in the
     * submatrix, meaning that neighbourhing rows/columns are owned by
     * submatrix.
     */
    acgidx_t ninnerrows;

    /**
     * ‘nborderrows’ is the number of border rows/columns in the
     * submatrix, meaning that the rows/columns are owned by the
     * submatrix, but their neighbourhood contains one or more
     * rows/columns that are owned by another submatrix of the
     * partitioned matrix.
     */
    acgidx_t nborderrows;

    /**
     * ‘borderrowoffset’ is an offset to the first border row/column
     * according to the numbering of rows/columns in the submatrix,
     * where interior rows/columns are grouped before border
     * rows/columns, which again are ordered before ghost
     * rows/columns.
     */
    acgidx_t borderrowoffset;

    /**
     * ‘nghostrows’ is the number of ghost rows/columns in the
     * submatrix, meaning that the rows/columns are not owned by the
     * submatrix, but they belong to the neighbourhood of one or more
     * (border) rows/columns owned by the submatrix.
     */
    acgidx_t nghostrows;

    /**
     * ‘ghostrowoffset’ is an offset to the first ghost row/column
     * according to the numbering of rows/columns in the submatrix,
     * where interior rows/columns are grouped before ghost
     * rows/columns, which again are ordered before ghost
     * rows/columns.
     */
    acgidx_t ghostrowoffset;

    /* interior and interface nonzeros */

    /**
     * ‘ninnernzs’ is the number of interior nonzeros in the
     * submatrix, such that the row and column of the nonzero are
     * owned by the submatrix.
     */
    int64_t ninnernzs;

    /**
     * ‘nneighournzs’ is the number of border nonzeros in the
     * submatrix, such that its row or column is owned by a
     * neighbouring submatrix.
     */
    int64_t ninterfacenzs;

    /**
     * ‘nborderrowinnernzs’ is an array of length ‘nborderrows’
     * containing the number of interior nonzeros for every border
     * row/column in the submatrix.
     */
    const int64_t * nborderrowinnernzs;

    /**
     * ‘nborderrowinterfacenzs’ is an array of length ‘nborderrows’
     * containing the number of interface nonzeros for every border
     * row/column in the submatrix.
     */
    const int64_t * nborderrowinterfacenzs;

    /*
     * matrix/submatrix nonzero values
     */

    /**
     * ‘a’ is an array of length ‘npnzs’, which contains the nonzero
     * matrix values.
     */
    double * a;

    /*
     * Full storage format - by default, the matrix is stored in a
     * packed storage format, which only stores the upper triangle of
     * the symmetric sparse matrix in a compressed row format.
     *
     * In some cases, it is desirable to store the matrix in a "full"
     * storage format, where both upper and lower triangular elements
     * are combined in the same compressed row storage. In particular,
     * sparse matrix-vector multiplication can then be performed
     * rowwise in parallel without data conflicts between threads.
     *
     * If the full storage format is not used, then ‘frowptr’,
     * ‘fcolidx’ and ‘fa’ may be set to ‘NULL’.
     */

    /**
     * ‘fnpnzs’ is the number of nonzeros in the submatrix when it is
     * stored in full storage format.
     */
    int64_t fnpnzs, onpnzs;

    /**
     * ‘frowptr’ is an array of length ‘nprows+1’ containing the
     * offsets to the first nonzero of each row of the matrix when it
     * is stored in full storage format. More specifically, if the
     * i-th row has one or more nonzeros, ‘aᵢⱼ₍₁₎’, ‘aᵢⱼ₍₂₎’, ...,
     * then the column number, ‘j(k)’, of the k-th nonzero is stored
     * at ‘fcolidx[frowptr[i]+k]’.
     */
    int64_t * frowptr, * orowptr;

    /**
     * ‘fcolidx’ is an array of length ‘fnpnzs’ containing the column
     * number of the column ‘j’ in the pair ‘(i,j)’ for every nonzero
     * of the submatrix when it is stored in full storage format.
     */
    acgidx_t * fcolidx, * ocolidx;

    /**
     * ‘fa’ is an array of length ‘fnpnzs’, which contains the nonzero
     * matrix values of the matrix in full storage format.
     */
    double * fa, * oa;
};

/**
 * ‘acgsymcsrmatrix_init_real_double()’ allocates and initialises a
 * matrix from nonzeros provided in coordinate format with real,
 * double precision coefficients.
 *
 * Only upper triangular entries of A should be provided.
 */
ACG_API int acgsymcsrmatrix_init_real_double(
    struct acgsymcsrmatrix * A,
    acgidx_t N,
    int64_t nnzs,
    int idxbase,
    const acgidx_t * rowidx,
    const acgidx_t * colidx,
    const double * a);

/**
 * ‘acgsymcsrmatrix_init_rowwise_real_double()’ allocates and
 * initialises a matrix in compressed row format with real, double
 * precision coefficients.
 *
 * Only upper triangular entries of A should be provided.
 */
ACG_API int acgsymcsrmatrix_init_rowwise_real_double(
    struct acgsymcsrmatrix * A,
    acgidx_t N,
    int idxbase,
    const int64_t * rowptr,
    const acgidx_t * colidx,
    const double * a);

/**
 * ‘acgsymcsrmatrix_free()’ frees storage allocated for a matrix.
 */
ACG_API void acgsymcsrmatrix_free(
    struct acgsymcsrmatrix * A);

/**
 * ‘acgsymcsrmatrix_copy()’ copies a matrix.
 */
ACG_API int acgsymcsrmatrix_copy(
    struct acgsymcsrmatrix * dst,
    const struct acgsymcsrmatrix * src);

/*
 * modifying values
 */

/**
 * ‘acgsymcsrmatrix_setzero()’ sets every value of a matrix to zero.
 */
ACG_API int acgsymcsrmatrix_setzero(
    struct acgsymcsrmatrix * A);

/*
 * output to Matrix Market format
 */

/**
 * ‘acgsymcsrmatrix_fwrite()’ writes a Matrix Market file to a stream.
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
ACG_API int acgsymcsrmatrix_fwrite(
    const struct acgsymcsrmatrix * A,
    FILE * f,
    const char * comments,
    const char * fmt,
    int64_t * bytes_written);

/*
 * vectors
 */

/**
 * ‘acgsymcsrmatrix_vector()’ allocates a vector of length N for a
 * given matrix of dimensions N-by-N.
 *
 * The allocated vector, which is of the same length as a single
 * row/column of the matrix, represents a vector in the column/row
 * space of the linear map associated with the matrix. As a column
 * vector, (i.e., arranged as N-by-1 values), it may be used for right
 * multiplication with an M-by-N matrix, such that the result is an
 * M-by-1 column vector in the range of the linear map associated with
 * the matrix.
 *
 * If the matrix is partitioned and distributed, then the vector will
 * also be partitioned accordingly. As a result, the vector can be
 * used for right multiplication with the given partitioned and
 * distributed matrix.
 */
ACG_API int acgsymcsrmatrix_vector(
    const struct acgsymcsrmatrix * A,
    struct acgvector * x);

/*
 * matrix partitioning
 */

/**
 * ‘acgsymcsrmatrix_partition_rows()’ partitions the rows of a
 * symmetric matrix into the given number of parts.
 *
 * The rows are partitioned using the METIS graph partitioner.
 */
ACG_API int acgsymcsrmatrix_partition_rows(
    struct acgsymcsrmatrix * A,
    int nparts,
    enum metis_partitioner partitioner,
    int * rowparts,
    acgidx_t * objval,
    acgidx_t seed,
    int verbose);

/**
 * ‘acgsymcsrmatrix_partition()’ partitions a matrix into a number of
 * submatrices based on a given partitioning of the matrix rows.
 *
 * ‘submatrices’ must point to an array of length ‘nparts’, which is
 * used to store the partitioned submatrices.
 */
ACG_API int acgsymcsrmatrix_partition(
    const struct acgsymcsrmatrix * A,
    int nparts,
    const int * rowparts,
    struct acgsymcsrmatrix * submatrices,
    int verbose);

/**
 * ‘acgsymcsrmatrix_partition_vector()’ partitions a vector into a
 * number of sparse vectors that conform to the partitioning of the
 * rows/columns of the given symmetric matrix.
 */
ACG_API int acgsymcsrmatrix_partition_vector(
    const struct acgsymcsrmatrix * A,
    const struct acgvector * x,
    struct acgvector * subvectors);

/*
 * matrix distribution
 */

#ifdef ACG_HAVE_MPI
ACG_API int acgsymcsrmatrix_send(
    const struct acgsymcsrmatrix * matrices,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

ACG_API int acgsymcsrmatrix_recv(
    struct acgsymcsrmatrix * matrices,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

ACG_API int acgsymcsrmatrix_scatter(
    struct acgsymcsrmatrix * sendmatrices,
    int sendcount,
    struct acgsymcsrmatrix * recvmatrices,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode);
#endif

/*
 * halo exchange/update for partitioned and distributed matrices
 */

struct acghalo;

/**
 * ‘acgsymcsrmatrix_halo()’ sets up a halo exchange communication
 * pattern to send and receive data associated with the “ghost”
 * rows/columns of partitioned and distributed vectors associated with
 * the row/column space of the given matrix.
 */
ACG_API int acgsymcsrmatrix_halo(
    const struct acgsymcsrmatrix * A,
    struct acghalo * halo);

/*
 * Level 2 BLAS operations (matrix-vector)
 */

ACG_API int acgsymcsrmatrix_dsymv_init(
    struct acgsymcsrmatrix * A,
    double eps);

/**
 * ‘acgsymcsrmatrix_dsymv()’ multiplies a matrix ‘A’ by a real scalar
 * ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to another
 * vector ‘y’ multiplied by another scalar real ‘beta’ (‘β’).  That
 * is, ‘y = α*A*x + β*y’ or ‘y = α*A'*x + β*y’.
 *
 * The scalars ‘alpha’ and ‘beta’ are given as double precision
 * floating point numbers.
 *
 * The vectors ‘x’ and ‘y’ must be provided in full storage (i.e., not
 * packed).
 */
ACG_API int acgsymcsrmatrix_dsymv(
    double alpha,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * x,
    double beta,
    struct acgvector * y,
    int64_t * num_flops,
    int64_t * num_bytes);

/*
 * distributed level 2 BLAS operations with MPI
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgsymcsrmatrix_dsymvmpi()’ multiplies a matrix ‘A’ by a real
 * scalar ‘alpha’ (‘α’) and a vector ‘x’, before adding the result to
 * another vector ‘y’.
 *
 * That is, ‘y = α*A*x + y’ or ‘y = α*A'*x + y’.
 *
 * The scalar ‘alpha’ is given as a double precision floating point
 * number.
 */
ACG_API int acgsymcsrmatrix_dsymvmpi(
    double alpha,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * x,
    struct acgvector * y,
    struct acghalo * halo,
    int64_t * num_flops,
    int64_t * num_bytes,
    MPI_Comm comm,
    int tag,
    int * mpierrcode);
#endif

#ifdef __cplusplus
}
#endif

#endif
