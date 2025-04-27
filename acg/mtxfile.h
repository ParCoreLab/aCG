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
 * Matrix Market files
 */

#ifndef ACG_MTXFILE_H
#define ACG_MTXFILE_H

#include "acg/config.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef ACG_HAVE_LIBZ
#include <zlib.h>
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Matrix Market data types
 */

/**
 * ‘mtxobject’ enumerates Matrix Market object types.
 */
enum mtxobject
{
    mtxmatrix,
    mtxvector,
};

/**
 * ‘mtxformat’ enumerates Matrix Market format types.
 */
enum mtxformat
{
    mtxarray,
    mtxcoordinate,
};

/**
 * ‘mtxfield’ enumerates Matrix Market field types.
 */
enum mtxfield
{
    mtxreal,
    mtxcomplex,
    mtxinteger,
    mtxpattern,
};

/**
 * ‘mtxsymmetry’ enumerates Matrix Market symmetry types.
 */
enum mtxsymmetry
{
    mtxgeneral,
    mtxsymmetric,
    mtxskewsymmetric,
    mtxhermitian,
};

/**
 * ‘mtxlayout’ enumerates layouts for matrices in array format.
 */
enum mtxlayout
{
    mtxrowmajor,
    mtxcolmajor,
};

/**
 * ‘mtxdatatype’ enumerates data types for matrix and vector values.
 */
enum mtxdatatype
{
    mtxdouble,
    mtxint,
};

/**
 * ‘mtxobjectstr()’ returns a string for a Matrix Market object type.
 */
ACG_API const char * mtxobjectstr(enum mtxobject object);

/**
 * ‘mtxformatstr()’ returns a string for a Matrix Market format type.
 */
ACG_API const char * mtxformatstr(enum mtxformat format);

/**
 * ‘mtxfieldstr()’ returns a string for a Matrix Market field type.
 */
ACG_API const char * mtxfieldstr(enum mtxfield field);

/**
 * ‘mtxsymmetrystr()’ returns a string for a Matrix Market symmetry type.
 */
ACG_API const char * mtxsymmetrystr(enum mtxsymmetry symmetry);

/**
 * ‘mtxdatatypestr()’ returns a string for a Matrix Market datatype type.
 */
ACG_API const char * mtxdatatypestr(enum mtxdatatype datatype);

/*
 * Data structures for Matrix Market files
 */

/**
 * ‘acgmtxfile’ is a data structure for storing Matrix Market files.
 *
 * A Matrix Market file consists of a header followed by data that
 * stores the actual nonzero matrix or vector values.
 *
 * 1. Header
 * ----------
 *
 * The Matrix Market header consists of three parts: 1) a header line,
 * 2) an optional section of comment lines, and 3) a size line.
 *
 * The header line is on the form
 *
 *   %%MatrixMarket object format field symmetry
 *
 * where
 *
 *   - ‘object’ is either ‘matrix’ or ‘vector’,
 *   - ‘format’ is either ‘array’ or ‘coordinate’,
 *   - ‘field’ is ‘real’, ‘complex’, ‘integer’, or ‘pattern’,
 *   - ‘symmetry’ is ‘general’, ‘symmetric’, ‘skew-symmetric’, or ‘Hermitian’.
 *
 * If present, comment lines must follow immediately after the header
 * line. Each comment line begins with the character ‘%’ and continues
 * until the end of the line.
 *
 * The size line, describes the size of the matrix or vector, and it
 * depends on the ‘object’ and ‘format’ values in the header, as shown
 * in the following table:
 *
 *     object   format       size line
 *    -------- ------------ -----------
 *     matrix   array        M N
 *     matrix   coordinate   M N K
 *     vector   array        M
 *     vector   coordinate   M K
 *
 * In the above table, ‘M’, ‘N’ and ‘K’ are decimal integers denoting
 * the number of rows, columns and nonzero values, respectively, of
 * the matrix or vector. Note that vectors always consist of a single
 * column. Also, the number of nonzeros for matrices or vectors in
 * array format can be inferred from the number of rows and columns
 * (and the symmetry). The number of columns or nonzeros are
 * therefore omitted in these cases.
 *
 * 2. Data
 * --------
 *
 * The format of data lines in a Matrix Market file depends on the
 * ‘object’, ‘format’ and ‘field’ values in the header. The different
 * data line formats are described in detail below.
 *
 * If ‘format’ is ‘array’, then each data line consists of 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’. A ‘field’ value of ‘pattern’ is not allowed.
 *
 * Otherwise, if ‘object’ is ‘matrix’ and ‘format’ is ‘coordinate’,
 * then each data line is on the form:
 *
 *   i j a
 *
 * where ‘i’ and ‘j’ are decimal integers denoting the row and column,
 * respectively, of the given (nonzero) value ‘a’. Furthermore, the
 * (nonzero) value ‘a’ is either 1) a single decimal number if ‘field’
 * is ‘real’, 2) a pair of decimal numbers if ‘field’ is ‘complex’, or
 * 3) a single decimal integer if ‘field’ is ‘integer’, or 4) it is
 * omitted if ‘field’ is ‘pattern’.
 *
 * Finally, if ‘object’ is ‘vector’ and ‘format’ is ‘coordinate’, then
 * each data line is on the form:
 *
 *   i a
 *
 * where ‘i’ is a decimal integer denoting the element of the given
 * (nonzero) value ‘a’. As before, the value ‘a’ is either 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’, or 4) it is omitted if ‘field’ is ‘pattern’.
 *
 * For matrices in array format, a layout must be specified to
 * designate whether values are stored in row or column major
 * order. For symmetric, skew-symmetric or Hermitian matrices, a row
 * major layout corresponds to storing the upper triangle of the
 * matrix in row major order or, equivalently, the lower triangle of
 * the matrix in column major order. Conversely, a column major layout
 * corresponds to storing the upper triangle of the matrix in column
 * major order or, equivalently, the lower triangle of the matrix in
 * row major order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 */
struct acgmtxfile
{
    /* header */
    enum mtxobject object;
    enum mtxformat format;
    enum mtxfield field;
    enum mtxsymmetry symmetry;

    /* size */
    acgidx_t nrows;
    acgidx_t ncols;
    int64_t nnzs;

    /* data */
    int idxbase;
    int nvalspernz;
    enum mtxdatatype datatype;
    acgidx_t * rowidx;
    acgidx_t * colidx;
    void * data;
};

/*
 * memory management
 */

/**
 * ‘acgmtxfile_free()’ frees resources associated with a Matrix
 * Market file.
 */
ACG_API void acgmtxfile_free(
    struct acgmtxfile * mtxfile);

/**
 * ‘acgmtxfile_alloc()’ allocates storage for a Matrix Market file.
 *
 * The object, format, field and symmetry must be supplied, as well as
 * the number of rows, columns and nonzeros.
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The ‘datatype’ argument specifies which data type to use for
 * storing the nonzero matrix or vector values.
 */
ACG_API int acgmtxfile_alloc(
    struct acgmtxfile * mtxfile,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnzs,
    int nvalspernz,
    int idxbase,
    enum mtxdatatype datatype);

/**
 * ‘acgmtxfile_copy()’ creates a copy of a Matrix Market file.
 */
ACG_API int acgmtxfile_copy(
    struct acgmtxfile * dst,
    const struct acgmtxfile * src);

/*
 * Matrix Market I/O
 */

/**
 * ‘acgmtxfile_fread()’ reads a Matrix Market file from a standard
 * I/O stream.
 *
 * The Matrix Market data is read from the given stream ‘f’.
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The ‘datatype’ argument specifies which data type to use for
 * storing the nonzero matrix or vector values.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int acgmtxfile_fread(
    struct acgmtxfile * mtxfile,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    enum mtxdatatype datatype,
    FILE * f,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);

#ifdef ACG_HAVE_LIBZ
/**
 * ‘acgmtxfile_gzread()’ reads a Matrix Market file from a
 * gzip-compressed stream.
 *
 * See also ‘acgmtxfile_fread()’.
 */
ACG_API int acgmtxfile_gzread(
    struct acgmtxfile * mtxfile,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    enum mtxdatatype datatype,
    gzFile f,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);
#endif

/**
 * ‘acgmtxfile_read()’ reads a Matrix Market file from a given path.
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The ‘datatype’ argument specifies which data type to use for
 * storing the nonzero matrix or vector values.
 *
 * If ‘path’ is ‘-’, then standard input is used.
 *
 * The file is assumed to be gzip-compressed if ‘gzip’ is ‘true’, and
 * uncompressed otherwise.
 *
 * The file is assumed to be stored in a binary Matrix Market format
 * if ‘binary’ is ‘true’, and in the usual text format otherwise.
 *
 * If an error code is returned, then ‘nlines’ and ‘nbytes’ are used
 * to return the line number and byte at which the error was
 * encountered when parsing the file.
 */
ACG_API int acgmtxfile_read(
    struct acgmtxfile * mtxfile,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    enum mtxdatatype datatype,
    const char * path,
    bool gzip,
    int64_t * nlines,
    int64_t * nbytes);

/*
 * partitioned matrices and vectors
 */

/**
 * ‘acgmtxfilepartition’ is a data structure that relates a part of a
 * partitioned Matrix Market file to an original, unpartitioned Matrix
 * Market file.
 */
struct acgmtxfilepartition
{
    int nparts;
    int part;

    /* total number of rows, columns and nonzeros */
    acgidx_t nrows;
    acgidx_t ncols;
    int64_t nnzs;

    /* nonzero rows */
    acgidx_t nnzrows;
    acgidx_t nexclusivenzrows;
    acgidx_t nownednzrows;
    acgidx_t nsharednzrows;
    acgidx_t * nzrows;
    int * ownednzrownsegments;
    acgidx_t ownednzrowsegmentssize;
    int * ownednzrowsegments;
    int * sharednzrowowners;

    /* nonzero columns */
    acgidx_t nnzcols;
    acgidx_t nexclusivenzcols;
    acgidx_t nownednzcols;
    acgidx_t nsharednzcols;
    acgidx_t * nzcols;
    int * ownednzcolnsegments;
    acgidx_t ownednzcolsegmentssize;
    int * ownednzcolsegments;
    int * sharednzcolowners;

    /* nonzeros */
    int64_t npnzs;
    int64_t nexclusivenzcolnzs;
    int64_t nownednzcolnzs;
    int64_t nsharednzcolnzs;
};

/**
 * ‘acgmtxfilepartition_free()’ frees resources associated with a
 * partition of a Matrix Market file.
 */
ACG_API void acgmtxfilepartition_free(
    struct acgmtxfilepartition * partition);

/**
 * ‘acgmtxfilepartition_copy()’ creates a copy of a partition.
 */
ACG_API int acgmtxfilepartition_copy(
    struct acgmtxfilepartition * dst,
    const struct acgmtxfilepartition * src);

/**
 * ‘acgmtxfile_partition_rowwise()’ performs a rowwise partitioning
 * of a matrix or vector.
 *
 * A partitioning of the matrix (or vector) rows is given by
 * specifying the number of parts, ‘nparts’, and a partitioning vector
 * ‘rowpart’, which must be of length ‘nrows’. The partitioning vector
 * contains integers from ‘0’ to ‘nparts-1’, such that ‘rowpart[i]’
 * indicates the partition to which row ‘i’ belongs.
 */
ACG_API int acgmtxfile_partition_rowwise(
    const struct acgmtxfile * src,
    int nparts,
    const int * rowpart,
    const int * colpart,
    struct acgmtxfile * dst,
    struct acgmtxfilepartition * parts);

struct acghalo;

/**
 * ‘acgmtxfilepartition_rowhalo()’ sets up a halo exchange
 * communication pattern to send and receive data associated with the
 * rows of a partitioned Matrix Market file.
 */
ACG_API int acgmtxfilepartition_rowhalo(
    const struct acgmtxfilepartition * part,
    struct acghalo * halo);

/**
 * ‘acgmtxfilepartition_colhalo()’ sets up a halo exchange
 * communication pattern to send and receive data associated with the
 * columns of a partitioned Matrix Market file.
 */
ACG_API int acgmtxfilepartition_colhalo(
    const struct acgmtxfilepartition * part,
    struct acghalo * halo);

/*
 * Matrix and vector reordering
 */

/**
 * ‘acgmtxfile_permutenzs()’ permutes the nonzero entries of a matrix
 * or vector according to a given permutation.
 *
 * A permutation of the matrix (or vector) nonzeros is given by the
 * permutation vector ‘nzperm’, which must be of length equal to the
 * number of nonzeros, (i.e., ‘mtxfile->nnzs’). Each integer in the
 * range from ‘0’ to ‘mtxfile->nnzs-1’ should appear exactly once. If
 * ‘rowidx’, ‘colidx’ and ‘data’ are arrays containing the rows,
 * columns and nonzero values of the matrix (or vector), then the
 * value of ‘rowidx[i]’ prior to applying the permutation will be
 * equal to ‘rowidx[nzperm[i]]’ after applying the permutation, and
 * similarly for ‘colidx’ and ‘data’.
 */
ACG_API int acgmtxfile_permutenzs(
    struct acgmtxfile * mtxfile,
    const int64_t * nzperm);

/*
 * MPI distributed-memory Matrix Market files
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgmtxfile_send()’ sends a Matrix Market file to another MPI
 * process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘acgmtxfile_recv()’.
 */
ACG_API int acgmtxfile_send(
    const struct acgmtxfile * mtxfile,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgmtxfile_recv()’ receives a Matrix Market file from another MPI
 * process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘acgmtxfile_send()’.
 */
ACG_API int acgmtxfile_recv(
    struct acgmtxfile * mtxfile,
    int src,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgmtxfilepartition_send()’ sends partitioning information to
 * another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to
 * ‘acgmtxfilepartition_recv()’.
 */
ACG_API int acgmtxfilepartition_send(
    const struct acgmtxfilepartition * part,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘acgmtxfilepartition_recv()’ receives partitioning information
 * from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘acgmtxfilepartition_send()’.
 */
ACG_API int acgmtxfilepartition_recv(
    struct acgmtxfilepartition * part,
    int src,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);
#endif

/*
 * helper functions
 */

/**
 * ‘mtxfile_nnz()’ returns the number of nonzeros stored in a Matrix
 * Market file with the given object, format, field and symmetry.
 *
 * If ‘object’ is ‘matrix’ and ‘format’ is ‘array’, then the number of
 * nonzero entries is:
 *
 *   - M times N if ‘field’ is ‘general’,
 *   - M(M+1)/2 if ‘field’ is ‘symmetric’ or ‘hermitian’,
 *   - M(M-1)/2 if ‘field’ is ‘skew-symmetric’,
 *
 * where M and N are the number of matrix rows and columns,
 * respectively, specified by the size line of the Matrix Market file.
 * Moreover, M and N must be equal if ‘field’ is ‘symmetric’,
 * ‘hermitian’ or ‘skew-symmetric’.
 *
 * If ‘object’ is ‘vector’ and ‘format’ is ‘array’, the number of
 * nonzero entries is equal to the vector dimensions M, as specified
 * by the size line in the Matrix Market file.
 *
 * In all other cases, an arbitrary number of nonzero entries are
 * allowed, and the number of nonzero entries is specified by the size
 * line in the Matrix Market file. In this case, this function does
 * nothing.
 *
 * The number of nonzero entries is stored in ‘nnz’.
 */
ACG_API int mtxfile_nnz(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t * nnz);

/**
 * ‘mtxfile_nvalspernz()’ returns the number of values per nonzero in
 * a Matrix Market file with the given field.
 *
 * There is a single value per data line if ‘field’ is ‘real’ or
 * ‘integer’, and two values per data line if ‘field’ is ‘complex’.
 * If ‘field’ is ‘pattern’, then there are no values.
 */
ACG_API int mtxfile_nvalspernz(
    enum mtxfield field,
    int * nvalspernz);

/*
 * Matrix Market I/O
 */

/**
 * ‘mtxfile_fread_header()’ reads the header of a Matrix Market file.
 *
 * The header of a Matrix Market file consists of the following three
 * parts: 1) a header line, 2) an optional section containing one or
 * more comment lines, and 3) a size line.
 *
 * The header line is on the form
 *
 *   %%MatrixMarket object format field symmetry
 *
 * where
 *
 *   - ‘object’ is either ‘matrix’ or ‘vector’,
 *   - ‘format’ is either ‘array’ or ‘coordinate’,
 *   - ‘field’ is ‘real’, ‘complex’, ‘integer’, or ‘pattern’,
 *   - ‘symmetry’ is ‘general’, ‘symmetric’, ‘skew-symmetric’, or ‘Hermitian’.
 *
 * If present, comment lines must follow immediately after the header
 * line. Each comment line begins with the character ‘%’ and continues
 * until the end of the line.
 *
 * The size line, describes the size of the matrix or vector, and it
 * depends on the ‘object’ and ‘format’ values in the header, as shown
 * in the following table:
 *
 *     object   format       size line
 *    -------- ------------ -----------
 *     matrix   array        M N
 *     matrix   coordinate   M N K
 *     vector   array        M
 *     vector   coordinate   M K
 *
 * In the above table, ‘M’, ‘N’ and ‘K’ are decimal integers denoting
 * the number of rows, columns and nonzero values, respectively, of
 * the matrix or vector. Note that vectors always consist of a single
 * column. Also, the number of nonzeros for matrices or vectors in
 * array format can be inferred from the number of rows and columns
 * (and the symmetry). The number of columns or nonzeros are
 * therefore omitted in these cases.
 *
 * The header is read from the given stream ‘f’.
 *
 * The object, format, field and symmetry are stored in the locations
 * pointed to by the corresponding function parameters. Similarly, the
 * number of rows, columns and nonzeros are stored in ‘nrows’, ‘ncols’
 * and ‘nnz’, respectively.
 *
 * The following rules determine the values of ‘ncols’ and ‘nnz’:
 *
 *  1. If ‘object’ is ‘vector’, then ‘*ncols’ is set to ‘1’.
 *
 *  2. If ‘object’ is ‘matrix’ and ‘format’ is ‘array’, then ‘*nnz’ is:
 *
 *       - M times N if ‘field’ is ‘general’,
 *       - M(M+1)/2 if ‘field’ is ‘symmetric’ or ‘hermitian’,
 *       - M(M-1)/2 if ‘field’ is ‘skew-symmetric’,
 *
 *     where M and N are the number of matrix rows and columns,
 *     respectively. Moreover, M and N must be equal if ‘field’ is
 *     ‘symmetric’, ‘hermitian’ or ‘skew-symmetric’.
 *
 *  3. If ‘object’ is ‘vector’ and ‘format’ is ‘array’, the number of data
 *     lines is equal to the vector dimensions M, as specified by the size
 *     line in the Matrix Market file.
 *
 *  4. In all other cases, the number of data lines is equal to the number
 *     of nonzeros K specified in the size line.
 *
 * The number of values per nonzero is stored in ‘nvalspernz’, which
 * is set to ‘1’ if ‘field’ is ‘real’ or ‘integer’, ‘2’ if ‘field’ is
 * ‘complex’, and ‘0’ if ‘field’ is ‘pattern’.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int mtxfile_fread_header(
    FILE * f,
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxfield * field,
    enum mtxsymmetry * symmetry,
    acgidx_t * nrows,
    acgidx_t * ncols,
    int64_t * nnz,
    int * nvalspernz,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);

/**
 * ‘mtxfile_fread_data_int()’ reads data lines of a Matrix Market file
 * from a standard I/O stream, storing nonzero values as integers.
 *
 * The format of data lines in a Matrix Market file depends on the
 * ‘object’, ‘format’ and ‘field’ values in the header. The different
 * data line formats are described in detail below.
 *
 * If ‘format’ is ‘array’, then each data line consists of 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’. A ‘field’ value of ‘pattern’ is not allowed.
 *
 * Otherwise, if ‘object’ is ‘matrix’ and ‘format’ is ‘coordinate’,
 * then each data line is on the form:
 *
 *   i j a
 *
 * where ‘i’ and ‘j’ are decimal integers denoting the row and column,
 * respectively, of the given (nonzero) value ‘a’. Furthermore, the
 * (nonzero) value ‘a’ is either 1) a single decimal number if ‘field’
 * is ‘real’, 2) a pair of decimal numbers if ‘field’ is ‘complex’, or
 * 3) a single decimal integer if ‘field’ is ‘integer’, or 4) it is
 * omitted if ‘field’ is ‘pattern’.
 *
 * Finally, if ‘object’ is ‘vector’ and ‘format’ is ‘coordinate’, then
 * each data line is on the form:
 *
 *   i a
 *
 * where ‘i’ is a decimal integer denoting the element of the given
 * (nonzero) value ‘a’. As before, the value ‘a’ is either 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’, or 4) it is omitted if ‘field’ is ‘pattern’.
 *
 * The Matrix Market data is read from the given stream ‘f’.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_fread_header()’).
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The rows, columns and values of the underlying matrix or vector are
 * stored in the arrays ‘rowidx’, ‘colidx’ and ‘a’, respectively. The
 * length of the ‘rowidx’ and ‘colidx’ arrays must be at least ‘nnz’,
 * whereas the length of the array ‘a’ must be at least equal to ‘nnz’
 * times ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file). Any of
 * the arrays may be set to ‘NULL’, if the data is not needed.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int mtxfile_fread_data_int(
    FILE * f,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    int * a,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);

/**
 * ‘mtxfile_fread_data_double()’ reads data lines of a Matrix Market
 * file from a standard I/O stream, storing nonzero values as
 * double-precision floating-point numbers.
 *
 * The format of data lines in a Matrix Market file depends on the
 * ‘object’, ‘format’ and ‘field’ values in the header. The different
 * data line formats are described in detail below.
 *
 * If ‘format’ is ‘array’, then each data line consists of 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’. A ‘field’ value of ‘pattern’ is not allowed.
 *
 * Otherwise, if ‘object’ is ‘matrix’ and ‘format’ is ‘coordinate’,
 * then each data line is on the form:
 *
 *   i j a
 *
 * where ‘i’ and ‘j’ are decimal integers denoting the row and column,
 * respectively, of the given (nonzero) value ‘a’. Furthermore, the
 * (nonzero) value ‘a’ is either 1) a single decimal number if ‘field’
 * is ‘real’, 2) a pair of decimal numbers if ‘field’ is ‘complex’, or
 * 3) a single decimal integer if ‘field’ is ‘integer’, or 4) it is
 * omitted if ‘field’ is ‘pattern’.
 *
 * Finally, if ‘object’ is ‘vector’ and ‘format’ is ‘coordinate’, then
 * each data line is on the form:
 *
 *   i a
 *
 * where ‘i’ is a decimal integer denoting the element of the given
 * (nonzero) value ‘a’. As before, the value ‘a’ is either 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’, or 4) it is omitted if ‘field’ is ‘pattern’.
 *
 * The Matrix Market data is read from the given stream ‘f’.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_fread_header()’).
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The rows, columns and values of the underlying matrix or vector are
 * stored in the arrays ‘rowidx’, ‘colidx’ and ‘a’, respectively. The
 * length of the ‘rowidx’ and ‘colidx’ arrays must be at least ‘nnz’,
 * whereas the length of the array ‘a’ must be at least equal to ‘nnz’
 * times ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file). Any of
 * the arrays may be set to ‘NULL’, if the data is not needed.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int mtxfile_fread_data_double(
    FILE * f,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * a,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);

/**
 * ‘mtxfile_fwrite_double()’ writes a Matrix Market file to a standard
 * I/O stream. Values are given as double-precision floating-point
 * numbers.
 *
 * The Matrix Market file is written to the given stream ‘f’.
 *
 * See ‘mtxfile_fread_data_double()’ for a description of the
 * different data formats of Matrix Market files.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’.
 *
 * For matrices in array format, the matrix or vector values are
 * simply written in the order in which they are stored. It is up to
 * the user to ensure that values are ordered, for example, in row or
 * column major order, if this is required. Similarly, for symmetric,
 * skew-symmetric or Hermitian matrices in array format, the user may
 * want to ensure, for example, that the upper triangle of the matrix
 * is written in row major order or, equivalently, the lower triangle
 * of the matrix in column major order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ if the values of
 * ‘rowidx’ and ‘colidx’ are 1-based, thus requiring no conversion, or
 * or ‘0’ if ‘rowidx’ and ‘colidx’ should be converted from 0-based
 * indexing.
 *
 * The rows, columns and values of the underlying matrix or vector are
 * stored in the arrays ‘rowidx’, ‘colidx’ and ‘vals’, respectively.
 * The length of the ‘rowidx’ and ‘colidx’ arrays must be at least
 * ‘nnz’, whereas the length of the array ‘vals’ must be at least
 * equal to ‘nnz’ times ‘nvalspernz’, (which depends on the object,
 * format, field and symmetry specified in the header of the Matrix
 * Market file). Any of the arrays may be set to ‘NULL’, if the data
 * is not needed.
 *
 * If it is not ‘NULL’, then ‘nbytes’ is used to store the number of
 * bytes that were written.
 */
ACG_API int mtxfile_fwrite_double(
    FILE * f,
    int binary,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    const char * comments,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    int idxbase,
    const acgidx_t * rowidx,
    const acgidx_t * colidx,
    const double * vals,
    const char * numfmt,
    int64_t * nbytes);

/**
 * ‘mtxfile_fwrite_int()’ writes a Matrix Market file to a standard
 * I/O stream. Values are given as integers.
 *
 * See also ‘mtxfile_fwrite_double()’.
 */
ACG_API int mtxfile_fwrite_int(
    FILE * f,
    int binary,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    const char * comments,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    int idxbase,
    const acgidx_t * rowidx,
    const acgidx_t * colidx,
    const int * vals,
    const char * numfmt,
    int64_t * nbytes);

#ifdef ACG_HAVE_MPI
/**
 * ‘mtxfile_fwrite_mpi_double()’ gathers Matrix Market data from all
 * processes in a given communicator to a specified root process and
 * outputs a Matrix Markte file to a standard I/O stream. Values are
 * given as double-precision floating-point numbers.
 *
 * The Matrix Market file is written to the given stream ‘f’.
 *
 * See ‘mtxfile_fread_data_double()’ for a description of the
 * different data formats of Matrix Market files.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘*nnz’ and ‘nvalspernz’.
 *
 * For matrices in array format, the matrix or vector values are
 * simply written in the order in which they are stored. It is up to
 * the user to ensure that values are ordered, for example, in row or
 * column major order, if this is required. Similarly, for symmetric,
 * skew-symmetric or Hermitian matrices in array format, the user may
 * want to ensure, for example, that the upper triangle of the matrix
 * is written in row major order or, equivalently, the lower triangle
 * of the matrix in column major order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ if the values of
 * ‘rowidx’ and ‘colidx’ are 1-based, thus requiring no conversion, or
 * or ‘0’ if ‘rowidx’ and ‘colidx’ should be converted from 0-based
 * indexing.
 *
 * On each process, the rows, columns and values of the underlying
 * matrix or vector are specified through the arrays ‘prowidx’,
 * ‘pcolidx’ and ‘pvals’, respectively. The length of the ‘prowidx’
 * and ‘pcolidx’ arrays must be at least ‘npnz’, whereas the length of
 * the array ‘pvals’ must be at least equal to ‘npnz’ times
 * ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file).
 *
 * Furthermore, on each process, the matrix or vector consists of
 * ‘npnzrows’ and ‘npnzcols’ nonzero rows and columns, respectively
 * These are mapped to the global numbering of rows and columns by the
 * arrays ‘pnzrows’ and ‘pnzcols’.
 *
 * If it is not ‘NULL’, then ‘nbytes’ is used to store the number of
 * bytes that were written.
 */
ACG_API int mtxfile_fwrite_mpi_double(
    FILE * f,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    const char * comments,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t * nnz,
    int nvalspernz,
    int64_t npnz,
    int idxbase,
    const acgidx_t * prowidx,
    const acgidx_t * pcolidx,
    const double * pvals,
    const acgidx_t npnzrows,
    const acgidx_t * pnzrows,
    const acgidx_t npnzcols,
    const acgidx_t * pnzcols,
    const char * numfmt,
    int64_t * nbytes,
    int root,
    MPI_Comm comm,
    int * mpierrcode);
#endif

/*
 * Matrix Market I/O for gzip-compressed streams
 */

#ifdef ACG_HAVE_LIBZ
/**
 * ‘mtxfile_gzread_header()’ reads the header of a Matrix Market file
 * from a gzip-compressed stream.
 *
 * The header of a Matrix Market file consists of the following three
 * parts: 1) a header line, 2) an optional section containing one or
 * more comment lines, and 3) a size line.
 *
 * The header line is on the form
 *
 *   %%MatrixMarket object format field symmetry
 *
 * where
 *
 *   - ‘object’ is either ‘matrix’ or ‘vector’,
 *   - ‘format’ is either ‘array’ or ‘coordinate’,
 *   - ‘field’ is ‘real’, ‘complex’, ‘integer’, or ‘pattern’,
 *   - ‘symmetry’ is ‘general’, ‘symmetric’, ‘skew-symmetric’, or ‘Hermitian’.
 *
 * If present, comment lines must follow immediately after the header
 * line. Each comment line begins with the character ‘%’ and continues
 * until the end of the line.
 *
 * The size line, describes the size of the matrix or vector, and it
 * depends on the ‘object’ and ‘format’ values in the header, as shown
 * in the following table:
 *
 *     object   format       size line
 *    -------- ------------ -----------
 *     matrix   array        M N
 *     matrix   coordinate   M N K
 *     vector   array        M
 *     vector   coordinate   M K
 *
 * In the above table, ‘M’, ‘N’ and ‘K’ are decimal integers denoting
 * the number of rows, columns and nonzero values, respectively, of
 * the matrix or vector. Note that vectors always consist of a single
 * column. Also, the number of nonzeros for matrices or vectors in
 * array format can be inferred from the number of rows and columns
 * (and the symmetry). The number of columns or nonzeros are
 * therefore omitted in these cases.
 *
 * The header is read from the given stream ‘f’.
 *
 * The object, format, field and symmetry are stored in the locations
 * pointed to by the corresponding function parameters. Similarly, the
 * number of rows, columns and nonzeros are stored in ‘nrows’, ‘ncols’
 * and ‘nnz’, respectively.
 *
 * The following rules determine the values of ‘ncols’ and ‘nnz’:
 *
 *  1. If ‘object’ is ‘vector’, then ‘*ncols’ is set to ‘1’.
 *
 *  2. If ‘object’ is ‘matrix’ and ‘format’ is ‘array’, then ‘*nnz’ is:
 *
 *       - M times N if ‘field’ is ‘general’,
 *       - M(M+1)/2 if ‘field’ is ‘symmetric’ or ‘hermitian’,
 *       - M(M-1)/2 if ‘field’ is ‘skew-symmetric’,
 *
 *     where M and N are the number of matrix rows and columns,
 *     respectively. Moreover, M and N must be equal if ‘field’ is
 *     ‘symmetric’, ‘hermitian’ or ‘skew-symmetric’.
 *
 *  3. If ‘object’ is ‘vector’ and ‘format’ is ‘array’, the number of data
 *     lines is equal to the vector dimensions M, as specified by the size
 *     line in the Matrix Market file.
 *
 *  4. In all other cases, the number of data lines is equal to the number
 *     of nonzeros K specified in the size line.
 *
 * The number of values per nonzero is stored in ‘nvalspernz’, which
 * is set to ‘1’ if ‘field’ is ‘real’ or ‘integer’, ‘2’ if ‘field’ is
 * ‘complex’, and ‘0’ if ‘field’ is ‘pattern’.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int mtxfile_gzread_header(
    gzFile f,
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxfield * field,
    enum mtxsymmetry * symmetry,
    acgidx_t * nrows,
    acgidx_t * ncols,
    int64_t * nnz,
    int * nvalspernz,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);

/**
 * ‘mtxfile_gzread_data_int()’ reads data lines of a Matrix Market
 * file from a gzip-compressed stream, storing nonzero values as
 * integers.
 *
 * The format of data lines in a Matrix Market file depends on the
 * ‘object’, ‘format’ and ‘field’ values in the header. The different
 * data line formats are described in detail below.
 *
 * If ‘format’ is ‘array’, then each data line consists of 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’. A ‘field’ value of ‘pattern’ is not allowed.
 *
 * Otherwise, if ‘object’ is ‘matrix’ and ‘format’ is ‘coordinate’,
 * then each data line is on the form:
 *
 *   i j a
 *
 * where ‘i’ and ‘j’ are decimal integers denoting the row and column,
 * respectively, of the given (nonzero) value ‘a’. Furthermore, the
 * (nonzero) value ‘a’ is either 1) a single decimal number if ‘field’
 * is ‘real’, 2) a pair of decimal numbers if ‘field’ is ‘complex’, or
 * 3) a single decimal integer if ‘field’ is ‘integer’, or 4) it is
 * omitted if ‘field’ is ‘pattern’.
 *
 * Finally, if ‘object’ is ‘vector’ and ‘format’ is ‘coordinate’, then
 * each data line is on the form:
 *
 *   i a
 *
 * where ‘i’ is a decimal integer denoting the element of the given
 * (nonzero) value ‘a’. As before, the value ‘a’ is either 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’, or 4) it is omitted if ‘field’ is ‘pattern’.
 *
 * The Matrix Market data is read from the given gzip-compressed
 * stream ‘f’.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_fread_header()’).
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The rows, columns and values of the underlying matrix or vector are
 * stored in the arrays ‘rowidx’, ‘colidx’ and ‘a’, respectively. The
 * length of the ‘rowidx’ and ‘colidx’ arrays must be at least ‘nnz’,
 * whereas the length of the array ‘a’ must be at least equal to ‘nnz’
 * times ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file). Any of
 * the arrays may be set to ‘NULL’, if the data is not needed.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int mtxfile_gzread_data_int(
    gzFile f,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    int * a,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);

/**
 * ‘mtxfile_gzread_data_double()’ reads data lines of a Matrix Market
 * file from a gzip-compressed stream, storing nonzero values as
 * double-precision floating-point numbers.
 *
 * The format of data lines in a Matrix Market file depends on the
 * ‘object’, ‘format’ and ‘field’ values in the header. The different
 * data line formats are described in detail below.
 *
 * If ‘format’ is ‘array’, then each data line consists of 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’. A ‘field’ value of ‘pattern’ is not allowed.
 *
 * Otherwise, if ‘object’ is ‘matrix’ and ‘format’ is ‘coordinate’,
 * then each data line is on the form:
 *
 *   i j a
 *
 * where ‘i’ and ‘j’ are decimal integers denoting the row and column,
 * respectively, of the given (nonzero) value ‘a’. Furthermore, the
 * (nonzero) value ‘a’ is either 1) a single decimal number if ‘field’
 * is ‘real’, 2) a pair of decimal numbers if ‘field’ is ‘complex’, or
 * 3) a single decimal integer if ‘field’ is ‘integer’, or 4) it is
 * omitted if ‘field’ is ‘pattern’.
 *
 * Finally, if ‘object’ is ‘vector’ and ‘format’ is ‘coordinate’, then
 * each data line is on the form:
 *
 *   i a
 *
 * where ‘i’ is a decimal integer denoting the element of the given
 * (nonzero) value ‘a’. As before, the value ‘a’ is either 1) a single
 * decimal number if ‘field’ is ‘real’, 2) a pair of decimal numbers
 * if ‘field’ is ‘complex’, or 3) a single decimal integer if ‘field’
 * is ‘integer’, or 4) it is omitted if ‘field’ is ‘pattern’.
 *
 * The Matrix Market data is read from the given gzip-compressed
 * stream ‘f’.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
 *
 * The ‘layout’ argument is used to specify whether matrices in array
 * format are stored in row or column major order. For symmetric,
 * skew-symmetric or Hermitian matrices, a row major layout
 * corresponds to storing the upper triangle of the matrix in row
 * major order or, equivalently, the lower triangle of the matrix in
 * column major order. Conversely, a column major layout corresponds
 * to storing the upper triangle of the matrix in column major order
 * or, equivalently, the lower triangle of the matrix in row major
 * order.
 *
 * The Matrix Market format uses 1-based indexing of rows and columns.
 * The ‘idxbase’ argument should be set to ‘1’ to keep the 1-based
 * indexing or ‘0’ to convert to 0-based indexing.
 *
 * The rows, columns and values of the underlying matrix or vector are
 * stored in the arrays ‘rowidx’, ‘colidx’ and ‘a’, respectively. The
 * length of the ‘rowidx’ and ‘colidx’ arrays must be at least ‘nnz’,
 * whereas the length of the array ‘a’ must be at least equal to ‘nnz’
 * times ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file). Any of
 * the arrays may be set to ‘NULL’, if the data is not needed.
 *
 * If they are not ‘NULL’, then ‘nlines’ and ‘nbytes’ are used to
 * store the number of lines and bytes that have been read,
 * respectively.
 *
 * If ‘linebuf’ is not ‘NULL’, then it must point to an array of
 * length ‘linemax’. This buffer is used for reading lines from the
 * stream. Otherwise, if ‘linebuf’ is ‘NULL’, then a temporary buffer
 * is allocated and used, and the maximum line length is determined by
 * calling ‘sysconf()’ with ‘_SC_LINE_MAX’.
 */
ACG_API int mtxfile_gzread_data_double(
    gzFile f,
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * a,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf);
#endif

/*
 * Matrix and vector partitioning
 */

/**
 * ‘mtxfile_partition_rowwise()’ performs a rowwise partitioning of
 * the nonzero entries of a matrix or vector.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’ and ‘nnz’.
 *
 * To use 1-based indexing of rows and columns, set ‘idxbase’ to ‘1’.
 * Otherwise, to use 0-based indexing, set ‘idxbase’ to ‘0’. The rows
 * of the underlying matrix or vector nonzeros must be provided by the
 * array ‘rowidx’, which must be of length ‘nnz’.
 *
 * A partitioning of the matrix (or vector) rows is given by
 * specifying the number of parts, ‘nparts’, and a partitioning vector
 * ‘rowpart’, which must be of length ‘nrows’. The partitioning vector
 * contains integers from ‘0’ to ‘nparts-1’, such that ‘rowpart[i]’
 * indicates the partition to which row ‘i’ belongs.
 *
 * The array ‘nzpart’ must be of length ‘nnz’, and it is used to write
 * the partition vector for the matrix (or vector) nonzeros, with
 * values ranging from ‘0’ to ‘nparts-1’.
 *
 * If ‘nzpartptr’ is not ‘NULL’, then it must point to an array of
 * length ‘nparts+1’, and it is used to write the prefix sum of the
 * size of each part in the final partitioning. In other words,
 * ‘nzpartptr[p]’ is the location of the first nonzero entry in the
 * ‘p’-th part, if the nonzeros are sorted by parts.
 *
 * If ‘nzperm’ is not ‘NULL’, then it must point to an array of length
 * ‘nnz’, and it is used to write a permutation of the matrix (or
 * vector) nonzeros that sorts nonzeros by parts according to the
 * partitioning.
 */
ACG_API int mtxfile_partition_rowwise(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int idxbase,
    const acgidx_t * rowidx,
    int nparts,
    const int * rowpart,
    int * nzpart,
    int64_t * nzpartptr,
    int64_t * nzperm);

/**
 * ‘mtxfile_partition_columnwise()’ performs a columnwise partitioning
 * of the nonzero entries of a matrix or vector.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’ and ‘nnz’.
 *
 * To use 1-based indexing of rows and columns, set ‘idxbase’ to ‘1’.
 * Otherwise, to use 0-based indexing, set ‘idxbase’ to ‘0’. The
 * columns of the underlying matrix or vector nonzeros must be
 * provided by the array ‘colidx’, which must be of length ‘nnz’.
 *
 * A partitioning of the matrix (or vector) columns is given by
 * specifying the number of parts, ‘nparts’, and a partitioning vector
 * ‘colpart’, which must be of length ‘ncols’. The partitioning vector
 * contains integers from ‘0’ to ‘nparts-1’, such that ‘colpart[i]’
 * indicates the partition to which column ‘i’ belongs.
 *
 * The array ‘nzpart’ must be of length ‘nnz’, and it is used to write
 * the partition vector for the matrix (or vector) nonzeros, with
 * values ranging from ‘0’ to ‘nparts-1’.
 *
 * If ‘nzpartptr’ is not ‘NULL’, then it must point to an array of
 * length ‘nparts+1’, and it is used to write the prefix sum of the
 * size of each part in the final partitioning. In other words,
 * ‘nzpartptr[p]’ is the location of the first nonzero entry in the
 * ‘p’-th part, if the nonzeros are sorted by parts.
 *
 * If ‘nzperm’ is not ‘NULL’, then it must point to an array of length
 * ‘nnz’, and it is used to write a permutation of the matrix (or
 * vector) nonzeros that sorts nonzeros by parts according to the
 * partitioning.
 */
ACG_API int mtxfile_partition_columnwise(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int idxbase,
    const acgidx_t * colidx,
    int nparts,
    const int * colpart,
    int * nzpart,
    int64_t * nzpartptr,
    int64_t * nzperm);

/*
 * Matrix and vector reordering
 */

/**
 * ‘mtxfile_compact()’ reorders the rows and/or columns of a matrix or
 * vector by ordering non-empty rows (or columns) first and empty rows
 * (or columns) last.
 *
 * This is often used after distributing a matrix or vector among
 * multiple processes, since each process usually has only a few
 * non-empty rows (or columns). The numbering of rows (or columns) may
 * change, but the ordering of the nonzeros remains the same.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’ and ‘nnz’.
 *
 * The rows and columns of the underlying matrix or vector nonzeros
 * are specified through ‘rowidx’ and ‘colidx’, respectively. Rows and
 * columns use 1-based indexing if ‘idxbase’ is ‘1’, and 0-based
 * indexing if ‘idxbase’ is ‘0’. The length of the ‘rowidx’ and
 * ‘colidx’ arrays must be at least ‘nnz’. Either of the arrays may be
 * set to ‘NULL’, if the data is not needed.
 *
 * If ‘nnzrows’ is not ‘NULL’, then it is used to store the number of
 * non-empty rows. Moreover, if ‘nzrows’ is not ‘NULL’, then it is
 * used to store a pointer to an array allocated with ‘malloc()’. The
 * size of the allocated array is equal to the number of non-empty
 * rows multiplied by ‘sizeof(acgidx_t)’. It is the caller's
 * responsibility to call ‘free()’ to release the allocated
 * memory. The underlying array stores the mapping from the non-empty,
 * reordered rows to the original row numbers prior to reordering.
 * Thus, one value is stored for each non-empty row. ‘nnzcols’ and
 * ‘nzcols’ are similarly used for column reordering.
 */
ACG_API int mtxfile_compact(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    acgidx_t * nnzrows,
    acgidx_t ** nzrows,
    acgidx_t * nnzcols,
    acgidx_t ** nzcols);

/**
 * ‘mtxfile_compactnzs()’ merges duplicate, neighbouring nonzero
 * entries of a matrix or vector.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘*nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
 *
 * The rows, columns and values of the underlying matrix or vector are
 * specified through the arrays ‘rowidx’, ‘colidx’ and ‘vals’,
 * respectively. The length of the ‘rowidx’ and ‘colidx’ arrays must
 * be at least ‘*nnz’, whereas the length of the array ‘vals’ must be
 * at least equal to ‘*nnz’ times ‘nvalspernz’, (which depends on the
 * object, format, field and symmetry specified in the header of the
 * Matrix Market file). Any of the arrays may be set to ‘NULL’, if the
 * data is not needed.
 *
 * If ‘nzperm’ is not ‘NULL’, then it must point to an array of length
 * ‘*nnz’, which is used to store the permutation applied to the
 * matrix (or vector) nonzeros. The array consists of integers in the
 * range from ‘0’ to ‘*nnz-1’, where a number may appear more than
 * once if multiple values have been merged together into a single
 * nonzero entry. The value of ‘rowidx[i]’ prior to performing the
 * compaction will be equal to ‘rowidx[nzperm[i]]’ after applying the
 * compaction. The same is true for ‘colidx’. However, the value of
 * ‘vals[j]’ after compaction will be equal to the sum of all nonzero
 * entries with the same row, ‘rowidx[j]’, and column, ‘colidx[j]’,
 * which have now been merged together.
 */
ACG_API int mtxfile_compactnzs(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t * nnz,
    int nvalspernz,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * vals,
    int64_t * nzperm);

/**
 * ‘mtxfile_compactnzs_unsorted()’ sorts nonzero entries of a matrix
 * or vector and then removes duplicate, neighbouring entries.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘*nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
 *
 * The rows, columns and values of the underlying matrix or vector are
 * specified through the arrays ‘rowidx’, ‘colidx’ and ‘vals’,
 * respectively. The length of the ‘rowidx’ and ‘colidx’ arrays must
 * be at least ‘*nnz’, whereas the length of the array ‘vals’ must be
 * at least equal to ‘*nnz’ times ‘nvalspernz’, (which depends on the
 * object, format, field and symmetry specified in the header of the
 * Matrix Market file). Any of the arrays may be set to ‘NULL’, if the
 * data is not needed.
 *
 * If ‘nzperm’ is not ‘NULL’, then it must point to an array of length
 * ‘*nnz’, which is used to store the permutation applied to the
 * matrix (or vector) nonzeros. The array consists of integers in the
 * range from ‘0’ to ‘*nnz-1’, where a number may appear more than
 * once if multiple values have been merged together into a single
 * nonzero entry. The value of ‘rowidx[i]’ prior to performing the
 * compaction will be equal to ‘rowidx[nzperm[i]]’ after applying the
 * compaction. The same is true for ‘colidx’. However, the value of
 * ‘vals[j]’ after compaction will be equal to the sum of all nonzero
 * entries with the same row, ‘rowidx[j]’, and column, ‘colidx[j]’,
 * which have now been merged together.
 */
ACG_API int mtxfile_compactnzs_unsorted(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t * nnz,
    int nvalspernz,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * vals,
    int64_t * nzperm);

/**
 * ‘mtxfile_permutenzs()’ permutes the nonzero entries of a matrix or
 * vector according to a given permutation.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
 *
 * The rows, columns and values of the underlying matrix or vector are
 * specified through the arrays ‘rowidx’, ‘colidx’ and ‘vals’,
 * respectively. The length of the ‘rowidx’ and ‘colidx’ arrays must
 * be at least ‘nnz’, whereas the length of the array ‘vals’ must be
 * at least equal to ‘nnz’ times ‘nvalspernz’, (which depends on the
 * object, format, field and symmetry specified in the header of the
 * Matrix Market file). Any of the arrays may be set to ‘NULL’, if the
 * data is not needed.
 *
 * A permutation of the matrix (or vector) nonzeros is given by the
 * permutation vector ‘nzperm’, which must be of length ‘nnz’. Each
 * integer in the range from ‘0’ to ‘nnz-1’ should appear exactly
 * once. On success, the value of ‘rowidx[i]’ prior to applying the
 * permutation will be equal to ‘rowidx[nzperm[i]]’ after applying the
 * permutation, and the same holds for ‘colidx’ and ‘vals’.
 */
ACG_API int mtxfile_permutenzs(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * vals,
    const int64_t * nzperm);

/**
 * ‘mtxfile_permutenzs_rowwise()’ permutes the nonzero entries of a
 * matrix or vector so they are grouped rowwise, as required, for
 * instance, by the compressed sparse row (CSR) storage format.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
 *
 * The rows, columns and values of the underlying matrix or vector are
 * specified through the arrays ‘rowidx’, ‘colidx’ and ‘vals’,
 * respectively. Rows and columns use 1-based indexing if ‘idxbase’ is
 * ‘1’, and 0-based indexing if ‘idxbase’ is ‘0’. The length of the
 * ‘rowidx’ and ‘colidx’ arrays must be at least ‘nnz’, whereas the
 * length of the array ‘vals’ must be at least equal to ‘nnz’ times
 * ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file). Any of
 * the arrays may be set to ‘NULL’, if the data is not needed.
 *
 * If ‘rowptr’ is not ‘NULL’, then it must point to an array of length
 * ‘nrows+1’, which is used to store the row pointers of the permuted
 * matrix. In other words, the offset to the first nonzero in the
 * ‘i’-th row is given by ‘rowptr[i]’, and ‘rowptr[nrows]’ is equal to
 * the total number of nonzeros.
 *
 * If ‘nzperm’ is not ‘NULL’, then it must point to an array of length
 * ‘nnz’, which is used to store the permutation applied to the matrix
 * (or vector) nonzeros. Each integer in the range from ‘0’ to ‘nnz-1’
 * appears exactly once, such that the value of ‘rowidx[i]’ prior to
 * applying the permutation will be equal to ‘rowidx[nzperm[i]]’ after
 * applying the permutation. The same holds for ‘colidx’ and ‘vals’.
 */
ACG_API int mtxfile_permutenzs_rowwise(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * vals,
    int64_t * rowptr,
    int64_t * nzperm);

/**
 * ‘mtxfile_permutenzs_rowmajor()’ permutes the nonzero entries of a
 * matrix or vector so they are sorted in row major order.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
 *
 * The rows, columns and values of the underlying matrix or vector are
 * specified through the arrays ‘rowidx’, ‘colidx’ and ‘vals’,
 * respectively. Rows and columns use 1-based indexing if ‘idxbase’ is
 * ‘1’, and 0-based indexing if ‘idxbase’ is ‘0’. The length of the
 * ‘rowidx’ and ‘colidx’ arrays must be at least ‘nnz’, whereas the
 * length of the array ‘vals’ must be at least equal to ‘nnz’ times
 * ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file). Any of
 * the arrays may be set to ‘NULL’, if the data is not needed.
 *
 * If ‘rowptr’ is not ‘NULL’, then it must point to an array of length
 * ‘nrows+1’, which is used to store the row pointers of the permuted
 * matrix. In other words, the offset to the first nonzero in the
 * ‘i’-th row is given by ‘rowptr[i]’, and ‘rowptr[nrows]’ is equal to
 * the total number of nonzeros.
 *
 * If ‘nzperm’ is not ‘NULL’, then it must point to an array of length
 * ‘nnz’, which is used to store the permutation applied to the matrix
 * (or vector) nonzeros. Each integer in the range from ‘0’ to ‘nnz-1’
 * appears exactly once, such that the value of ‘rowidx[i]’ prior to
 * applying the permutation will be equal to ‘rowidx[nzperm[i]]’ after
 * applying the permutation. The same holds for ‘colidx’ and ‘vals’.
 */
ACG_API int mtxfile_permutenzs_rowmajor(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    int idxbase,
    acgidx_t * rowidx,
    acgidx_t * colidx,
    double * vals,
    int64_t * rowptr,
    int64_t * nzperm);

/*
 * MPI distributed-memory Matrix Market files
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘mtxfile_bcast_header()’ broadcasts a Matrix Market file header
 * from a process to all other processes in the same communicator.
 * This is a collective operation; it must be called by all processes
 * in the communicator.
 *
 * This is usually the first step in distributing a matrix or vector
 * among multiple processes when one of the processes has read the
 * Matrix Market data from a file.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’ and ‘nnz’.
 *
 * The rank of the MPI process that broadcasts the data is specified
 * by ‘root’. All other processes in the communicator ‘comm’ will
 * receive the data broadcasted.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int mtxfile_bcast_header(
    enum mtxobject * object,
    enum mtxformat * format,
    enum mtxfield * field,
    enum mtxsymmetry * symmetry,
    acgidx_t * nrows,
    acgidx_t * ncols,
    int64_t * nnz,
    int * nvalspernz,
    int root,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxfile_gatherv_double()’ gathers a Matrix Market data from all
 * processes in a given communicator to a specified process. Values
 * are stored as double-precision floating-point numbers.
 *
 * This is usually needed to collect a distributed a matrix or vector
 * to a single processes before writing it to a file.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’ and ‘nnz’. These values
 * should be the same on every process.
 *
 * Rows and columns use 1-based indexing if ‘idxbase’ is ‘1’, and
 * 0-based indexing if ‘idxbase’ is ‘0’. This value should also be the
 * same on every process.
 *
 * On each process, the rows, columns and values of the underlying
 * matrix or vector, which are to be gathered onto the root process,
 * are specified through the arrays ‘sendrowidx’, ‘sendcolidx’ and
 * ‘sendvals’, respectively. The length of the ‘sendrowidx’ and
 * ‘sendcolidx’ arrays must be at least ‘nnz’, whereas the length of
 * the array ‘sendvals’ must be at least equal to ‘nnz’ times
 * ‘nvalspernz’, (which depends on the object, format, field and
 * symmetry specified in the header of the Matrix Market file).
 *
 * On the root process, the arrays ‘recvrowidx’, ‘recvcolidx’ and
 * ‘recvvals’ are used to store the gathered rows, column and values,
 * respectively, of the underlying matrix or vector. Moreover, the
 * array ‘recvcounts’ contains the number of nonzeros to receive from
 * each process, whereas ‘displs’ contains the offset to the location
 * where the first nonzero to be received from each process will be
 * stored within the arrays ‘recvrowidx’, ‘recvcolidx’ and
 * ‘recvvals’. The displacement is expressed in the number of
 * nonzeros. The receive parameters are only used on the root process,
 * and they are ignored on non-root processes.
 *
 * The rank of the MPI process that gathers the data is specified by
 * ‘root’. All other processes in the communicator ‘comm’ will send
 * data to the root.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int mtxfile_gatherv_double(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    int idxbase,
    const acgidx_t * sendrowidx,
    const acgidx_t * sendcolidx,
    const double * sendvals,
    int sendcount,
    acgidx_t * recvrowidx,
    acgidx_t * recvcolidx,
    double * recvvals,
    const int * recvcounts,
    const int * displs,
    int root,
    MPI_Comm comm,
    int * mpierrcode);

/**
 * ‘mtxfile_scatterv_double()’ scatters a Matrix Market file from a
 * process across all processes in the same communicator. Values are
 * stored as double-precision floating-point numbers.
 *
 * This is usually the second step in distributing a matrix or vector
 * among multiple processes after one of the processes has read the
 * Matrix Market data from a file.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’ and ‘nnz’. These values
 * should be the same on every process.
 *
 * The value stored in ‘idxbase’ is broadcast from the root process to
 * all other processes in the communicator. Rows and columns use
 * 1-based indexing if ‘*idxbase’ is ‘1’, and 0-based indexing if
 * ‘*idxbase’ is ‘0’.
 *
 * The rows, columns and values of the underlying matrix or vector to
 * scatter from the root process are specified through the arrays
 * ‘sendrowidx’, ‘sendcolidx’ and ‘sendvals’, respectively. The length
 * of the ‘sendrowidx’ and ‘sendcolidx’ arrays must be at least ‘nnz’,
 * whereas the length of the array ‘sendvals’ must be at least equal
 * to ‘nnz’ times ‘nvalspernz’, (which depends on the object, format,
 * field and symmetry specified in the header of the Matrix Market
 * file). Any of the arrays may be set to ‘NULL’, if the data is not
 * needed. These values are only used on the root process and are
 * ignored on other processes.
 *
 * The array ‘sendcounts’ contains the number of nonzeros to send to
 * each process, whereas ‘displs’ contains the offset to the first
 * nonzero to be sent to each process. The displacement is expressed
 * in the number of nonzeros. These send parameters are only used on
 * the root process, and they are ignored on non-root processes.
 *
 * For each process in the communicator, the arrays ‘recvrowidx’,
 * ‘recvcolidx’ and ‘recvvals’ are used to store the scattered rows,
 * column and values, respectively, of the underlying matrix or
 * vector. The number of nonzeros to receive is given by ‘recvcount’.
 *
 * The rank of the MPI process that scatters the data is specified by
 * ‘root’. All other processes in the communicator ‘comm’ will receive
 * data from the root.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int mtxfile_scatterv_double(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t nnz,
    int nvalspernz,
    int * idxbase,
    const acgidx_t * sendrowidx,
    const acgidx_t * sendcolidx,
    const double * sendvals,
    const int * sendcounts,
    const int * displs,
    acgidx_t * recvrowidx,
    acgidx_t * recvcolidx,
    double * recvvals,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode);
#endif

#ifdef __cplusplus
}
#endif

#endif
