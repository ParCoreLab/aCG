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

#include "acg/config.h"
#include "acg/error.h"
#include "acg/fmtspec.h"
#include "acg/mtxfile.h"
#include "acg/halo.h"
#include "acg/sort.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef ACG_HAVE_LIBZ
#include <zlib.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <locale.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef ACG_IDX_SIZE
#define parse_acgidx_t parse_int
#elif ACG_IDX_SIZE == 32
#define parse_acgidx_t parse_int32_t
#elif ACG_IDX_SIZE == 64
#define parse_acgidx_t parse_int64_t
#else
#error "invalid ACG_IDX_SIZE; expected 32 or 64"
#endif

/*
 * Matrix Market data types
 */

const char * mtxobjectstr(enum mtxobject object) {
    if (object == mtxmatrix) { return "matrix"; }
    else if (object == mtxvector) { return "vector"; }
    else { return "unknown"; }
}

const char * mtxformatstr(enum mtxformat format) {
    if (format == mtxarray) { return "array"; }
    else if (format == mtxcoordinate) { return "coordinate"; }
    else { return "unknown"; }
}

const char * mtxfieldstr(enum mtxfield field) {
    if (field == mtxreal) { return "real"; }
    else if (field == mtxcomplex) { return "complex"; }
    else if (field == mtxinteger) { return "integer"; }
    else if (field == mtxpattern) { return "pattern"; }
    else { return "unknown"; }
}

const char * mtxsymmetrystr(enum mtxsymmetry symmetry) {
    if (symmetry == mtxgeneral) { return "general"; }
    else if (symmetry == mtxsymmetric) { return "symmetric"; }
    else if (symmetry == mtxskewsymmetric) { return "skew-symmetric"; }
    else if (symmetry == mtxhermitian) { return "hermitian"; }
    else { return "unknown"; }
}

const char * mtxdatatypestr(enum mtxdatatype datatype) {
    if (datatype == mtxint) { return "int"; }
    else if (datatype == mtxdouble) { return "double"; }
    else { return "unknown"; }
}

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
int mtxfile_nnz(
    enum mtxobject object,
    enum mtxformat format,
    enum mtxfield field,
    enum mtxsymmetry symmetry,
    acgidx_t nrows,
    acgidx_t ncols,
    int64_t * nnz)
{
    if (nrows < 0 || ncols < 0 || nnz < 0) return EINVAL;
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (symmetry == mtxgeneral) *nnz = nrows*ncols;
            else if (symmetry == mtxsymmetric) *nnz = nrows*(nrows+1)/2;
            else if (symmetry == mtxskewsymmetric) *nnz = nrows > 0 ? nrows*(nrows-1)/2 : 0;
            else return EINVAL;
        } else if (format == mtxcoordinate) { /* do nothing */ }
        else return EINVAL;
    } else if (object == mtxvector) {
        if (format == mtxarray) *nnz = nrows;
        else if (format == mtxcoordinate) { /* do nothing */ }
        else return EINVAL;
    } else return EINVAL;
    return 0;
}

/**
 * ‘mtxfile_nvalspernz()’ returns the number of values per nonzero in
 * a Matrix Market file with the given field.
 *
 * There is a single value per data line if ‘field’ is ‘real’ or
 * ‘integer’, and two values per data line if ‘field’ is ‘complex’.
 * If ‘field’ is ‘pattern’, then there are no values.
 */
int mtxfile_nvalspernz(
    enum mtxfield field,
    int * nvalspernz)
{
    if (field == mtxreal) *nvalspernz = 1;
    else if (field == mtxcomplex) *nvalspernz = 2;
    else if (field == mtxinteger) *nvalspernz = 1;
    else if (field == mtxpattern) *nvalspernz = 0;
    else return EINVAL;
    return 0;
}

/*
 * Matrix Market I/O
 */

/**
 * ‘freadline()’ reads a single line from a stream.
 */
static int freadline(
    char * linebuf,
    size_t linemax,
    FILE * f,
    size_t * len)
{
    char * s = fgets(linebuf, linemax+1, f);
    if (!s && feof(f)) return ACG_ERR_EOF;
    else if (!s) return ACG_ERR_ERRNO;
    int n = strlen(s);
    if (n > 0 && n == linemax && s[n-1] != '\n') return ACG_ERR_LINE_TOO_LONG;
    if (len) *len = n;
    return ACG_SUCCESS;
}

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    const char * s,
    char ** outendptr,
    int base,
    long long int * out_number,
    int64_t * nbytes)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return ACG_ERR_ERRNO;
    if (outendptr) *outendptr = endptr;
    if (nbytes) *nbytes += endptr - s;
    *out_number = number;
    return ACG_SUCCESS;
}

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
static int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * nbytes)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, nbytes);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) { errno = ERANGE; return ACG_ERR_ERRNO; }
    *x = y;
    return ACG_SUCCESS;
}

/**
 * ‘parse_int32_t()’ parses a string to produce a number that may be
 * represented as a signed, 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
static int parse_int32_t(
    int32_t * x,
    const char * s,
    char ** endptr,
    int64_t * nbytes)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, nbytes);
    if (err) return err;
    if (y < INT32_MIN || y > INT32_MAX) { errno = ERANGE; return ACG_ERR_ERRNO; }
    *x = y;
    return ACG_SUCCESS;
}

/**
 * ‘parse_int64_t()’ parses a string to produce a number that may be
 * represented as a signed, 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
static int parse_int64_t(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * nbytes)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, nbytes);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) { errno = ERANGE; return ACG_ERR_ERRNO; }
    *x = y;
    return ACG_SUCCESS;
}

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a double, ‘ERANGE’ is returned.
 */
static int parse_double(
    double * x,
    const char * s,
    char ** outendptr,
    int64_t * nbytes)
{
    errno = 0;
    char * endptr;
    *x = strtod(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
        (errno != 0 && x == 0)) { return ACG_ERR_ERRNO; }
    if (outendptr) *outendptr = endptr;
    if (nbytes) *nbytes += endptr - s;
    return ACG_SUCCESS;
}

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
int mtxfile_fread_header(
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
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return errno;
    }

    /* read and parse header line */
    size_t len;
    int err = freadline(linebuf, linemax, f, &len);
    if (err) { if (freelinebuf) free(linebuf); return err; }
    char * s = linebuf;
    char * t = s;
    if (strncmp("%%MatrixMarket ", t, strlen("%%MatrixMarket ")) == 0) {
        t += strlen("%%MatrixMarket ");
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        t += strlen("matrix ");
        *object = mtxmatrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        t += strlen("vector ");
        *object = mtxvector;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("array ", t, strlen("array ")) == 0) {
        t += strlen("array ");
        *format = mtxarray;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        t += strlen("coordinate ");
        *format = mtxcoordinate;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("real ", t, strlen("real ")) == 0) {
        t += strlen("real ");
        *field = mtxreal;
        *nvalspernz = 1;
    } else if (strncmp("complex ", t, strlen("complex ")) == 0) {
        t += strlen("complex ");
        *field = mtxcomplex;
        *nvalspernz = 2;
    } else if (strncmp("integer ", t, strlen("integer ")) == 0) {
        t += strlen("integer ");
        *field = mtxinteger;
        *nvalspernz = 1;
    } else if (strncmp("pattern ", t, strlen("pattern ")) == 0) {
        t += strlen("pattern ");
        *field = mtxpattern;
        *nvalspernz = 0;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
        *symmetry = mtxgeneral;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtxsymmetric;
    } else if (strncmp("skew-symmetric", t, strlen("skew-symmetric")) == 0) {
        t += strlen("skew-symmetric");
        *symmetry = mtxskewsymmetric;
    } else if (strncmp("hermitian", t, strlen("hermitian")) == 0 ||
               strncmp("Hermitian", t, strlen("Hermitian")) == 0) {
        t += strlen("hermitian");
        *symmetry = mtxhermitian;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }

    /* skip lines starting with '%' */
    do {
        t = linebuf + len-1;
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) *nbytes += t-s;
        s = t;
        if (nlines) (*nlines)++;
        err = freadline(linebuf, linemax, f, &len);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        s = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    if (*object == mtxmatrix && *format == mtxarray) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        err = parse_acgidx_t(ncols, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
        if (*symmetry == mtxgeneral) *nnz = (int64_t)(*nrows)*(*ncols);
        else if ((*symmetry == mtxsymmetric || *symmetry == mtxhermitian) && *nrows == *ncols) *nnz = (*nrows)*((*nrows)+1)/2;
        else if (*symmetry == mtxskewsymmetric && *nrows == *ncols) *nnz = (*nrows) > 0 ? (*nrows)*((*nrows)-1)/2 : 0;
        else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else if (*object == mtxmatrix && *format == mtxcoordinate) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        err = parse_acgidx_t(ncols, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        err = parse_int64_t(nnz, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
    } else if (*object == mtxvector && *format == mtxarray) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
        *ncols = 1;
        *nnz = *nrows;
    } else if (*object == mtxvector && *format == mtxcoordinate) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        *ncols = 1;
        err = parse_int64_t(nnz, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (freelinebuf) free(linebuf);
    return 0;
}

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
int mtxfile_fread_data_int(
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
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return errno;
    }

    size_t ret;
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else if (format == mtxcoordinate) {
            acgidx_t i, j;
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        ret = fread(rowidx, sizeof(*rowidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*rowidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*rowidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        ret = fread(colidx, sizeof(*colidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*colidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*colidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        double * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = fread(tmp, sizeof(*tmp), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*tmp);
                        if (ret != nnz) { free(tmp); return ACG_ERR_EOF; }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err  || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) { a[k*nvalspernz+0] = x; a[k*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        ret = fread(rowidx, sizeof(*rowidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*rowidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*rowidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        ret = fread(colidx, sizeof(*colidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*colidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*colidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        ret = fread(a, sizeof(*a), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*a);
                        if (ret != nnz*sizeof(*a)) return ACG_ERR_EOF;
                    }
                }
            } else if (field == mtxpattern) {
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a && nvalspernz > 0) a[k*nvalspernz] = 1;
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        double * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = fread(tmp, sizeof(*tmp), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*tmp);
                        if (ret != nnz) { free(tmp); return ACG_ERR_EOF; }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) { a[i*nvalspernz+0] = x; a[i*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        ret = fread(a, sizeof(*a), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*a);
                        if (ret != nnz) return ACG_ERR_EOF;
                    }
                }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else if (format == mtxcoordinate) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (freelinebuf) free(linebuf);
    return 0;
}

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
int mtxfile_fread_data_double(
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
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return errno;
    }

    size_t ret;
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0, k = 0; i < nrows; i++) {
                        for (acgidx_t j = 0; j < ncols; j++, k++) {
                            int err = freadline(linebuf, linemax, f, NULL);
                            if (err) { if (freelinebuf) free(linebuf); return err; }
                            char * s = linebuf, * t = s;
                            double x;
                            err = parse_double(&x, s, &t, nbytes);
                            if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                            if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                            if (rowidx) { rowidx[k] = i+idxbase; } if (colidx) { colidx[k] = j+idxbase; }
                            if (a) a[k*nvalspernz] = x;
                        }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0, k = 0; i < nrows; i++) {
                        for (acgidx_t j = 0; j < ncols; j++, k++) {
                            int err = freadline(linebuf, linemax, f, NULL);
                            if (err) { if (freelinebuf) free(linebuf); return err; }
                            char * s = linebuf, * t = s;
                            double x, y;
                            err = parse_double(&x, s, &t, nbytes);
                            if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                            if (nbytes) (*nbytes)++;
                            s = t+1;
                            err = parse_double(&y, s, &t, nbytes);
                            if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                            if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                            if (rowidx) { rowidx[k] = i+idxbase; } if (colidx) { colidx[k] = j+idxbase; }
                            if (a) { a[k*nvalspernz+0] = x; a[k*nvalspernz+1] = y; }
                        }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0, k = 0; i < nrows; i++) {
                        for (acgidx_t j = 0; j < ncols; j++, k++) {
                            int err = freadline(linebuf, linemax, f, NULL);
                            if (err) { if (freelinebuf) free(linebuf); return err; }
                            char * s = linebuf, * t = s;
                            int x;
                            err = parse_int(&x, s, &t, nbytes);
                            if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                            if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                            if (rowidx) { rowidx[k] = i+idxbase; } if (colidx) { colidx[k] = j+idxbase; }
                            if (a) a[k*nvalspernz] = x;
                        }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else if (format == mtxcoordinate) {
            acgidx_t i, j;
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        ret = fread(rowidx, sizeof(*rowidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*rowidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*rowidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        ret = fread(colidx, sizeof(*colidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*colidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*colidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        ret = fread(a, sizeof(*a), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*a);
                        if (ret != nnz) return ACG_ERR_EOF;
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) { a[k*nvalspernz+0] = x; a[k*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        ret = fread(rowidx, sizeof(*rowidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*rowidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*rowidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        ret = fread(colidx, sizeof(*colidx), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*colidx);
                        if (ret != nnz) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = fseek(f, SEEK_CUR, nnz*sizeof(*colidx)); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        int * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = fread(tmp, sizeof(*tmp), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*tmp);
                        if (ret != nnz) { free(tmp); return ACG_ERR_EOF; }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else if (field == mtxpattern) {
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a && nvalspernz > 0) a[k*nvalspernz] = 1;
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (a && nvalspernz != 1) { if (freelinebuf) free(linebuf); return EINVAL; }
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        ret = fread(a, sizeof(*a), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*a);
                        if (ret != nnz) return ACG_ERR_EOF;
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) { a[i*nvalspernz+0] = x; a[i*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = freadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        int * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = fread(tmp, sizeof(*tmp), nnz, f);
                        if (nbytes) *nbytes += ret*sizeof(*tmp);
                        if (ret != nnz) { free(tmp); return ACG_ERR_EOF; }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else if (format == mtxcoordinate) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (freelinebuf) free(linebuf);
    return 0;
}

/**
 * ‘validate_format_string()’ parses and validates a format string to
 * be used for outputting numerical values.
 */
static int validate_format_string(
    const char * fmtstr,
    int * fmtwidth)
{
    struct fmtspec format;
    const char * endptr;
    int err = fmtspec_parse(&format, fmtstr, &endptr);
    if (err) { errno = err; return ACG_ERR_ERRNO; }
    else if (*endptr != '\0') return ACG_ERR_INVALID_FORMAT_SPECIFIER;
    if (format.width == fmtspec_width_star ||
        format.precision == fmtspec_precision_star ||
        format.length != fmtspec_length_none ||
        ((format.specifier != fmtspec_e &&
          format.specifier != fmtspec_E &&
          format.specifier != fmtspec_f &&
          format.specifier != fmtspec_F &&
          format.specifier != fmtspec_g &&
          format.specifier != fmtspec_G)))
    {
        return ACG_ERR_INVALID_FORMAT_SPECIFIER;
    }
    if (fmtwidth && format.width > 0) *fmtwidth = format.width;
    return ACG_SUCCESS;
}

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
 * For matrices in arrary format, the matrix or vector values are
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
int mtxfile_fwrite_double(
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
    int64_t * nbytes)
{
    int err = ACG_SUCCESS;
    int olderrno;
    int64_t n = 0;

    /* check that each comment line begins with '%' */
    if (comments) {
        const char * s = comments;
        while (*s != '\0') {
            if (*s != '%') return ACG_ERR_MTX_INVALID_COMMENT;
            s++;
            while (*s != '\0' && *s != '\n') s++;
            if (*s == '\0') return ACG_ERR_MTX_INVALID_COMMENT;
            s++;
        }
    }

    /* validate format string */
    int fmtwidth = DBL_DIG+4;
    if (numfmt) {
        int err = validate_format_string(numfmt, &fmtwidth);
        if (err) return err;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, do not affect output. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale) return ACG_ERR_ERRNO;
    setlocale(LC_ALL, "C");

    /* 1. write the Matrix Market header */
    n += fprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        mtxobjectstr(object), mtxformatstr(format),
        mtxfieldstr(field), mtxsymmetrystr(symmetry));

    /* 2. write an optional comments section */
    if (comments) {
        int ret = fputs(comments, f);
        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (nbytes) *nbytes += strlen(comments);
    }

    /* 3. write the size line */
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            n += fprintf(f, "%"PRIdx" %"PRIdx"\n", nrows, ncols);
        } else if (format == mtxcoordinate) {
            n += fprintf(f, "%"PRIdx" %"PRIdx" %"PRId64"\n", nrows, ncols, nnz);
        } else { err = EINVAL; goto fwrite_exit; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            n += fprintf(f, "%"PRIdx"\n", nrows);
        } else if (format == mtxcoordinate) {
            n += fprintf(f, "%"PRIdx" %"PRId64"\n", nrows, nnz);
        } else { err = EINVAL; goto fwrite_exit; }
    } else { err = EINVAL; goto fwrite_exit; }

    /* 4. write the data lines */
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, "%.*g", DBL_DIG, vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else { err = ENOTSUP; goto fwrite_exit; }
        } else if (format == mtxcoordinate) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        n += fprintf(f, " %"PRIdx, colidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        n += fprintf(f, " %"PRIdx, colidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, "%.*g", DBL_DIG, vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else {
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) rowidx)[i] = rowidx[i]+(1-idxbase);
                fwrite(rowidx, sizeof(*rowidx), nnz, f);
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) rowidx)[i] = rowidx[i]+(idxbase-1);
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) colidx)[i] = colidx[i]+(1-idxbase);
                fwrite(colidx, sizeof(*colidx), nnz, f);
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) colidx)[i] = colidx[i]+(idxbase-1);
                fwrite(vals, sizeof(*vals), nvalspernz*nnz, f);
            }
        } else { err = EINVAL; goto fwrite_exit; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, "%.*g", DBL_DIG, vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else { err = ENOTSUP; goto fwrite_exit; }
        } else if (format == mtxcoordinate) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, "%.*g", DBL_DIG, vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else { err = ENOTSUP; goto fwrite_exit; }
        } else { err = EINVAL; goto fwrite_exit; }
    } else { err = EINVAL; goto fwrite_exit; }

fwrite_exit:
    if (nbytes) *nbytes += n;
    olderrno = errno;
    setlocale(LC_ALL, locale);
    errno = olderrno;
    free(locale);
    return err;
}

/**
 * ‘mtxfile_fwrite_int()’ writes a Matrix Market file to a standard
 * I/O stream. Values are given as integers.
 *
 * See also ‘mtxfile_fwrite_double()’.
 */
int mtxfile_fwrite_int(
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
    int64_t * nbytes)
{
    int err = ACG_SUCCESS;
    int olderrno;
    int64_t n = 0;

    /* check that each comment line begins with '%' */
    if (comments) {
        const char * s = comments;
        while (*s != '\0') {
            if (*s != '%') return ACG_ERR_MTX_INVALID_COMMENT;
            s++;
            while (*s != '\0' && *s != '\n') s++;
            if (*s == '\0') return ACG_ERR_MTX_INVALID_COMMENT;
            s++;
        }
    }

    /* validate format string */
    int fmtwidth = DBL_DIG+4;
    if (numfmt) {
        int err = validate_format_string(numfmt, &fmtwidth);
        if (err) return err;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, do not affect output. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale) return ACG_ERR_ERRNO;
    setlocale(LC_ALL, "C");

    /* 1. write the Matrix Market header */
    n += fprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        mtxobjectstr(object), mtxformatstr(format),
        mtxfieldstr(field), mtxsymmetrystr(symmetry));

    /* 2. write an optional comments section */
    if (comments) {
        int ret = fputs(comments, f);
        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (nbytes) *nbytes += strlen(comments);
    }

    /* 3. write the size line */
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            n += fprintf(f, "%"PRIdx" %"PRIdx"\n", nrows, ncols);
        } else if (format == mtxcoordinate) {
            n += fprintf(f, "%"PRIdx" %"PRIdx" %"PRId64"\n", nrows, ncols, nnz);
        } else { err = EINVAL; goto fwrite_exit; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            n += fprintf(f, "%"PRIdx"\n", nrows);
        } else if (format == mtxcoordinate) {
            n += fprintf(f, "%"PRIdx" %"PRId64"\n", nrows, nnz);
        } else { err = EINVAL; goto fwrite_exit; }
    } else { err = EINVAL; goto fwrite_exit; }

    /* 4. write the data lines */
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, "%d", vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else { err = ENOTSUP; goto fwrite_exit; }
        } else if (format == mtxcoordinate) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        n += fprintf(f, " %"PRIdx, colidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        n += fprintf(f, " %"PRIdx, colidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, "%d", vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else {
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) rowidx)[i] = rowidx[i]+(1-idxbase);
                fwrite(rowidx, sizeof(*rowidx), nnz, f);
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) rowidx)[i] = rowidx[i]+(idxbase-1);
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) colidx)[i] = colidx[i]+(1-idxbase);
                fwrite(colidx, sizeof(*colidx), nnz, f);
                for (int64_t i = 0; i < nnz; i++) ((acgidx_t *) colidx)[i] = colidx[i]+(idxbase-1);
                fwrite(vals, sizeof(*vals), nvalspernz*nnz, f);
            }
        } else { err = EINVAL; goto fwrite_exit; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        for (int j = 0; j < nvalspernz; j++) {
                            if (j > 0) { fputc(' ', f); n++; }
                            n += fprintf(f, "%d", vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else {
                fwrite(vals, sizeof(*vals), nvalspernz*nnz, f);
            }
        } else if (format == mtxcoordinate) {
            if (!binary) {
                if (numfmt) {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, numfmt, vals[i*nvalspernz+j]);
                        }
                        fputc('\n', f); n++;
                    }
                } else {
                    for (int64_t i = 0; i < nnz; i++) {
                        n += fprintf(f, "%"PRIdx, rowidx[i]+(1-idxbase));
                        for (int j = 0; j < nvalspernz; j++) {
                            fputc(' ', f); n++;
                            n += fprintf(f, "%d", vals[i]);
                        }
                        fputc('\n', f); n++;
                    }
                }
            } else { err = ENOTSUP; goto fwrite_exit; }
        } else { err = EINVAL; goto fwrite_exit; }
    } else { err = EINVAL; goto fwrite_exit; }

fwrite_exit:
    if (nbytes) *nbytes += n;
    olderrno = errno;
    setlocale(LC_ALL, locale);
    errno = olderrno;
    free(locale);
    return err;
}

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
int mtxfile_fwrite_mpi_double(
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
    int * mpierrcode)
{
    int commsize, rank;
    int err = MPI_Comm_size(comm, &commsize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Comm_rank(comm, &rank);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    /* count and gather vector sizes from each process */
    int * recvcounts = rank == root ? malloc(commsize*sizeof(int)) : NULL;
    if (rank == root && !recvcounts) err = ACG_ERR_ERRNO;
    err = acgerrmpi(comm, err, NULL, &errno, NULL);
    if (err) return err;

    int * displs = rank == root ? malloc(commsize*sizeof(int)) : NULL;
    if (rank == root && !displs) err = ACG_ERR_ERRNO;
    err = acgerrmpi(comm, err, NULL, &errno, NULL);
    if (err) {
        if (rank == root) free(recvcounts);
        return err;
    }

    int sendcount = npnz;
    err = MPI_Gather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; err = ACG_ERR_MPI; }
    err = acgerrmpi(comm, err, NULL, NULL, mpierrcode);
    if (err) {
        if (rank == root) { free(displs); free(recvcounts); }
        return err;
    }

    *nnz = 0;
    if (rank == root) {
        displs[0] = 0;
        for (int p = 1; p < commsize; p++) displs[p] = displs[p-1] + recvcounts[p-1];
        *nnz = displs[commsize-1] + recvcounts[commsize-1];
    }

    acgidx_t * sendrowidx = malloc(sendcount*sizeof(acgidx_t));
    if (!sendrowidx) { err = ACG_ERR_ERRNO; }
    err = acgerrmpi(comm, err, NULL, &errno, NULL);
    if (err) {
        if (rank == root) { free(displs); free(recvcounts); }
        return err;
    }
    if (prowidx && pnzrows) {
        for (int64_t k = 0; k < npnz; k++) { sendrowidx[k] = pnzrows[prowidx[k]]; }
    } else if (pnzrows) {
        for (int64_t k = 0; k < npnz; k++) sendrowidx[k] = pnzrows[k];
    } else if (prowidx) {
        for (int64_t k = 0; k < npnz; k++) sendrowidx[k] = prowidx[k];
    } else {
        for (int64_t k = 0; k < npnz; k++) sendrowidx[k] = k+idxbase;
    }

    acgidx_t * sendcolidx = NULL;
    if (pcolidx || pnzcols) {
        sendcolidx = malloc(sendcount*sizeof(acgidx_t));
        if (!sendcolidx) { err = ACG_ERR_ERRNO; }
        err = acgerrmpi(comm, err, NULL, &errno, NULL);
        if (err) {
            free(sendrowidx);
            if (rank == root) { free(displs); free(recvcounts); }
            return err;
        }
    }
    if (pnzcols && pcolidx) {
        for (int64_t k = 0; k < npnz; k++) sendcolidx[k] = pnzcols[pcolidx[k]];
    } else if (pnzcols) {
        for (int64_t k = 0; k < npnz; k++) sendcolidx[k] = pnzcols[k];
    } else if (pcolidx) {
        for (int64_t k = 0; k < npnz; k++) sendcolidx[k] = pcolidx[k];
    }

    /* if (verbose > 0 && rank == root) { */
    /*     fprintf(stderr, "nonzeros: %"PRId64"\n", *nnz); */
    /*     if (verbose > 1) { */
    /*         for (int p = 0; p < commsize; p++) */
    /*             fprintf(stderr, "    rank %'d nonzeros: %'d\n", p, recvcounts[p]); */
    /*     } */
    /* } */

    /* allocate storage for receiving matrix market data on each process */
    acgidx_t * recvrowidx = rank == root ? malloc(*nnz*sizeof(acgidx_t)) : NULL;
    if (rank == root && !recvrowidx) err = ACG_ERR_ERRNO;
    err = acgerrmpi(comm, err, NULL, &errno, NULL);
    if (err) {
        free(sendrowidx);
        if (rank == root) { free(displs); free(recvcounts); }
        return err;
    }
    acgidx_t * recvcolidx = NULL;
    if (pcolidx || pnzcols) {
        recvcolidx = rank == root ? malloc(*nnz*sizeof(acgidx_t)) : NULL;
        if (rank == root && !recvcolidx) err = ACG_ERR_ERRNO;
        err = acgerrmpi(comm, err, NULL, &errno, NULL);
        if (err) {
            if (rank == root) free(recvrowidx);
            free(sendcolidx); free(sendrowidx);
            if (rank == root) { free(displs); free(recvcounts); }
            return err;
        }
    }

    double * recvvals = rank == root ? malloc(*nnz*nvalspernz*sizeof(double)) : NULL;
    if (rank == root && !recvvals) err = ACG_ERR_ERRNO;
    err = acgerrmpi(comm, err, NULL, &errno, NULL);
    if (err) {
        if (rank == root) { free(recvcolidx); free(recvrowidx); }
        free(sendcolidx); free(sendrowidx);
        if (rank == root) { free(displs); free(recvcounts); }
        return err;
    }

    /* if (verbose > 0 && rank == root) { */
    /*     int64_t nrowidxbytes = *nnz*sizeof(acgidx_t); */
    /*     int64_t ncolidxbytes = pcolidx ? *nnz*sizeof(acgidx_t) : 0; */
    /*     int64_t nvalsbytes = *nnz*nvalspernz*sizeof(double); */
    /*     int64_t nbytes = nrowidxbytes+ncolidxbytes+nvalsbytes; */
    /*     fprintf(stderr, "storage allocated for gathering Matrix Market data" */
    /*             " rowidx: %'.1f MB colidx: %'.1f MB val: %'.1f MB" */
    /*             " total: %'.1f MB\n", */
    /*             1.0e-6*nrowidxbytes, 1.0e-6*ncolidxbytes, 1.0e-6*nvalsbytes, */
    /*             1.0e-6*(nrowidxbytes+ncolidxbytes+nvalsbytes)); */
    /* } */

    /* gather matrix market data */
    err = mtxfile_gatherv_double(
        object, format, field, symmetry,
        nrows, ncols, npnz, nvalspernz,
        idxbase, sendrowidx, sendcolidx, pvals, sendcount,
        recvrowidx, recvcolidx, recvvals, recvcounts, displs,
        root, comm, mpierrcode);
    if (err) {
        if (rank == root) { free(recvvals); free(recvcolidx); free(recvrowidx); }
        free(sendcolidx); free(sendrowidx);
        if (rank == root) { free(displs); free(recvcounts); }
        return err;
    }
    free(sendcolidx); free(sendrowidx); free(displs); free(recvcounts);

    err = mtxfile_compactnzs_unsorted(
        object, format, field, symmetry,
        nrows, ncols, nnz, nvalspernz,
        idxbase, recvrowidx, recvcolidx, recvvals, NULL);
    if (err) {
        if (rank == root) { free(recvvals); free(recvcolidx); free(recvrowidx); }
        return err;
    }

    if (rank == root) {
        /* if the output vector is in array format, we need to
         * sort the received values in ascending order of rows */
        int binary = 0;
        err = mtxfile_fwrite_double(
            f, binary, object, format, field, symmetry, comments,
            nrows, ncols, *nnz, nvalspernz,
            idxbase, recvrowidx, recvcolidx, recvvals,
            numfmt, nbytes);
        if (err) {
            err = acgerrmpi(comm, err, NULL, &errno, NULL);
            free(recvvals); free(recvcolidx); free(recvrowidx);
            return err;
        }
        free(recvvals); free(recvcolidx); free(recvrowidx);
        err = acgerrmpi(comm, err, NULL, &errno, NULL);
    } else {
        err = acgerrmpi(comm, err, NULL, &errno, NULL);
        if (err) return err;
    }
    return ACG_SUCCESS;
}
#endif

/*
 * Matrix Market I/O for gzip-compressed streams
 */

#ifdef ACG_HAVE_LIBZ
/**
 * ‘gzreadline()’ reads a single line from a stream.
 */
static int gzreadline(
    char * linebuf,
    size_t linemax,
    gzFile f,
    size_t * len)
{
    char * s = gzgets(f, linebuf, linemax+1);
    if (!s && gzeof(f)) return ACG_ERR_EOF;
    else if (!s) return ACG_ERR_ERRNO;
    size_t n = strlen(s);
    if (n > 0 && n == linemax && s[n-1] != '\n') return ACG_ERR_LINE_TOO_LONG;
    if (len) *len = n;
    return ACG_SUCCESS;
}

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
int mtxfile_gzread_header(
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
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return errno;
    }

    /* read and parse header line */
    size_t len;
    int err = gzreadline(linebuf, linemax, f, &len);
    if (err) { if (freelinebuf) free(linebuf); return err; }
    char * s = linebuf;
    char * t = s;
    if (strncmp("%%MatrixMarket ", t, strlen("%%MatrixMarket ")) == 0) {
        t += strlen("%%MatrixMarket ");
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("matrix ", t, strlen("matrix ")) == 0) {
        t += strlen("matrix ");
        *object = mtxmatrix;
    } else if (strncmp("vector ", t, strlen("vector ")) == 0) {
        t += strlen("vector ");
        *object = mtxvector;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("array ", t, strlen("array ")) == 0) {
        t += strlen("array ");
        *format = mtxarray;
    } else if (strncmp("coordinate ", t, strlen("coordinate ")) == 0) {
        t += strlen("coordinate ");
        *format = mtxcoordinate;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("real ", t, strlen("real ")) == 0) {
        t += strlen("real ");
        *field = mtxreal;
        *nvalspernz = 1;
    } else if (strncmp("complex ", t, strlen("complex ")) == 0) {
        t += strlen("complex ");
        *field = mtxcomplex;
        *nvalspernz = 2;
    } else if (strncmp("integer ", t, strlen("integer ")) == 0) {
        t += strlen("integer ");
        *field = mtxinteger;
        *nvalspernz = 1;
    } else if (strncmp("pattern ", t, strlen("pattern ")) == 0) {
        t += strlen("pattern ");
        *field = mtxpattern;
        *nvalspernz = 0;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (nbytes) *nbytes += t-s;
    s = t;
    if (strncmp("general", t, strlen("general")) == 0) {
        t += strlen("general");
        *symmetry = mtxgeneral;
    } else if (strncmp("symmetric", t, strlen("symmetric")) == 0) {
        t += strlen("symmetric");
        *symmetry = mtxsymmetric;
    } else if (strncmp("skew-symmetric", t, strlen("skew-symmetric")) == 0) {
        t += strlen("skew-symmetric");
        *symmetry = mtxskewsymmetric;
    } else if (strncmp("hermitian", t, strlen("hermitian")) == 0 ||
               strncmp("Hermitian", t, strlen("Hermitian")) == 0) {
        t += strlen("hermitian");
        *symmetry = mtxhermitian;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }

    /* skip lines starting with '%' */
    do {
        t = linebuf + len-1;
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) *nbytes += t-s;
        s = t;
        if (nlines) (*nlines)++;
        err = gzreadline(linebuf, linemax, f, &len);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        s = linebuf;
    } while (linebuf[0] == '%');

    /* parse size line */
    if (*object == mtxmatrix && *format == mtxarray) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        err = parse_acgidx_t(ncols, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
        if (*symmetry == mtxgeneral) *nnz = (int64_t)(*nrows)*(*ncols);
        else if ((*symmetry == mtxsymmetric || *symmetry == mtxhermitian) && *nrows == *ncols) *nnz = (*nrows)*((*nrows)+1)/2;
        else if (*symmetry == mtxskewsymmetric && *nrows == *ncols) *nnz = (*nrows) > 0 ? (*nrows)*((*nrows)-1)/2 : 0;
        else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else if (*object == mtxmatrix && *format == mtxcoordinate) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        err = parse_acgidx_t(ncols, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        err = parse_int64_t(nnz, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
    } else if (*object == mtxvector && *format == mtxarray) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
        *ncols = 1;
        *nnz = *nrows;
    } else if (*object == mtxvector && *format == mtxcoordinate) {
        err = parse_acgidx_t(nrows, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t || *t != ' ') { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        s = t+1;
        *ncols = 1;
        err = parse_int64_t(nnz, s, &t, nbytes);
        if (err) { if (freelinebuf) free(linebuf); return err; }
        if (s == t) { if (freelinebuf) free(linebuf); return EINVAL; }
        if (*t == '\n') { t++; } else { if (freelinebuf) free(linebuf); return EINVAL; }
        if (nbytes) (*nbytes)++;
        if (nlines) (*nlines)++;
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (freelinebuf) free(linebuf);
    return 0;
}

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
int mtxfile_gzread_data_int(
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
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return errno;
    }

    int ret;
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else if (format == mtxcoordinate) {
            acgidx_t i, j;
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        ret = gzread(f, rowidx, nnz*sizeof(*rowidx));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*rowidx)) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*rowidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        ret = gzread(f, colidx, nnz*sizeof(*colidx));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*colidx)) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*colidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        double * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = gzread(f, tmp, nnz*sizeof(*tmp));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*tmp)) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err  || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) { a[k*nvalspernz+0] = x; a[k*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        ret = gzread(f, rowidx, nnz*sizeof(*rowidx));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*rowidx)) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*rowidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        ret = gzread(f, colidx, nnz*sizeof(*colidx));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*colidx)) return ACG_ERR_EOF;
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*colidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        ret = gzread(f, a, nnz*sizeof(*a));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*a)) return ACG_ERR_EOF;
                    }
                }
            } else if (field == mtxpattern) {
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a && nvalspernz > 0) a[k*nvalspernz] = 1;
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        double * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = gzread(f, tmp, nnz*sizeof(*tmp));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*tmp)) { free(tmp); return ACG_ERR_EOF; }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) { a[i*nvalspernz+0] = x; a[i*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        ret = gzread(f, a, nnz*sizeof(*a));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*a)) return ACG_ERR_EOF;
                    }
                }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else if (format == mtxcoordinate) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (freelinebuf) free(linebuf);
    return 0;
}

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
int mtxfile_gzread_data_double(
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
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return errno;
    }

    int ret;
    if (object == mtxmatrix) {
        if (format == mtxarray) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else if (format == mtxcoordinate) {
            acgidx_t i, j;
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        size_t offset = 0, sz = nnz*sizeof(*rowidx);
                        while (sz > 0) {
                            int r = sz < INT_MAX ? sz : INT_MAX;
                            ret = gzread(f, ((char *) rowidx)+offset, r);
                            if (nbytes) *nbytes += ret;
                            if (ret != r) return ACG_ERR_EOF;
                            offset += r, sz -= r;
                        }
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*rowidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        size_t offset = 0, sz = nnz*sizeof(*colidx);
                        while (sz > 0) {
                            int r = sz < INT_MAX ? sz : INT_MAX;
                            ret = gzread(f, ((char *) colidx)+offset, r);
                            if (nbytes) *nbytes += ret;
                            if (ret != r) return ACG_ERR_EOF;
                            offset += r, sz -= r;
                        }
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*colidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        size_t offset = 0, sz = nnz*sizeof(*a);
                        while (sz > 0) {
                            int r = sz < INT_MAX ? sz : INT_MAX;
                            ret = gzread(f, ((char *) a)+offset, r);
                            if (nbytes) *nbytes += ret;
                            if (ret != r) return ACG_ERR_EOF;
                            offset += r, sz -= r;
                        }
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) { a[k*nvalspernz+0] = x; a[k*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a) a[k*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) {
                        size_t offset = 0, sz = nnz*sizeof(*rowidx);
                        while (sz > 0) {
                            int r = sz < INT_MAX ? sz : INT_MAX;
                            ret = gzread(f, ((char *) rowidx)+offset, r);
                            if (nbytes) *nbytes += ret;
                            if (ret != r) return ACG_ERR_EOF;
                            offset += r, sz -= r;
                        }
                        for (int64_t k = 0; k < nnz; k++) rowidx[k] = rowidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*rowidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (colidx) {
                        size_t offset = 0, sz = nnz*sizeof(*colidx);
                        while (sz > 0) {
                            int r = sz < INT_MAX ? sz : INT_MAX;
                            ret = gzread(f, ((char *) colidx)+offset, r);
                            if (nbytes) *nbytes += ret;
                            if (ret != r) return ACG_ERR_EOF;
                            offset += r, sz -= r;
                        }
                        for (int64_t k = 0; k < nnz; k++) colidx[k] = colidx[k]+(idxbase-1);
                    } else { ret = gzseek(f, nnz*sizeof(*colidx), SEEK_CUR); if (ret) return ACG_ERR_ERRNO; }
                    if (a) {
                        int * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        size_t offset = 0, sz = nnz*sizeof(*tmp);
                        while (sz > 0) {
                            int r = sz < INT_MAX ? sz : INT_MAX;
                            ret = gzread(f, ((char *) tmp)+offset, r);
                            if (nbytes) *nbytes += ret;
                            if (ret != r) return ACG_ERR_EOF;
                            offset += r, sz -= r;
                        }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else if (field == mtxpattern) {
                if (!binary) {
                    for (int64_t k = 0; k < nnz; k++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        err = parse_acgidx_t(&i, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_acgidx_t(&j, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[k] = i+(idxbase-1); } if (colidx) { colidx[k] = j+(idxbase-1); }
                        if (a && nvalspernz > 0) a[k*nvalspernz] = 1;
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else if (object == mtxvector) {
        if (format == mtxarray) {
            if (field == mtxreal) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        ret = gzread(f, a, nnz*sizeof(*a));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*a)) return ACG_ERR_EOF;
                    }
                }
            } else if (field == mtxcomplex) {
                if (a && nvalspernz < 2) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        double x, y;
                        err = parse_double(&x, s, &t, nbytes);
                        if (err || s == t || *t != ' ') { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (nbytes) (*nbytes)++;
                        s = t+1;
                        err = parse_double(&y, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) { a[i*nvalspernz+0] = x; a[i*nvalspernz+1] = y; }
                    }
                } else { if (freelinebuf) free(linebuf); return ACG_ERR_NOT_SUPPORTED; }
            } else if (field == mtxinteger) {
                if (a && nvalspernz < 1) return EINVAL;
                if (!binary) {
                    for (acgidx_t i = 0; i < nrows; i++) {
                        int err = gzreadline(linebuf, linemax, f, NULL);
                        if (err) { if (freelinebuf) free(linebuf); return err; }
                        char * s = linebuf, * t = s;
                        int x;
                        err = parse_int(&x, s, &t, nbytes);
                        if (err || s == t) { if (freelinebuf) free(linebuf); return err ? err : EINVAL; }
                        if (*t == '\n') { if (nbytes) (*nbytes)++; if (nlines) (*nlines)++; }
                        if (rowidx) { rowidx[i] = i+idxbase; } if (colidx) { colidx[i] = idxbase; }
                        if (a) a[i*nvalspernz] = x;
                    }
                } else {
                    if (rowidx) { for (acgidx_t i = 0; i < nrows; i++) rowidx[i] = i+idxbase; }
                    if (colidx) { for (acgidx_t i = 0; i < nrows; i++) colidx[i] = idxbase; }
                    if (a) {
                        int * tmp = malloc(nnz*sizeof(*tmp));
                        if (!tmp) return ACG_ERR_ERRNO;
                        ret = gzread(f, tmp, nnz*sizeof(*tmp));
                        if (nbytes) *nbytes += ret;
                        if (ret != nnz*sizeof(*tmp)) { free(tmp); return ACG_ERR_EOF; }
                        for (int64_t k = 0; k < nnz; k++) a[k*nvalspernz] = tmp[k];
                        free(tmp);
                    }
                }
            } else { if (freelinebuf) free(linebuf); return EINVAL; }
        } else if (format == mtxcoordinate) {
            if (freelinebuf) free(linebuf);
            return ACG_ERR_NOT_SUPPORTED;
        } else { if (freelinebuf) free(linebuf); return EINVAL; }
    } else { if (freelinebuf) free(linebuf); return EINVAL; }
    if (freelinebuf) free(linebuf);
    return 0;
}
#endif

/*
 * Matrix and vector partitioning
 */

/**
 * ‘mtxfile_partition_rowwise()’ performs a rowwise partitioning of
 * the nonzero entries of a matrix or vector.
 *
 * The Matrix Market file header is specified by ‘object’, ‘format’,
 * ‘field’, ‘symmetry’, ‘nrows’, ‘ncols’, ‘nnz’ and ‘nvalspernz’,
 * (which are usually obtained by calling ‘mtxfile_gzread_header()’).
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
int mtxfile_partition_rowwise(
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
    int64_t * nzperm)
{
    if (nzpartptr) { for (int p = 0; p <= nparts; p++) nzpartptr[p] = 0; }
    for (int64_t k = 0; k < nnz; k++) {
        acgidx_t i = rowidx[k]-idxbase;
        if (i < 0 || i >= nrows) return EINVAL;
        int p = rowpart[i];
        if (p < 0 || p >= nparts) return EINVAL;
        nzpart[k] = p;
        if (nzpartptr) nzpartptr[p+1]++;
    }
    if (nzpartptr) { for (int p = 1; p <= nparts; p++) nzpartptr[p] += nzpartptr[p-1]; }
    if (nzperm) {
        int * nzpartsorted = malloc(nnz*sizeof(int));
        if (!nzpartsorted) return errno;
        for (int64_t k = 0; k < nnz; k++) nzpartsorted[k] = nzpart[k];
        int err = acgradixsort_int(nnz, sizeof(int), nzpartsorted, nzperm, NULL);
        if (err) { free(nzpartsorted); return err; }
        free(nzpartsorted);
    }
    return 0;
}

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
int mtxfile_partition_columnwise(
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
    int64_t * nzperm)
{
    if (nzpartptr) { for (int p = 0; p <= nparts; p++) nzpartptr[p] = 0; }
    for (int64_t k = 0; k < nnz; k++) {
        acgidx_t i = colidx[k]-idxbase;
        if (i < 0 || i >= ncols) return EINVAL;
        int p = colpart[i];
        if (p < 0 || p >= nparts) return EINVAL;
        nzpart[k] = p;
        if (nzpartptr) nzpartptr[p+1]++;
    }
    if (nzpartptr) { for (int p = 1; p <= nparts; p++) nzpartptr[p] += nzpartptr[p-1]; }
    if (nzperm) {
        int * nzpartsorted = malloc(nnz*sizeof(int));
        if (!nzpartsorted) return errno;
        for (int64_t k = 0; k < nnz; k++) nzpartsorted[k] = nzpart[k];
        int err = acgradixsort_int(nnz, sizeof(int), nzpartsorted, nzperm, NULL);
        if (err) { free(nzpartsorted); return err; }
        free(nzpartsorted);
    }
    return 0;
}

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
int mtxfile_compact(
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
    acgidx_t ** nzcols)
{
    if (!rowidx && !colidx) {
        if (nnzrows) *nzrows = 0;
        if (nzrows) *nzrows = NULL;
        if (nnzcols) *nzcols = 0;
        if (nzcols) *nzcols = NULL;
        return 0;
    }

    int64_t * perm = malloc(nnz*sizeof(int64_t));
    if (!perm) return errno;
    acgidx_t * dstidx = malloc(nnz*sizeof(acgidx_t));
    if (!dstidx) { free(perm); return errno; }

    if (rowidx) {
        /* sort the row numbers of every nonzero */
        int err = acgradixsort_idx_t(nnz, sizeof(acgidx_t), rowidx, perm, NULL);
        if (err) { free(perm); return err; }

        /* count the number of unique rows and allocate storage */
        int64_t i = 0, k = 0;
        while (i < nnz) { k++; for (i++; i < nnz && rowidx[i] == rowidx[i-1]; i++) {} }
        if (nnzrows) *nnzrows = k;
        if (nzrows) {
            *nzrows = malloc(k*sizeof(acgidx_t));
            if (!*nzrows) { free(dstidx); free(perm); return errno; }
        }

        /* enumerate the unique rows */
        i = 0, k = 0;
        while (i < nnz) {
            if (nzrows) { (*nzrows)[k] = rowidx[i]; }
            dstidx[i] = k;
            for (i++; i < nnz && rowidx[i] == rowidx[i-1]; i++) { dstidx[i] = k; }
            k++;
        }

        /* renumber the rows based on the new ordering */
        for (int64_t i = 0; i < nnz; i++) rowidx[i] = dstidx[perm[i]]+idxbase;
    }

    if (colidx) {
        /* sort the column numbers of every nonzero */
        int err = acgradixsort_idx_t(nnz, sizeof(acgidx_t), colidx, perm, NULL);
        if (err) { if (rowidx && nzrows) free(*nzrows); free(dstidx); free(perm); return err; }

        /* count the number of unique columns and allocate storage */
        int64_t i = 0, k = 0;
        while (i < nnz) { k++; for (i++; i < nnz && colidx[i] == colidx[i-1]; i++) {} }
        if (nnzcols) *nnzcols = k;
        *nzcols = malloc(k*sizeof(acgidx_t));
        if (!*nzcols) { if (rowidx && nzrows) free(*nzrows); free(dstidx); free(perm); return errno; }

        /* enumerate the unique columns */
        i = 0, k = 0;
        while (i < nnz) {
            dstidx[i] = k; (*nzcols)[k++] = colidx[i];
            for (i++; i < nnz && colidx[i] == colidx[i-1]; i++) { dstidx[i] = k-1; }
        }

        /* renumber the columns based on the new ordering */
        for (int64_t i = 0; i < nnz; i++) colidx[i] = dstidx[perm[i]]+idxbase;
    }
    free(dstidx); free(perm);
    return 0;
}

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
int mtxfile_permutenzs(
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
    const int64_t * nzperm)
{
    for (int64_t k = 0; k < nnz; k++) {
        if (nzperm[k] < 0 || nzperm[k] >= nnz) return EINVAL;
    }
    acgidx_t * tmpidx = NULL;
    if (rowidx || colidx) {
        tmpidx = malloc(nnz*sizeof(acgidx_t));
        if (!tmpidx) return errno;
    }
    if (rowidx) {
        for (int64_t k = 0; k < nnz; k++) tmpidx[nzperm[k]] = rowidx[k];
        for (int64_t k = 0; k < nnz; k++) rowidx[k] = tmpidx[k];
    }
    if (colidx) {
        for (int64_t k = 0; k < nnz; k++) tmpidx[nzperm[k]] = colidx[k];
        for (int64_t k = 0; k < nnz; k++) colidx[k] = tmpidx[k];
    }
    if (rowidx || colidx) free(tmpidx);
    if (vals) {
        double * tmpvals = malloc(nnz*nvalspernz*sizeof(double));
        if (!tmpvals) return errno;
        for (int64_t k = 0; k < nnz; k++) {
            for (int l = 0; l < nvalspernz; l++)
                tmpvals[nzperm[k]*nvalspernz+l] = vals[k*nvalspernz+l];
        }
        for (int64_t k = 0; k < nnz; k++) {
            for (int l = 0; l < nvalspernz; l++)
                vals[k*nvalspernz+l] = tmpvals[k*nvalspernz+l];
        }
        free(tmpvals);
    }
    return 0;
}

/**
 * ‘mtxfile_compactnzs()’ sorts and merges duplicate nonzero entries
 * of a matrix or vector.
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
int mtxfile_compactnzs(
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
    int64_t * nzperm)
{
    if (rowidx && colidx) {
        acgidx_t i = 0, k = 0;
        while (i < *nnz) {
            if (nzperm) nzperm[i] = k;
            rowidx[k] = rowidx[i];
            colidx[k] = colidx[i];
            vals[k] = vals[i];
            for (i++; i < *nnz &&
                     rowidx[i] == rowidx[i-1] &&
                     colidx[i] == colidx[i-1]; i++)
            {
                if (nzperm) nzperm[i] = k;
                vals[k] += vals[i];
            }
            k++;
        }
        *nnz = k;
    } else if (rowidx) {
        acgidx_t i = 0, k = 0;
        while (i < *nnz) {
            if (nzperm) nzperm[i] = k;
            rowidx[k] = rowidx[i];
            vals[k] = vals[i];
            for (i++; i < *nnz && rowidx[i] == rowidx[i-1]; i++) {
                if (nzperm) nzperm[i] = k;
                vals[k] += vals[i];
            }
            k++;
        }
        *nnz = k;
    }
    return ACG_SUCCESS;
}

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
int mtxfile_compactnzs_unsorted(
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
    int64_t * nzperm)
{
    int64_t nnzorig = *nnz;
    int64_t * nzpermrowmajor = NULL;
    int64_t * nzpermcompact = NULL;
    if (nzperm) {
        nzpermrowmajor = malloc(nnzorig*sizeof(*nzpermrowmajor));
        if (!nzpermrowmajor) return ACG_ERR_ERRNO;
        nzpermcompact = malloc(nnzorig*sizeof(*nzpermcompact));
        if (!nzpermcompact) { free(nzpermrowmajor); return ACG_ERR_ERRNO; }
    }
    int err = mtxfile_permutenzs_rowmajor(
        object, format, field, symmetry,
        nrows, ncols, nnzorig, nvalspernz,
        idxbase, rowidx, colidx, vals,
        NULL, nzperm ? nzpermrowmajor : NULL);
    if (err) { free(nzpermcompact); free(nzpermrowmajor); return err; }
    err = mtxfile_compactnzs(
        object, format, field, symmetry,
        nrows, ncols, nnz, nvalspernz,
        idxbase, rowidx, colidx, vals,
        nzperm ? nzpermcompact : NULL);
    if (err) { free(nzpermcompact); free(nzpermrowmajor); return err; }
    if (nzperm) {
        for (int64_t k = 0; k < nnzorig; k++)
            nzperm[k] = nzpermcompact[nzpermrowmajor[k]];
        free(nzpermcompact); free(nzpermrowmajor);
    }
    return ACG_SUCCESS;
}

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
int mtxfile_permutenzs_rowwise(
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
    int64_t * nzperm)
{
    int freerowptr = !rowptr;
    if (!rowptr) {
        rowptr = malloc((nrows+1)*sizeof(int64_t));
        if (!rowptr) return errno;
    }
    int freenzperm = !nzperm;
    if (!nzperm) {
        nzperm = malloc(nnz*sizeof(int64_t));
        if (!nzperm) { if (freerowptr) free(rowptr); return errno; }
    }

    /* count nonzeros in each row and obtain a permutation that groups
     * nonzeros rowwise, then permute the nonzeros */
    for (acgidx_t i = 0; i <= nrows; i++) rowptr[i] = 0;
    for (int64_t k = 0; k < nnz; k++) nzperm[k] = rowptr[rowidx[k]-idxbase+1]++;
    for (acgidx_t i = 1; i <= nrows; i++) rowptr[i] += rowptr[i-1];
    for (int64_t k = 0; k < nnz; k++) nzperm[k] += rowptr[rowidx[k]-idxbase];
    if (freerowptr) free(rowptr);
    int err = mtxfile_permutenzs(
        object, format, field, symmetry,
        nrows, ncols, nnz, nvalspernz,
        rowidx, colidx, vals, nzperm);
    if (err) { if (freenzperm) free(nzperm); return err; }
    if (freenzperm) free(nzperm);
    return 0;
}

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
int mtxfile_permutenzs_rowmajor(
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
    int64_t * nzperm)
{
    if (rowidx && colidx) {
        int freenzperm = !nzperm;
        if (!nzperm) {
            nzperm = malloc(nnz*sizeof(int64_t));
            if (!nzperm) return ACG_ERR_ERRNO;
        }

        /* sort nonzeros by row and column */
        int err = acgradixsortpair_idx_t(
            nnz, sizeof(*rowidx), rowidx, sizeof(*colidx), colidx, nzperm, NULL);
        if (err) { if (freenzperm) free(nzperm); return err; }

        /* permute the nonzero values */
        if (vals) {
            double * tmpvals = malloc(nnz*nvalspernz*sizeof(double));
            if (!tmpvals) return errno;
            for (int64_t k = 0; k < nnz; k++) {
                for (int l = 0; l < nvalspernz; l++)
                    tmpvals[nzperm[k]*nvalspernz+l] = vals[k*nvalspernz+l];
            }
            for (int64_t k = 0; k < nnz; k++) {
                for (int l = 0; l < nvalspernz; l++)
                    vals[k*nvalspernz+l] = tmpvals[k*nvalspernz+l];
            }
            free(tmpvals);
        }
        if (freenzperm) free(nzperm);

        /* count nonzeros in each row */
        if (rowptr) {
            for (acgidx_t i = 0; i <= nrows; i++) rowptr[i] = 0;
            for (int64_t k = 0; k < nnz; k++) rowptr[rowidx[k]-idxbase+1]++;
            for (acgidx_t i = 1; i <= nrows; i++) rowptr[i] += rowptr[i-1];
        }
    } else if (rowidx) {
        return mtxfile_permutenzs_rowwise(
            object, format, field, symmetry,
            nrows, ncols, nnz, nvalspernz,
            idxbase, rowidx, colidx, vals,
            rowptr, nzperm);
    }
    return ACG_SUCCESS;
}

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
int mtxfile_bcast_header(
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
    int * mpierrcode)
{
    int err = MPI_Bcast(object, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(format, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(field, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(symmetry, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(nrows, 1, MPI_ACGIDX_T, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(ncols, 1, MPI_ACGIDX_T, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(nnz, 1, MPI_INT64_T, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Bcast(nvalspernz, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    return ACG_SUCCESS;
}

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
int mtxfile_gatherv_double(
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
    int * mpierrcode)
{
    int err;
    if (sendrowidx) {
        err = MPI_Gatherv(
            sendrowidx, sendcount, MPI_ACGIDX_T,
            recvrowidx, recvcounts, displs, MPI_ACGIDX_T, root, comm);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    }
    if (sendcolidx) {
        err = MPI_Gatherv(
            sendcolidx, sendcount, MPI_ACGIDX_T,
            recvcolidx, recvcounts, displs, MPI_ACGIDX_T, root, comm);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    }
    if (sendvals) {
        MPI_Datatype valtype;
        err = MPI_Type_contiguous(nvalspernz, MPI_DOUBLE, &valtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        MPI_Type_commit(&valtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        err = MPI_Gatherv(
            sendvals, sendcount, valtype,
            recvvals, recvcounts, displs, valtype, root, comm);
        if (err) {
            MPI_Type_free(&valtype);
            if (mpierrcode) *mpierrcode = err;
            return ACG_ERR_MPI;
        }
        MPI_Type_free(&valtype);
    }
    return ACG_SUCCESS;
}

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
int mtxfile_scatterv_double(
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
    int * mpierrcode)
{
    int err = MPI_Bcast(idxbase, 1, MPI_INT, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    if (recvrowidx) {
        err = MPI_Scatterv(
            sendrowidx, sendcounts, displs, MPI_ACGIDX_T,
            recvrowidx, recvcount, MPI_ACGIDX_T, root, comm);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    }
    if (recvcolidx) {
        err = MPI_Scatterv(
            sendcolidx, sendcounts, displs, MPI_ACGIDX_T,
            recvcolidx, recvcount, MPI_ACGIDX_T, root, comm);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    }
    if (recvvals) {
        MPI_Datatype valtype;
        err = MPI_Type_contiguous(nvalspernz, MPI_DOUBLE, &valtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        MPI_Type_commit(&valtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        err = MPI_Scatterv(
            sendvals, sendcounts, displs, valtype,
            recvvals, recvcount, valtype, root, comm);
        if (err) {
            MPI_Type_free(&valtype);
            if (mpierrcode) *mpierrcode = err;
            return ACG_ERR_MPI;
        }
        MPI_Type_free(&valtype);
    }
    return ACG_SUCCESS;
}
#endif

/*
 * memory management
 */

/**
 * ‘acgmtxfile_free()’ frees resources associated with a Matrix
 * Market file.
 */
void acgmtxfile_free(
    struct acgmtxfile * mtxfile)
{
    free(mtxfile->rowidx);
    free(mtxfile->colidx);
    free(mtxfile->data);
}

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
int acgmtxfile_alloc(
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
    enum mtxdatatype datatype)
{
    mtxfile->object = object;
    mtxfile->format = format;
    mtxfile->field = field;
    mtxfile->symmetry = symmetry;
    mtxfile->nrows = nrows;
    mtxfile->ncols = ncols;
    mtxfile->nnzs = nnzs;
    mtxfile->nvalspernz = nvalspernz;
    mtxfile->idxbase = idxbase;
    mtxfile->datatype = datatype;
    mtxfile->rowidx = malloc(mtxfile->nnzs*sizeof(*mtxfile->rowidx));
    if (!mtxfile->rowidx) return ACG_ERR_ERRNO;
    mtxfile->colidx = malloc(mtxfile->nnzs*sizeof(*mtxfile->colidx));
    if (!mtxfile->colidx) { free(mtxfile->rowidx); return ACG_ERR_ERRNO; }
    if (mtxfile->datatype == mtxint) {
        mtxfile->data = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(int));
    } else if (mtxfile->datatype == mtxdouble) {
        mtxfile->data = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(double));
    } else {
        free(mtxfile->colidx); free(mtxfile->rowidx);
        return ACG_ERR_INVALID_VALUE;
    }
    if (!mtxfile->data) {
        free(mtxfile->colidx); free(mtxfile->rowidx);
        return ACG_ERR_ERRNO;
    }
    return ACG_SUCCESS;
}

/**
 * ‘acgmtxfile_copy()’ creates a copy of a Matrix Market file.
 */
int acgmtxfile_copy(
    struct acgmtxfile * dst,
    const struct acgmtxfile * src)
{
    int err = acgmtxfile_alloc(
        dst, src->object, src->format, src->field, src->symmetry,
        src->nrows, src->ncols, src->nnzs, src->nvalspernz,
        src->idxbase, src->datatype);
    if (err) return err;
    for (int64_t k = 0; k < dst->nnzs; k++) {
        dst->rowidx[k] = src->rowidx[k];
        dst->colidx[k] = src->colidx[k];
    }
    if (src->datatype == mtxdouble) {
        double * dstdata = dst->data; const double * srcdata = src->data;
        for (int64_t k = 0; k < dst->nnzs; k++) {
            for (int l = 0; l < dst->nvalspernz; l++)
                dstdata[k*dst->nvalspernz+l] = srcdata[k*src->nvalspernz+l];
        }
    } else if (src->datatype == mtxint) {
        int * dstdata = dst->data; const int * srcdata = src->data;
        for (int64_t k = 0; k < dst->nnzs; k++) {
            for (int l = 0; l < dst->nvalspernz; l++)
                dstdata[k*dst->nvalspernz+l] = srcdata[k*src->nvalspernz+l];
        }
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}

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
int acgmtxfile_fread(
    struct acgmtxfile * mtxfile,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    enum mtxdatatype datatype,
    FILE * f,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return ACG_ERR_ERRNO;
    }

    /* read header */
    int err = mtxfile_fread_header(
        f, &mtxfile->object, &mtxfile->format, &mtxfile->field, &mtxfile->symmetry,
        &mtxfile->nrows, &mtxfile->ncols, &mtxfile->nnzs, &mtxfile->nvalspernz,
        nlines, nbytes, linemax, linebuf);
    if (err) {
        if (freelinebuf) free(linebuf);
        errno = err; return ACG_ERR_ERRNO;
    }

    /* allocate storage */
    mtxfile->idxbase = idxbase;
    mtxfile->datatype = datatype;
    mtxfile->rowidx = malloc(mtxfile->nnzs*sizeof(*mtxfile->rowidx));
    if (!mtxfile->rowidx) {
        if (freelinebuf) free(linebuf);
        return ACG_ERR_ERRNO;
    }
    mtxfile->colidx = malloc(mtxfile->nnzs*sizeof(*mtxfile->colidx));
    if (!mtxfile->colidx) {
        free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return ACG_ERR_ERRNO;
    }
    if (mtxfile->datatype == mtxint) {
        mtxfile->data = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(int));
    } else if (mtxfile->datatype == mtxdouble) {
        mtxfile->data = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(double));
    } else {
        free(mtxfile->colidx); free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return ACG_ERR_INVALID_VALUE;
    }
    if (!mtxfile->data) {
        free(mtxfile->colidx); free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return ACG_ERR_ERRNO;
    }

    /* read data */
    if (mtxfile->datatype == mtxint) {
        err = mtxfile_fread_data_int(
            f, mtxfile->object, mtxfile->format, mtxfile->field, mtxfile->symmetry,
            mtxfile->nrows, mtxfile->ncols, mtxfile->nnzs, mtxfile->nvalspernz,
            layout, binary, mtxfile->idxbase,
            mtxfile->rowidx, mtxfile->colidx, mtxfile->data,
            nlines, nbytes, linemax, linebuf);
    } else if (mtxfile->datatype == mtxdouble) {
        err = mtxfile_fread_data_double(
            f, mtxfile->object, mtxfile->format, mtxfile->field, mtxfile->symmetry,
            mtxfile->nrows, mtxfile->ncols, mtxfile->nnzs, mtxfile->nvalspernz,
            layout, binary, mtxfile->idxbase,
            mtxfile->rowidx, mtxfile->colidx, mtxfile->data,
            nlines, nbytes, linemax, linebuf);
    }
    if (err) {
        free(mtxfile->data); free(mtxfile->colidx); free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return err;
    }
    if (freelinebuf) free(linebuf);
    return ACG_SUCCESS;
}

#ifdef ACG_HAVE_LIBZ
/**
 * ‘acgmtxfile_gzread()’ reads a Matrix Market file from a
 * gzip-compressed stream.
 *
 * See also ‘acgmtxfile_fread()’.
 */
int acgmtxfile_gzread(
    struct acgmtxfile * mtxfile,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    enum mtxdatatype datatype,
    gzFile f,
    int64_t * nlines,
    int64_t * nbytes,
    long linemax,
    char * linebuf)
{
    bool freelinebuf = !linebuf;
    if (!linebuf) {
        linemax = sysconf(_SC_LINE_MAX);
        linebuf = malloc(linemax+1);
        if (!linebuf) return ACG_ERR_ERRNO;
    }

    /* read header */
    int err = mtxfile_gzread_header(
        f, &mtxfile->object, &mtxfile->format, &mtxfile->field, &mtxfile->symmetry,
        &mtxfile->nrows, &mtxfile->ncols, &mtxfile->nnzs, &mtxfile->nvalspernz,
        nlines, nbytes, linemax, linebuf);
    if (err) {
        if (freelinebuf) free(linebuf);
        errno = err; return ACG_ERR_ERRNO;
    }

    /* allocate storage */
    mtxfile->idxbase = idxbase;
    mtxfile->datatype = datatype;
    mtxfile->rowidx = malloc(mtxfile->nnzs*sizeof(*mtxfile->rowidx));
    if (!mtxfile->rowidx) {
        if (freelinebuf) free(linebuf);
        return ACG_ERR_ERRNO;
    }
    mtxfile->colidx = malloc(mtxfile->nnzs*sizeof(*mtxfile->colidx));
    if (!mtxfile->colidx) {
        free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return ACG_ERR_ERRNO;
    }
    if (mtxfile->datatype == mtxint) {
        mtxfile->data = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(int));
    } else if (mtxfile->datatype == mtxdouble) {
        mtxfile->data = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(double));
    } else {
        free(mtxfile->colidx); free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return ACG_ERR_INVALID_VALUE;
    }
    if (!mtxfile->data) {
        free(mtxfile->colidx); free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return ACG_ERR_ERRNO;
    }

    /* read data */
    if (mtxfile->datatype == mtxint) {
        err = mtxfile_gzread_data_int(
            f, mtxfile->object, mtxfile->format, mtxfile->field, mtxfile->symmetry,
            mtxfile->nrows, mtxfile->ncols, mtxfile->nnzs, mtxfile->nvalspernz,
            layout, binary, mtxfile->idxbase,
            mtxfile->rowidx, mtxfile->colidx, mtxfile->data,
            nlines, nbytes, linemax, linebuf);
    } else if (mtxfile->datatype == mtxdouble) {
        err = mtxfile_gzread_data_double(
            f, mtxfile->object, mtxfile->format, mtxfile->field, mtxfile->symmetry,
            mtxfile->nrows, mtxfile->ncols, mtxfile->nnzs, mtxfile->nvalspernz,
            layout, binary, mtxfile->idxbase,
            mtxfile->rowidx, mtxfile->colidx, mtxfile->data,
            nlines, nbytes, linemax, linebuf);
    }
    if (err) {
        free(mtxfile->data); free(mtxfile->colidx); free(mtxfile->rowidx);
        if (freelinebuf) free(linebuf);
        return err;
    }
    if (freelinebuf) free(linebuf);
    return ACG_SUCCESS;
}
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
int acgmtxfile_read(
    struct acgmtxfile * mtxfile,
    enum mtxlayout layout,
    int binary,
    int idxbase,
    enum mtxdatatype datatype,
    const char * path,
    bool gzip,
    int64_t * nlines,
    int64_t * nbytes)
{
    int err;
    *nlines = -1;
    *nbytes = 0;
    if (!gzip) {
        FILE * f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDIN_FILENO);
            if (fd == -1) return ACG_ERR_ERRNO;
            if ((f = fdopen(fd, "r")) == NULL) { close(fd); return ACG_ERR_ERRNO; }
        } else if ((f = fopen(path, "r")) == NULL) { return ACG_ERR_ERRNO; }
        *nlines = 0;
        err = acgmtxfile_fread(
            mtxfile, layout, binary, idxbase, datatype,
            f, nlines, nbytes, 0, NULL);
        if (err) { fclose(f); return err; }
        fclose(f);
    } else {
#ifdef ACG_HAVE_LIBZ
        gzFile f;
        if (strcmp(path, "-") == 0) {
            int fd = dup(STDIN_FILENO);
            if (fd == -1) return ACG_ERR_ERRNO;
            if ((f = gzdopen(fd, "r")) == NULL) { close(fd); return ACG_ERR_ERRNO; }
        } else if ((f = gzopen(path, "r")) == NULL) { return ACG_ERR_ERRNO; }
        *nlines = 0;
        err = acgmtxfile_gzread(
            mtxfile, layout, binary, idxbase, datatype,
            f, nlines, nbytes, 0, NULL);
        if (err) { gzclose(f); return err; }
        gzclose(f);
#else
        return ACG_ERR_LIBZ_NOT_SUPPORTED;
#endif
    }
    return ACG_SUCCESS;
}

/*
 * partitioned matrices and vectors
 */

/**
 * ‘acgmtxfilepartition_free()’ frees resources associated with a
 * partition of a Matrix Market file.
 */
void acgmtxfilepartition_free(
    struct acgmtxfilepartition * partition)
{
    free(partition->nzrows);
    free(partition->ownednzrownsegments);
    free(partition->ownednzrowsegments);
    free(partition->sharednzrowowners);
    free(partition->nzcols);
    free(partition->ownednzcolnsegments);
    free(partition->ownednzcolsegments);
    free(partition->sharednzcolowners);
}

/**
 * ‘acgmtxfilepartition_copy()’ creates a copy of a partition.
 */
int acgmtxfilepartition_copy(
    struct acgmtxfilepartition * dst,
    const struct acgmtxfilepartition * src)
{
    dst->part = src->part;
    dst->nrows = src->nrows;
    dst->ncols = src->ncols;
    dst->nnzs = src->nnzs;
    dst->nnzrows = src->nnzrows;
    dst->nexclusivenzrows = src->nexclusivenzrows;
    dst->nownednzrows = src->nownednzrows;
    dst->nsharednzrows = src->nsharednzrows;
    dst->nzrows = malloc(dst->nnzrows*sizeof(*dst->nzrows));
    if (!dst->nzrows) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->nnzrows; i++) dst->nzrows[i] = src->nzrows[i];
    dst->ownednzrownsegments = malloc(
        dst->nownednzrows*sizeof(*dst->ownednzrownsegments));
    if (!dst->ownednzrownsegments) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->nownednzrows; i++)
        dst->ownednzrownsegments[i] = src->ownednzrownsegments[i];
    dst->ownednzrowsegmentssize = src->ownednzrowsegmentssize;
    dst->ownednzrowsegments = malloc(
        dst->ownednzrowsegmentssize*sizeof(*dst->ownednzrowsegments));
    if (!dst->ownednzrowsegments) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->ownednzrowsegmentssize; i++)
        dst->ownednzrowsegments[i] = src->ownednzrowsegments[i];
    dst->sharednzrowowners = malloc(
        dst->nsharednzrows*sizeof(*dst->sharednzrowowners));
    for (acgidx_t i = 0; i < dst->nsharednzrows; i++)
        dst->sharednzrowowners[i] = src->sharednzrowowners[i];
    dst->nnzcols = src->nnzcols;
    dst->nexclusivenzcols = src->nexclusivenzcols;
    dst->nownednzcols = src->nownednzcols;
    dst->nsharednzcols = src->nsharednzcols;
    dst->nzcols = malloc(dst->nnzcols*sizeof(*dst->nzcols));
    if (!dst->nzcols) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->nnzcols; i++) dst->nzcols[i] = src->nzcols[i];
    dst->ownednzcolnsegments = malloc(
        dst->nownednzcols*sizeof(*dst->ownednzcolnsegments));
    if (!dst->ownednzcolnsegments) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->nownednzcols; i++)
        dst->ownednzcolnsegments[i] = src->ownednzcolnsegments[i];
    dst->ownednzcolsegmentssize = src->ownednzcolsegmentssize;
    dst->ownednzcolsegments = malloc(
        dst->ownednzcolsegmentssize*sizeof(*dst->ownednzcolsegments));
    if (!dst->ownednzcolsegments) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->ownednzcolsegmentssize; i++)
        dst->ownednzcolsegments[i] = src->ownednzcolsegments[i];
    dst->sharednzcolowners = malloc(
        dst->nsharednzcols*sizeof(*dst->sharednzcolowners));
    for (acgidx_t i = 0; i < dst->nsharednzcols; i++)
        dst->sharednzcolowners[i] = src->sharednzcolowners[i];
    dst->npnzs = src->npnzs;
    dst->nexclusivenzcolnzs = src->nexclusivenzcolnzs;
    dst->nownednzcolnzs = src->nownednzcolnzs;
    dst->nsharednzcolnzs = src->nsharednzcolnzs;
    return ACG_SUCCESS;
}

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
int acgmtxfile_partition_rowwise(
    const struct acgmtxfile * src,
    int nparts,
    const int * rowpart,
    const int * colpart,
    struct acgmtxfile * dsts,
    struct acgmtxfilepartition * parts)
{
    int err;

    /* check the partitioning vectors */
    for (int64_t k = 0; k < src->nnzs; k++) {
        acgidx_t i = src->rowidx[k]-src->idxbase;
        if (i < 0 || i >= src->nrows) return ACG_ERR_INVALID_VALUE;
        int p = rowpart[i];
        if (p < 0 || p >= nparts) return ACG_ERR_INVALID_VALUE;
        acgidx_t j = src->colidx[k]-src->idxbase;
        if (j < 0 || j >= src->ncols) return ACG_ERR_INVALID_VALUE;
        int q = colpart[j];
        if (q < 0 || q >= nparts) return ACG_ERR_INVALID_VALUE;
    }

    /* 1. create compressed sparse column representation */
    int idxbase = src->idxbase;
    acgidx_t * rowidx = malloc(src->nnzs*sizeof(*rowidx));
    if (!rowidx) return ACG_ERR_ERRNO;
    int64_t * colptr = malloc((src->ncols+1)*sizeof(*colptr));
    if (!colptr) return ACG_ERR_ERRNO;
    for (acgidx_t j = 0; j <= src->ncols; j++) colptr[j] = 0;
    for (int64_t k = 0; k < src->nnzs; k++) colptr[src->colidx[k]-idxbase+1]++;
    for (acgidx_t j = 1; j <= src->ncols; j++) colptr[j] += colptr[j-1];
    for (int64_t k = 0; k < src->nnzs; k++) {
        int64_t l = colptr[src->colidx[k]-idxbase];
        rowidx[l] = src->rowidx[k];
        colptr[src->colidx[k]-idxbase]++;
    }
    for (acgidx_t j = src->ncols; j > 0; j--) colptr[j] = colptr[j-1];
    colptr[0] = 0;

    /* 2. for each column, find its non-empty column segments (i.e.,
     * parts of the row partitioning with nonzeros in that column) */
    int * nnzcolsegments = malloc(src->ncols*sizeof(*nnzcolsegments));
    if (!nnzcolsegments) return ACG_ERR_ERRNO;
    acgidx_t * tmpnzcolsegments = malloc(src->nnzs*sizeof(*tmpnzcolsegments));
    if (!tmpnzcolsegments) return ACG_ERR_ERRNO;
    for (acgidx_t j = 0; j < src->ncols; j++) {
        nnzcolsegments[j] = 0;
        for (int64_t k = colptr[j]; k < colptr[j+1]; k++) {
            acgidx_t i = rowidx[k]-idxbase;
            int p = rowpart[i];
            bool insert = true;
            int64_t l = colptr[j];
            for (; l < colptr[j]+nnzcolsegments[j]; l++) {
                if (tmpnzcolsegments[l] == p) { insert = false; break; } }
            if (insert) { tmpnzcolsegments[l] = p; nnzcolsegments[j]++; }
        }
    }
    acgidx_t * nzcolsegmentsptr =
        malloc((src->ncols+1)*sizeof(*nzcolsegmentsptr));
    if (!nzcolsegmentsptr) return ACG_ERR_ERRNO;
    nzcolsegmentsptr[0] = 0;
    for (acgidx_t j = 1; j <= src->ncols; j++)
        nzcolsegmentsptr[j] = nzcolsegmentsptr[j-1] + nnzcolsegments[j-1];
    int * nzcolsegments =
        malloc(nzcolsegmentsptr[src->ncols]*sizeof(*nzcolsegments));
    if (!nzcolsegments) return ACG_ERR_ERRNO;
    for (acgidx_t j = 0, l = 0; j < src->ncols; j++) {
        for (int64_t k = colptr[j]; k < colptr[j]+nnzcolsegments[j]; k++, l++)
            nzcolsegments[l] = tmpnzcolsegments[k];
    }
    free(tmpnzcolsegments);

    /*
     * For each part, we consider three groups of non-empty columns:
     *
     *  1. columns owned exclusively by the given part, meaning that
     *     nonzeros are located only within the column segment of the
     *     given part and not in any other column segments
     *
     *  2. shared columns owned by the given part, meaning that one or
     *     more nonzeros are located in column segments corresponding
     *     to other parts of the row partitioning
     *
     *  3. shared columns not owned by the given part, such that one
     *     or more nonzeros are located in the column segment of the
     *     given part, but the column is owned by another part of the
     *     row partitioning
     *
     * Next, the nonzeros of each part are grouped by part, and then
     * within each part, we group them according to their columns
     * following the categories described above.
     */

    int * nzgroup = malloc(src->nnzs*sizeof(*nzgroup));
    if (!nzgroup) return ACG_ERR_ERRNO;
    int64_t * nzgroupptr = malloc((3*nparts+1)*sizeof(*nzgroupptr));
    for (int l = 0; l <= 3*nparts; l++) nzgroupptr[l] = 0;
    for (int64_t k = 0; k < src->nnzs; k++) {
        acgidx_t i = src->rowidx[k]-src->idxbase;
        acgidx_t j = src->colidx[k]-src->idxbase;
        int p = rowpart[i], q = colpart[j];
        if (nnzcolsegments[j] == 1) { nzgroup[k] = 3*p+0; nzgroupptr[3*p+1]++; }
        else if (p == q) { nzgroup[k] = 3*p+1; nzgroupptr[3*p+2]++; }
        else { nzgroup[k] = 3*p+2; nzgroupptr[3*p+3]++; }
    }
    for (int l = 1; l <= 3*nparts; l++) nzgroupptr[l] += nzgroupptr[l-1];
    int64_t * nzinvperm = malloc(src->nnzs*sizeof(*nzinvperm));
    if (!nzinvperm) return ACG_ERR_ERRNO;
    for (int64_t k = 0; k < src->nnzs; k++) {
        acgidx_t i = src->rowidx[k]-src->idxbase;
        acgidx_t j = src->colidx[k]-src->idxbase;
        int p = rowpart[i], q = colpart[j];
        if (nnzcolsegments[j] == 1) { nzinvperm[nzgroupptr[3*p]] = k; nzgroupptr[3*p]++; }
        else if (p == q) { nzinvperm[nzgroupptr[3*p+1]] = k; nzgroupptr[3*p+1]++; }
        else { nzinvperm[nzgroupptr[3*p+2]] = k; nzgroupptr[3*p+2]++; }
    }
    for (int l = 3*nparts; l > 0; l--) nzgroupptr[l] = nzgroupptr[l-1];
    nzgroupptr[0] = 0;

    /* extract partitioned matrices */
    for (int p = 0; p < nparts; p++) {

        /* allocate storage for partitioned Matrix Market file */
        struct acgmtxfile * dst = &dsts[p];
        int64_t npnzs = nzgroupptr[3*p+3]-nzgroupptr[3*p+0];
        err = acgmtxfile_alloc(
            dst, src->object, src->format, src->field, src->symmetry,
            src->nrows, src->ncols, npnzs, src->nvalspernz,
            src->idxbase, src->datatype);
        if (err) return err;

        /* gather nonzeros for the current part */
        for (int64_t l = 0; l < dst->nnzs; l++) {
            int64_t k = nzinvperm[nzgroupptr[3*p+0]+l];
            dst->rowidx[l] = src->rowidx[k];
            dst->colidx[l] = src->colidx[k];
        }
        if (src->datatype == mtxdouble) {
            double * dstdata = dst->data; const double * srcdata = src->data;
            for (int64_t l = 0; l < dst->nnzs; l++) {
                int64_t k = nzinvperm[nzgroupptr[3*p+0]+l];
                dstdata[l] = srcdata[k];
            }
        } else if (src->datatype == mtxint) {
            int * dstdata = dst->data; const int * srcdata = src->data;
            for (int64_t l = 0; l < dst->nnzs; l++) {
                int64_t k = nzinvperm[nzgroupptr[3*p+0]+l];
                dstdata[l] = srcdata[k];
            }
        } else { return ACG_ERR_INVALID_VALUE; }

        /* set up partition information */
        struct acgmtxfilepartition * dstpart = &parts[p];
        dstpart->nparts = nparts;
        dstpart->part = p;
        dstpart->nrows = src->nrows;
        dstpart->ncols = src->ncols;
        dstpart->nnzs = src->nnzs;

        /* count nonzeros */
        dstpart->nexclusivenzcolnzs = nzgroupptr[3*p+1]-nzgroupptr[3*p+0];
        dstpart->nownednzcolnzs = nzgroupptr[3*p+2]-nzgroupptr[3*p+1];
        dstpart->nsharednzcolnzs = nzgroupptr[3*p+3]-nzgroupptr[3*p+2];
        dstpart->npnzs = dstpart->nexclusivenzcolnzs
            + dstpart->nownednzcolnzs
            + dstpart->nsharednzcolnzs;

        /* count non-empty rows */
        acgidx_t * rowidxsorted = malloc(dst->nnzs*sizeof(*rowidxsorted));
        if (!rowidxsorted) return ACG_ERR_ERRNO;
        for (int64_t k = 0; k < dst->nnzs; k++) rowidxsorted[k] = dst->rowidx[k];
        int64_t * rowidxinvperm = malloc(dst->nnzs*sizeof(*rowidxinvperm));
        if (!rowidxinvperm) return ACG_ERR_ERRNO;
        err = acgradixsort_idx_t(
            dst->nnzs, sizeof(*rowidxsorted), rowidxsorted, NULL, rowidxinvperm);
        if (err) return err;
        dstpart->nnzrows = 0;
        dstpart->nexclusivenzrows = 0;
        dstpart->nownednzrows = 0;
        dstpart->nsharednzrows = 0;
        for (int64_t k = 0; k < dst->nnzs;) {
            for (k++; k < dst->nnzs && rowidxsorted[k] == rowidxsorted[k-1]; k++) {}
            dstpart->nexclusivenzrows++; dstpart->nnzrows++;
        }
        int64_t * nzrowidx = malloc(dstpart->nnzrows*sizeof(*nzrowidx));
        for (int64_t k = 0, l = 0; k < dst->nnzs;) {
            nzrowidx[l++] = rowidxinvperm[k];
            for (k++; k < dst->nnzs && rowidxsorted[k] == rowidxsorted[k-1]; k++) {}
        }
        err = acgradixsort_int64_t(
            dstpart->nnzrows, sizeof(*nzrowidx), nzrowidx, NULL, NULL);
        if (err) return err;
        dstpart->nzrows = malloc(dstpart->nnzrows*sizeof(*dstpart->nzrows));
        if (!dstpart->nzrows) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dstpart->nnzrows; i++)
            dstpart->nzrows[i] = dst->rowidx[nzrowidx[i]];

        /* renumber rows of the current part from idxbase up to
         * dstpart->nnzrows+idxbase */
        acgidx_t * nzrowssorted = malloc(dstpart->nnzrows*sizeof(*nzrowssorted));
        if (!nzrowssorted) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dstpart->nnzrows; i++) nzrowssorted[i] = dstpart->nzrows[i];
        int64_t * nzrowsinvperm = malloc(dstpart->nnzrows*sizeof(*nzrowsinvperm));
        if (!nzrowsinvperm) return ACG_ERR_ERRNO;
        err = acgradixsort_idx_t(
            dstpart->nnzrows, sizeof(*nzrowssorted), nzrowssorted, NULL, nzrowsinvperm);
        if (err) return err;
        int64_t k = 0;
        acgidx_t l = 0;
        while (k < dst->nnzs && l < dstpart->nnzrows) {
            if (rowidxsorted[k] == nzrowssorted[l]) {
                dst->rowidx[rowidxinvperm[k]] = nzrowsinvperm[l]+idxbase;
                k++;
            } else if (rowidxsorted[k] > nzrowssorted[l]) {
                l++;
            } else {
                fprintf(stderr, "%s:%d: part %'d of %'d:"
                        " nonzero row %'"PRIdx" not found\n",
                        __FILE__, __LINE__, p+1, nparts, rowidxsorted[k]);
                return ACG_ERR_INVALID_VALUE;
            }
        }
        free(nzrowsinvperm); free(nzrowssorted); free(nzrowidx);
        free(rowidxinvperm); free(rowidxsorted);

        /* only rowwise partitioning is supported for now, so there
         * are no for each shared rows */
        dstpart->ownednzrownsegments = NULL;
        dstpart->ownednzrowsegmentssize = 0;
        dstpart->ownednzrowsegments = NULL;
        dstpart->sharednzrowowners = NULL;

        /* count non-empty columns */
        acgidx_t * colidxsorted = malloc(dst->nnzs*sizeof(*colidxsorted));
        if (!colidxsorted) return ACG_ERR_ERRNO;
        for (int64_t k = 0; k < dst->nnzs; k++) colidxsorted[k] = dst->colidx[k];
        int64_t * colidxinvperm = malloc(dst->nnzs*sizeof(*colidxinvperm));
        if (!colidxinvperm) return ACG_ERR_ERRNO;
        err = acgradixsort_idx_t(
            dst->nnzs, sizeof(*colidxsorted), colidxsorted, NULL, colidxinvperm);
        if (err) return err;
        dstpart->nnzcols = 0;
        dstpart->nexclusivenzcols = 0;
        dstpart->nownednzcols = 0;
        dstpart->nsharednzcols = 0;
        for (int64_t k = 0; k < dst->nnzs;) {
            if (colidxinvperm[k] < dstpart->nexclusivenzcolnzs) { dstpart->nexclusivenzcols++; }
            else if (colidxinvperm[k] < dstpart->nexclusivenzcolnzs+dstpart->nownednzcolnzs) { dstpart->nownednzcols++; }
            else { dstpart->nsharednzcols++; }
            dstpart->nnzcols++;
            for (k++; k < dst->nnzs && colidxsorted[k] == colidxsorted[k-1]; k++) {}
        }
        int64_t * nzcolidx = malloc(dstpart->nnzcols*sizeof(*nzcolidx));
        for (int64_t k = 0, l = 0; k < dst->nnzs;) {
            nzcolidx[l++] = colidxinvperm[k];
            for (k++; k < dst->nnzs && colidxsorted[k] == colidxsorted[k-1]; k++) {}
        }
        err = acgradixsort_int64_t(
            dstpart->nnzcols, sizeof(*nzcolidx), nzcolidx, NULL, NULL);
        if (err) return err;
        dstpart->nzcols = malloc(dstpart->nnzcols*sizeof(*dstpart->nzcols));
        if (!dstpart->nzcols) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dstpart->nnzcols; i++)
            dstpart->nzcols[i] = dst->colidx[nzcolidx[i]];

        /* renumber columns of the current part from idxbase up to
         * dstpart->nnzcols+idxbase */
        acgidx_t * nzcolssorted = malloc(dstpart->nnzcols*sizeof(*nzcolssorted));
        if (!nzcolssorted) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dstpart->nnzcols; i++) nzcolssorted[i] = dstpart->nzcols[i];
        int64_t * nzcolsinvperm = malloc(dstpart->nnzcols*sizeof(*nzcolsinvperm));
        if (!nzcolsinvperm) return ACG_ERR_ERRNO;
        err = acgradixsort_idx_t(
            dstpart->nnzcols, sizeof(*nzcolssorted), nzcolssorted, NULL, nzcolsinvperm);
        if (err) return err;
        k = 0; l = 0;
        while (k < dst->nnzs && l < dstpart->nnzcols) {
            if (colidxsorted[k] == nzcolssorted[l]) {
                dst->colidx[colidxinvperm[k]] = nzcolsinvperm[l]+idxbase;
                k++;
            } else if (colidxsorted[k] > nzcolssorted[l]) {
                l++;
            } else {
                fprintf(stderr, "%s:%d: part %d of %d:"
                        " nonzero column %'"PRIdx" not found\n",
                        __FILE__, __LINE__, p+1, nparts, colidxsorted[k]);
                return ACG_ERR_INVALID_VALUE;
            }
        }
        free(nzcolsinvperm); free(nzcolssorted); free(nzcolidx);
        free(colidxinvperm); free(colidxsorted);

        /* for each shared, nonzero column owned by the current part,
         * find neigbhouring parts with nonzeros in that column */
        dstpart->ownednzcolnsegments =
            malloc(dstpart->nownednzcols*sizeof(*dstpart->ownednzcolnsegments));
        if (!dstpart->ownednzcolnsegments) return ACG_ERR_ERRNO;
        dstpart->ownednzcolsegmentssize = 0;
        for (acgidx_t i = 0; i < dstpart->nownednzcols; i++) {
            acgidx_t j = dstpart->nzcols[dstpart->nexclusivenzcols+i]-idxbase;
            dstpart->ownednzcolnsegments[i] =
                nzcolsegmentsptr[j+1]-nzcolsegmentsptr[j]-1;
            dstpart->ownednzcolsegmentssize += dstpart->ownednzcolnsegments[i];
        }
        dstpart->ownednzcolsegments = malloc(
            dstpart->ownednzcolsegmentssize*sizeof(*dstpart->ownednzcolsegments));
        if (!dstpart->ownednzcolsegments) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0, l = 0; i < dstpart->nownednzcols; i++) {
            acgidx_t j = dstpart->nzcols[dstpart->nexclusivenzcols+i]-idxbase;
            for (acgidx_t k = nzcolsegmentsptr[j]; k < nzcolsegmentsptr[j+1]; k++) {
                if (p == nzcolsegments[k]) continue;
                dstpart->ownednzcolsegments[l++] = nzcolsegments[k];
            }
        }

        /* for each shared nonzero column not owned by the current
         * part, find the part that owns the column */
        dstpart->sharednzcolowners =
            malloc(dstpart->nsharednzcols*sizeof(*dstpart->sharednzcolowners));
        if (!dstpart->sharednzcolowners) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dstpart->nsharednzcols; i++) {
            acgidx_t j = dstpart->nzcols[
                dstpart->nexclusivenzcols+dstpart->nownednzcols+i]-idxbase;
            dstpart->sharednzcolowners[i] = colpart[j];
        }
    }

    free(nzinvperm); free(nzgroupptr); free(nzgroup);
    free(nzcolsegments); free(nzcolsegmentsptr); free(nnzcolsegments);
    free(rowidx); free(colptr);
    return ACG_SUCCESS;
}

/*
 * communication for partitioned (and distributed) Matrix Market files
 */

/**
 * ‘acgmtxfilepartition_rowhalo()’ sets up a halo exchange
 * communication pattern to send and receive data associated with the
 * rows of a partitioned Matrix Market file.
 */
int acgmtxfilepartition_rowhalo(
    const struct acgmtxfilepartition * part,
    struct acghalo * halo)
{
    acgidx_t sendnodeoffset = part->nexclusivenzrows;
    acgidx_t nsendnodes = part->nownednzrows;
    const acgidx_t * sendnodetags = &part->nzrows[sendnodeoffset];
    const int * sendnodenneighbours = part->ownednzrownsegments;
    const int * sendnodeneighbours = part->ownednzrowsegments;
    acgidx_t recvnodeoffset = part->nexclusivenzrows+part->nownednzrows;
    acgidx_t nrecvnodes = part->nsharednzrows;
    const acgidx_t * recvnodetags = &part->nzrows[recvnodeoffset];
    const int * recvnodeparts = part->sharednzrowowners;
    int err = acghalo_init(
        halo, nsendnodes, sendnodetags, sendnodenneighbours, sendnodeneighbours,
        nrecvnodes, recvnodetags, recvnodeparts);
    if (err) return err;
    return ACG_SUCCESS;
}

/**
 * ‘acgmtxfilepartition_colhalo()’ sets up a halo exchange
 * communication pattern to send and receive data associated with the
 * columns of a partitioned Matrix Market file.
 */
int acgmtxfilepartition_colhalo(
    const struct acgmtxfilepartition * part,
    struct acghalo * halo)
{
    acgidx_t sendnodeoffset = part->nexclusivenzcols;
    acgidx_t nsendnodes = part->nownednzcols;
    const acgidx_t * sendnodetags = &part->nzcols[sendnodeoffset];
    const int * sendnodenneighbours = part->ownednzcolnsegments;
    const int * sendnodeneighbours = part->ownednzcolsegments;
    acgidx_t recvnodeoffset = part->nexclusivenzcols+part->nownednzcols;
    acgidx_t nrecvnodes = part->nsharednzcols;
    const acgidx_t * recvnodetags = &part->nzcols[recvnodeoffset];
    const int * recvnodeparts = part->sharednzcolowners;
    int err = acghalo_init(
        halo, nsendnodes, sendnodetags, sendnodenneighbours, sendnodeneighbours,
        nrecvnodes, recvnodetags, recvnodeparts);
    if (err) return err;
    return ACG_SUCCESS;
}

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
int acgmtxfile_permutenzs(
    struct acgmtxfile * mtxfile,
    const int64_t * nzperm)
{
    for (int64_t k = 0; k < mtxfile->nnzs; k++) {
        if (nzperm[k] < 0 || nzperm[k] >= mtxfile->nnzs) return ACG_ERR_INVALID_VALUE;
    }
    acgidx_t * tmpidx = malloc(mtxfile->nnzs*sizeof(*tmpidx));
    if (!tmpidx) return ACG_ERR_ERRNO;
    if (mtxfile->rowidx) {
        for (int64_t k = 0; k < mtxfile->nnzs; k++) tmpidx[nzperm[k]] = mtxfile->rowidx[k];
        for (int64_t k = 0; k < mtxfile->nnzs; k++) mtxfile->rowidx[k] = tmpidx[k];
    }
    if (mtxfile->colidx) {
        for (int64_t k = 0; k < mtxfile->nnzs; k++) tmpidx[nzperm[k]] = mtxfile->colidx[k];
        for (int64_t k = 0; k < mtxfile->nnzs; k++) mtxfile->colidx[k] = tmpidx[k];
    }
    free(tmpidx);

    if (mtxfile->datatype == mtxdouble) {
        double * data = mtxfile->data;
        double * tmpdata = malloc(mtxfile->nnzs*mtxfile->nvalspernz*sizeof(*tmpdata));
        if (!tmpdata) return ACG_ERR_ERRNO;
        for (int64_t k = 0; k < mtxfile->nnzs; k++) {
            for (int l = 0; l < mtxfile->nvalspernz; l++)
                tmpdata[nzperm[k]*mtxfile->nvalspernz+l] = 
                    data[k*mtxfile->nvalspernz+l];
        }
        for (int64_t k = 0; k < mtxfile->nnzs; k++) {
            for (int l = 0; l < mtxfile->nvalspernz; l++)
                data[k*mtxfile->nvalspernz+l] = tmpdata[k*mtxfile->nvalspernz+l];
        }
        free(tmpdata);
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}

/*
 * MPI distributed-memory Matrix Market files
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgmtxfile_scatterv()’ scatters a Matrix Market file from a root
 * process to all processes in a communicator. This is a collective
 * operation; it must be called by all processes in the communicator.
 *
 * The rank of the MPI process responsible for scattering the data is
 * specified by ‘root’. All other processes in the communicator ‘comm’
 * will receive data from the root process.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acgmtxfile_scatterv(
    const struct acgmtxfile * sendmtx,
    const int * sendcounts,
    const int * displs,
    struct acgmtxfile * recvmtx,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int rank;
    int err = MPI_Comm_rank(comm, &rank);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Scatterv(
        sendmtx->rowidx, sendcounts, displs, MPI_ACGIDX_T,
        recvmtx->rowidx, recvcount, MPI_ACGIDX_T, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Scatterv(
        sendmtx->colidx, sendcounts, displs, MPI_ACGIDX_T,
        recvmtx->colidx, recvcount, MPI_ACGIDX_T, root, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    MPI_Datatype sendtype = MPI_DATATYPE_NULL;
    if (rank == root) {
        if (sendmtx->datatype == mtxint) {
            err = MPI_Type_contiguous(sendmtx->nvalspernz, MPI_INT, &sendtype);
            if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        } else if (sendmtx->datatype == mtxdouble) {
            err = MPI_Type_contiguous(sendmtx->nvalspernz, MPI_DOUBLE, &sendtype);
            if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        } else { return ACG_ERR_INVALID_VALUE; }
        err = MPI_Type_commit(&sendtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    }
    MPI_Datatype recvtype;
    if (recvmtx->datatype == mtxint) {
        err = MPI_Type_contiguous(recvmtx->nvalspernz, MPI_INT, &recvtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    } else if (recvmtx->datatype == mtxdouble) {
        err = MPI_Type_contiguous(recvmtx->nvalspernz, MPI_DOUBLE, &recvtype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    } else { return ACG_ERR_INVALID_VALUE; }
    err = MPI_Type_commit(&recvtype);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    err = MPI_Scatterv(
        sendmtx->data, sendcounts, displs, sendtype,
        recvmtx->data, recvcount, recvtype, root, comm);
    if (err) {
        MPI_Type_free(&recvtype);
        if (rank == root) MPI_Type_free(&sendtype);
        if (mpierrcode) *mpierrcode = err;
        return ACG_ERR_MPI;
    }
    MPI_Type_free(&recvtype);
    if (rank == root) MPI_Type_free(&sendtype);
    return ACG_SUCCESS;
}

static int mtxdatatype_to_MPI_Type(
    MPI_Datatype * mpitype,
    enum mtxdatatype datatype,
    int nvalspernz,
    int * mpierrcode)
{
    int err;
    if (datatype == mtxint) {
        err = MPI_Type_contiguous(nvalspernz, MPI_INT, mpitype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    } else if (datatype == mtxdouble) {
        err = MPI_Type_contiguous(nvalspernz, MPI_DOUBLE, mpitype);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    } else { return ACG_ERR_INVALID_VALUE; }
    err = MPI_Type_commit(mpitype);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    return ACG_SUCCESS;
}

/**
 * ‘acgmtxfile_send()’ sends a Matrix Market file to another MPI
 * process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘acgmtxfile_recv()’.
 */
int acgmtxfile_send(
    const struct acgmtxfile * mtxfile,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    err = MPI_Send(&mtxfile->object, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->format, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->field, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->symmetry, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->nrows, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->ncols, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->nnzs, 1, MPI_INT64_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->idxbase, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->nvalspernz, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&mtxfile->datatype, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(mtxfile->rowidx, mtxfile->nnzs, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(mtxfile->colidx, mtxfile->nnzs, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    MPI_Datatype mpitype;
    err = mtxdatatype_to_MPI_Type(
        &mpitype, mtxfile->datatype, mtxfile->nvalspernz, mpierrcode);
    if (err) return err;
    err = MPI_Send(mtxfile->data, mtxfile->nnzs, mpitype, dest, tag, comm);
    MPI_Type_free(&mpitype);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    return ACG_SUCCESS;
}

/**
 * ‘acgmtxfile_recv()’ receives a Matrix Market file from another MPI
 * process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘acgmtxfile_send()’.
 */
int acgmtxfile_recv(
    struct acgmtxfile * mtxfile,
    int src,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    enum mtxobject object;
    err = MPI_Recv(&object, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    enum mtxformat format;
    err = MPI_Recv(&format, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    enum mtxfield field;
    err = MPI_Recv(&field, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    enum mtxsymmetry symmetry;
    err = MPI_Recv(&symmetry, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    acgidx_t nrows;
    err = MPI_Recv(&nrows, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    acgidx_t ncols;
    err = MPI_Recv(&ncols, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    int64_t nnzs;
    err = MPI_Recv(&nnzs, 1, MPI_INT64_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    int idxbase;
    err = MPI_Recv(&idxbase, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    int nvalspernz;
    err = MPI_Recv(&nvalspernz, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    enum mtxdatatype datatype;
    err = MPI_Recv(&datatype, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = acgmtxfile_alloc(
        mtxfile, object, format, field, symmetry,
        nrows, ncols, nnzs, nvalspernz, idxbase, datatype);
    if (err) return err;
    err = MPI_Recv(mtxfile->rowidx, mtxfile->nnzs, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { acgmtxfile_free(mtxfile); if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(mtxfile->colidx, mtxfile->nnzs, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { acgmtxfile_free(mtxfile); if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    MPI_Datatype mpitype;
    err = mtxdatatype_to_MPI_Type(
        &mpitype, mtxfile->datatype, mtxfile->nvalspernz, mpierrcode);
    if (err) { acgmtxfile_free(mtxfile); return err; }
    err = MPI_Recv(mtxfile->data, mtxfile->nnzs, mpitype, src, tag, comm, MPI_STATUS_IGNORE);
    MPI_Type_free(&mpitype);
    if (err) { acgmtxfile_free(mtxfile); if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    return ACG_SUCCESS;
}

/**
 * ‘acgmtxfilepartition_send()’ sends partitioning information to
 * another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to
 * ‘acgmtxfilepartition_recv()’.
 */
int acgmtxfilepartition_send(
    const struct acgmtxfilepartition * part,
    int dest,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    err = MPI_Send(&part->part, 1, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nrows, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->ncols, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nnzs, 1, MPI_INT64_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nnzrows, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nexclusivenzrows, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nownednzrows, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nsharednzrows, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->nzrows, part->nnzrows, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->ownednzrownsegments, part->nownednzrows, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->ownednzrowsegmentssize, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->ownednzrowsegments, part->ownednzrowsegmentssize, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->sharednzrowowners, part->nsharednzrows, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nnzcols, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nexclusivenzcols, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nownednzcols, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nsharednzcols, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->nzcols, part->nnzcols, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->ownednzcolnsegments, part->nownednzcols, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->ownednzcolsegmentssize, 1, MPI_ACGIDX_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->ownednzcolsegments, part->ownednzcolsegmentssize, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(part->sharednzcolowners, part->nsharednzcols, MPI_INT, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->npnzs, 1, MPI_INT64_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nexclusivenzcolnzs, 1, MPI_INT64_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nownednzcolnzs, 1, MPI_INT64_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Send(&part->nsharednzcolnzs, 1, MPI_INT64_T, dest, tag, comm);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    return ACG_SUCCESS;
}

/**
 * ‘acgmtxfilepartition_recv()’ receives partitioning information
 * from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘acgmtxfilepartition_send()’.
 */
int acgmtxfilepartition_recv(
    struct acgmtxfilepartition * part,
    int src,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    err = MPI_Recv(&part->part, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nrows, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->ncols, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nnzs, 1, MPI_INT64_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nnzrows, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nexclusivenzrows, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nownednzrows, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nsharednzrows, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->nzrows = malloc(part->nnzrows*sizeof(*part->nzrows));
    if (!part->nzrows) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->nzrows, part->nnzrows, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->ownednzrownsegments = malloc(part->nownednzrows*sizeof(*part->ownednzrownsegments));
    if (!part->ownednzrownsegments) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->ownednzrownsegments, part->nownednzrows, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->ownednzrowsegmentssize, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->ownednzrowsegments = malloc(part->ownednzrowsegmentssize*sizeof(*part->ownednzrowsegments));
    if (!part->ownednzrowsegments) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->ownednzrowsegments, part->ownednzrowsegmentssize, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->sharednzrowowners = malloc(part->nsharednzrows*sizeof(*part->sharednzrowowners));
    if (!part->sharednzrowowners) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->sharednzrowowners, part->nsharednzrows, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nnzcols, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nexclusivenzcols, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nownednzcols, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nsharednzcols, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->nzcols = malloc(part->nnzcols*sizeof(*part->nzcols));
    if (!part->nzcols) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->nzcols, part->nnzcols, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->ownednzcolnsegments = malloc(part->nownednzcols*sizeof(*part->ownednzcolnsegments));
    if (!part->ownednzcolnsegments) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->ownednzcolnsegments, part->nownednzcols, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->ownednzcolsegmentssize, 1, MPI_ACGIDX_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->ownednzcolsegments = malloc(part->ownednzcolsegmentssize*sizeof(*part->ownednzcolsegments));
    if (!part->ownednzcolsegments) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->ownednzcolsegments, part->ownednzcolsegmentssize, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    part->sharednzcolowners = malloc(part->nsharednzcols*sizeof(*part->sharednzcolowners));
    if (!part->sharednzcolowners) return ACG_ERR_ERRNO;
    err = MPI_Recv(part->sharednzcolowners, part->nsharednzcols, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->npnzs, 1, MPI_INT64_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nexclusivenzcolnzs, 1, MPI_INT64_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nownednzcolnzs, 1, MPI_INT64_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Recv(&part->nsharednzcolnzs, 1, MPI_INT64_T, src, tag, comm, MPI_STATUS_IGNORE);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    return ACG_SUCCESS;
}
#endif
