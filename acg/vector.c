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

#include "acg/config.h"
#include "acg/error.h"
#include "acg/fmtspec.h"
#include "acg/sort.h"
#include "acg/vector.h"

#include <errno.h>

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

/*
 * memory management
 */

/**
 * ‘acgvector_init_empty()’ initialises an empty vector.
 */
void acgvector_init_empty(
    struct acgvector * x)
{
    x->size = 0;
    x->x = NULL;
    x->num_nonzeros = 0;
    x->idxbase = 0;
    x->idx = NULL;
    x->num_ghost_nonzeros = 0;
}

/**
 * ‘acgvector_free()’ frees storage allocated for a vector.
 */
void acgvector_free(
    struct acgvector * x)
{
    free(x->idx);
    free(x->x);
}

/**
 * ‘acgvector_alloc_copy()’ allocates a copy of a vector without
 * initialising the values.
 */
int acgvector_alloc_copy(
    struct acgvector * dst,
    const struct acgvector * src)
{
    dst->size = src->size;
    dst->num_nonzeros = src->num_nonzeros;
    dst->idxbase = src->idxbase;
    if (src->idx) {
        dst->idx = malloc(dst->num_nonzeros*sizeof(*dst->idx));
        if (!dst->idx) return ACG_ERR_ERRNO;
        #pragma omp parallel for
        for (acgidx_t i = 0; i < dst->num_nonzeros; i++) dst->idx[i] = src->idx[i];
    } else { dst->idx = NULL; }
    dst->x = malloc((dst->idx ? dst->num_nonzeros : dst->size)*sizeof(*dst->x));
    if (!dst->x) { free(dst->idx); return ACG_ERR_ERRNO; }
    dst->num_ghost_nonzeros = src->num_ghost_nonzeros;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_init_copy()’ allocates a copy of a vector and also
 * copies the values.
 */
int acgvector_init_copy(
    struct acgvector * dst,
    const struct acgvector * src)
{
    int err = acgvector_alloc_copy(dst, src);
    if (err) return err;
    err = acgvector_copy(dst, src, NULL);
    if (err) { acgvector_free(dst); return err; }
    return ACG_SUCCESS;
}

/*
 * initialise vectors in full storage format
 */

/**
 * ‘acgvector_alloc()’ allocates a vector.
 */
int acgvector_alloc(
    struct acgvector * x,
    acgidx_t size)
{
    x->size = size;
    x->x = malloc(size * sizeof(*x->x));
    if (!x->x) return ACG_ERR_ERRNO;
    x->num_nonzeros = size;
    x->idxbase = 0;
    x->idx = NULL;
    x->num_ghost_nonzeros = 0;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_init_real_double()’ allocates and initialises a vector
 * with real, double precision coefficients.
 */
int acgvector_init_real_double(
    struct acgvector * x,
    acgidx_t size,
    const double * data)
{
    int err = acgvector_alloc(x, size);
    if (err) return err;
    #pragma omp parallel for
    for (acgidx_t k = 0; k < size; k++) x->x[k] = data[k];
    return ACG_SUCCESS;
}

/*
 * initialise vectors in packed storage format
 */

/**
 * ‘acgvector_alloc_packed()’ allocates a vector in packed storage
 * format.
 */
int acgvector_alloc_packed(
    struct acgvector * x,
    acgidx_t size,
    acgidx_t num_nonzeros,
    int idxbase,
    const acgidx_t * idx)
{
#ifndef NDEBUG
    for (acgidx_t k = 0; k < num_nonzeros; k++) {
        if (idx[k] < idxbase || idx[k] >= size+idxbase) {
            free(x->idx); free(x->x);
            return ACG_ERR_INDEX_OUT_OF_BOUNDS;
        }
    }
#endif

    x->size = size;
    x->x = malloc((idx ? num_nonzeros : size)*sizeof(*x->x));
    if (!x->x) return ACG_ERR_ERRNO;
    x->num_nonzeros = num_nonzeros;
    x->idxbase = idxbase;
    if (idx) {
        x->idx = malloc(num_nonzeros*sizeof(*x->idx));
        if (!x->idx) { free(x->x); return ACG_ERR_ERRNO; }
        #pragma omp parallel for
        for (acgidx_t k = 0; k < num_nonzeros; k++) {
            x->idx[k] = idx[k];
        }
    } else { x->idx = NULL; }
    x->num_ghost_nonzeros = 0;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_init_packed_real_double()’ allocates and initialises a
 * vector with real, double precision coefficients.
 */
int acgvector_init_packed_real_double(
    struct acgvector * x,
    acgidx_t size,
    acgidx_t num_nonzeros,
    int idxbase,
    const acgidx_t * idx,
    const double * data)
{
    int err = acgvector_alloc_packed(x, size, num_nonzeros, idxbase, idx);
    if (err) return err;
    for (acgidx_t k = 0; k < num_nonzeros; k++) x->x[k] = data[k];
    return ACG_SUCCESS;
}

/*
 * modifying values
 */

/**
 * ‘acgvector_setzero()’ sets every value of a vector to zero.
 */
int acgvector_setzero(
    struct acgvector * x)
{
    acgidx_t nnzs = x->idx ? x->num_nonzeros : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) x->x[k] = 0;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_set_constant_real_double()’ sets every value of a
 * vector equal to a constant, double precision floating point number.
 */
int acgvector_set_constant_real_double(
    struct acgvector * x,
    double a)
{
    acgidx_t nnzs = x->idx ? x->num_nonzeros : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) x->x[k] = a;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_set_real_double()’ sets values of a vector based on an
 * array of double precision floating point numbers.
 */
int acgvector_set_real_double(
    struct acgvector * x,
    acgidx_t size,
    const double * a,
    bool include_ghosts)
{
    acgidx_t xsize = x->idx
        ? x->num_nonzeros - (!include_ghosts ? x->num_ghost_nonzeros : 0) : x->size;
    if (xsize != size) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    double * xdata = x->x;
    for (acgidx_t i = 0; i < size; i++) xdata[i] = a[i];
    return ACG_SUCCESS;
}

/*
 * convert to and from Matrix Market format
 */

/**
 * ‘validate_format_string()’ parses and validates a format string to
 * be used for outputting numerical values of a Matrix Market file.
 */
static int validate_format_string(
    const char * fmtstr)
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
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_fwrite()’ writes a Matrix Market file to a stream.
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
int acgvector_fwrite(
    const struct acgvector * x,
    FILE * f,
    const char * comments,
    const char * fmt,
    int64_t * bytes_written)
{
    int err = ACG_SUCCESS;
    int olderrno;

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

    /* check for a valid format string */
    if (fmt) {
        err = validate_format_string(fmt);
        if (err) return err;
    }

    /* Set the locale to "C" to ensure that locale-specific settings,
     * such as the type of decimal point, do not affect output. */
    char * locale;
    locale = strdup(setlocale(LC_ALL, NULL));
    if (!locale) return ACG_ERR_ERRNO;
    setlocale(LC_ALL, "C");

    int ret = fprintf(
        f, "%%%%MatrixMarket %s %s %s %s\n",
        "vector", !x->idx ? "array" : "coordinate", "real", "general");
    if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
    if (bytes_written) *bytes_written += ret;

    if (comments) {
        ret = fputs(comments, f);
        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (bytes_written) *bytes_written += strlen(comments);
    }

    if (!x->idx) {
        ret = fprintf(f, "%"PRIdx"\n", x->size);
        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (bytes_written) *bytes_written += ret;
    } else {
        ret = fprintf(f, "%"PRIdx" %"PRIdx"\n", x->size, x->num_nonzeros);
        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (bytes_written) *bytes_written += ret;
    }

    if (!x->idx) {
        if (fmt) {
            for (acgidx_t k = 0; k < x->size; k++) {
                ret = fprintf(f, fmt, x->x[k]);
                if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) *bytes_written += ret;
                ret = fputc('\n', f);
                if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) (*bytes_written)++;
            }
        } else {
            for (acgidx_t k = 0; k < x->size; k++) {
                ret = fprintf(f, "%.*g", DBL_DIG, x->x[k]);
                if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) *bytes_written += ret;
                ret = fputc('\n', f);
                if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) (*bytes_written)++;
            }
        }
    } else {
        if (fmt) {
            for (acgidx_t k = 0; k < x->num_nonzeros; k++) {
                ret = fprintf(f, "%"PRIdx" ", x->idx[k]-x->idxbase+1);
                if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) *bytes_written += ret;
                ret = fprintf(f, fmt, x->x[k]);
                if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) *bytes_written += ret;
                ret = fputc('\n', f);
                if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) (*bytes_written)++;
            }
        } else {
            for (acgidx_t k = 0; k < x->num_nonzeros; k++) {
                ret = fprintf(f, "%"PRIdx" ", x->idx[k]-x->idxbase+1);
                if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) *bytes_written += ret;
                ret = fprintf(f, "%.*g", DBL_DIG, x->x[k]);
                if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) *bytes_written += ret;
                ret = fputc('\n', f);
                if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                if (bytes_written) (*bytes_written)++;
            }
        }
    }

fwrite_exit:
    olderrno = errno;
    setlocale(LC_ALL, locale);
    errno = olderrno;
    free(locale);
    return err;
}

/*
 * Level 1 BLAS operations
 */

/**
 * ‘acgvector_swap()’ swaps values of two vectors, simultaneously
 * performing ‘y <- x’ and ‘x <- y’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
int acgvector_swap(
    struct acgvector * x,
    struct acgvector * y,
    int64_t * num_bytes)
{
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    if ((x->idx ? x->num_nonzeros : x->size) !=
        (y->idx ? y->num_nonzeros : y->size))
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    double * xdata = x->x;
    double * ydata = y->x;
    acgidx_t nnzs = x->idx ? x->num_nonzeros : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) {
        double z = ydata[k];
        ydata[k] = xdata[k];
        xdata[k] = z;
    }
    if (num_bytes) *num_bytes += nnzs*(sizeof(*xdata)+sizeof(*ydata));
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_copy()’ copies values of a vector, ‘y = x’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
int acgvector_copy(
    struct acgvector * y,
    const struct acgvector * x,
    int64_t * num_bytes)
{
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    if ((x->idx ? x->num_nonzeros-x->num_ghost_nonzeros : x->size) !=
        (y->idx ? y->num_nonzeros-y->num_ghost_nonzeros : y->size))
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const double * xdata = x->x;
    double * ydata = y->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    #pragma omp parallel for simd
    for (acgidx_t k = 0; k < nnzs; k++) ydata[k] = xdata[k];
    if (num_bytes) *num_bytes += nnzs*(sizeof(*xdata)+sizeof(*ydata));
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_dscal()’ scales a vector by a double precision floating
 * point scalar, ‘x = a*x’.
 */
int acgvector_dscal(
    double a,
    struct acgvector * x,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    if (a == 1) return ACG_SUCCESS;
    double * xdata = x->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    if (a == 0) {
        for (acgidx_t k = 0; k < nnzs; k++) xdata[k] = 0;
    } else {
        for (acgidx_t k = 0; k < nnzs; k++) xdata[k] *= a;
        if (num_flops) *num_flops += nnzs;
    }
    if (num_bytes) *num_bytes += nnzs*sizeof(*xdata);
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_daxpy()’ adds a vector to another one multiplied by a
 * double precision floating point value, ‘y = a*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
int acgvector_daxpy(
    double a,
    const struct acgvector * x,
    struct acgvector * y,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    if (y->size != x->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    if ((x->idx ? x->num_nonzeros-x->num_ghost_nonzeros : x->size) !=
        (y->idx ? y->num_nonzeros-y->num_ghost_nonzeros : y->size))
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const double * xdata = x->x;
    double * ydata = y->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) ydata[k] += a*xdata[k];
    if (num_flops) *num_flops += 2*nnzs;
    if (num_bytes) *num_bytes += nnzs*(sizeof(*xdata)+sizeof(*ydata));
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_daypx()’ multiplies a vector by a double precision
 * floating point scalar and adds another vector, ‘y = a*y + x’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size.
 */
int acgvector_daypx(
    double a,
    struct acgvector * y,
    const struct acgvector * x,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    if (y->size != x->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    if ((x->idx ? x->num_nonzeros-x->num_ghost_nonzeros : x->size) !=
        (y->idx ? y->num_nonzeros-y->num_ghost_nonzeros : y->size))
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const double * xdata = x->x;
    double * ydata = y->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    for (acgidx_t k = 0; k < nnzs; k++)
        ydata[k] = a*ydata[k]+xdata[k];
    if (num_flops) *num_flops += 2*nnzs;
    if (num_bytes) *num_bytes += nnzs*(sizeof(*xdata)+sizeof(*ydata));
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_ddot()’ computes the Euclidean dot product of two
 * vectors in double precision floating point.
 *
 * The vectors ‘x’ and ‘y’ must be the same size. Furthermore, both
 * vectors are treated as if they are in full storage format.
 */
int acgvector_ddot(
    const struct acgvector * x,
    const struct acgvector * y,
    double * dot,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    if ((x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size) !=
        (y->idx ? (y->num_nonzeros-y->num_ghost_nonzeros) : y->size))
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const double * xdata = x->x;
    const double * ydata = y->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
#ifdef ACG_VECTOR_DDOT_NO_UNROLL
    double c = 0;
    for (acgidx_t k = 0; k < nnzs; k++) c += xdata[k]*ydata[k];
    *dot = c;
#else
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    for (acgidx_t k = 0; k < nnzs-nnzs%4; k+=4) {
        c1 += xdata[k+0]*ydata[k+0];
        c2 += xdata[k+1]*ydata[k+1];
        c3 += xdata[k+2]*ydata[k+2];
        c4 += xdata[k+3]*ydata[k+3];
    }
    for (acgidx_t k = nnzs-nnzs%4; k < nnzs; k++) c1 += xdata[k]*ydata[k];
    *dot = c1+c2+c3+c4;
#endif
    if (num_flops) *num_flops += 2*nnzs;
    if (num_bytes) *num_bytes += nnzs*(sizeof(*xdata)+sizeof(*ydata));
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_dnrm2()’ computes the Euclidean norm of a vector in
 * double precision floating point.
 */
int acgvector_dnrm2(
    const struct acgvector * x,
    double * nrm2,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    const double * xdata = x->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
#ifdef ACG_VECTOR_DNRM2_NO_UNROLL
    double c = 0;
    for (acgidx_t k = 0; k < nnzs; k++) c += xdata[k]*xdata[k];
    *nrm2 = sqrt(c);
#else
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    for (acgidx_t k = 0; k < nnzs-nnzs%4; k+=4) {
        c1 += xdata[k+0]*xdata[k+0];
        c2 += xdata[k+1]*xdata[k+1];
        c3 += xdata[k+2]*xdata[k+2];
        c4 += xdata[k+3]*xdata[k+3];
    }
    for (acgidx_t k = nnzs-nnzs%4; k < nnzs; k++) c1 += xdata[k]*xdata[k];
    *nrm2 = sqrt(c1+c2+c3+c4);
#endif
    if (num_flops) *num_flops += 2*nnzs+1;
    if (num_bytes) *num_bytes += nnzs*sizeof(*xdata);
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_dnrm2sqr()’ computes the square of the Euclidean norm
 * of a vector in double precision floating point.
 */
int acgvector_dnrm2sqr(
    const struct acgvector * x,
    double * nrm2sqr,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    const double * xdata = x->x;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
#ifdef ACG_VECTOR_DNRM2SQR_NO_UNROLL
    double c = 0;
    for (acgidx_t k = 0; k < nnzs; k++) c += xdata[k]*xdata[k];
    *nrm2sqr = c;
#else
    double c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    for (acgidx_t k = 0; k < nnzs-nnzs%4; k+=4) {
        c1 += xdata[k+0]*xdata[k+0];
        c2 += xdata[k+1]*xdata[k+1];
        c3 += xdata[k+2]*xdata[k+2];
        c4 += xdata[k+3]*xdata[k+3];
    }
    for (acgidx_t k = nnzs-nnzs%4; k < nnzs; k++) c1 += xdata[k]*xdata[k];
    *nrm2sqr = c1+c2+c3+c4;
#endif
    if (num_flops) *num_flops += 2*nnzs;
    if (num_bytes) *num_bytes += nnzs*sizeof(*xdata);
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_dasum()’ computes the sum of absolute values (1-norm)
 * of a vector in double precision floating point.
 */
int acgvector_dasum(
    const struct acgvector * x,
    double * asum,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    const double * xdata = x->x;
    double c = 0;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) c += fabs(xdata[k]);
    *asum = c;
    if (num_flops) *num_flops += nnzs;
    if (num_bytes) *num_bytes += nnzs*sizeof(*xdata);
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_iamax()’ finds the index of the first element having
 * the maximum absolute value.
 */
int acgvector_iamax(
    const struct acgvector * x,
    int * iamax)
{
    const double * xdata = x->x;
    *iamax = 0;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    double max = nnzs > 0 ? fabs(xdata[0]) : 0;
    for (acgidx_t k = 1; k < nnzs; k++) {
        if (max < fabs(xdata[k])) {
            max = fabs(xdata[k]);
            *iamax = k;
        }
    }
    return ACG_SUCCESS;
}

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
int acgvector_usddot(
    const struct acgvector * x,
    const struct acgvector * y,
    double * dot,
    int64_t * num_flops)
{
    if (!x->idx) return acgvector_ddot(x, y, dot, num_flops, NULL);
    if (y->idx) return ACG_ERR_VECTOR_EXPECTED_FULL;
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const acgidx_t * idx = x->idx;
    const double * xdata = x->x;
    const double * ydata = y->x;
    double c = 0;
    for (acgidx_t k = 0; k < x->num_nonzeros; k++)
        c += xdata[k]*ydata[idx[k]-y->idxbase];
    *dot = c;
    if (num_flops) *num_flops += 2*x->num_nonzeros;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_usdaxpy()’ performs a sparse vector update, multiplying
 * a sparse vector ‘x’ in packed form by a scalar ‘alpha’ and adding
 * the result to a vector ‘y’. That is, ‘y = alpha*x + y’.
 *
 * The vectors ‘x’ and ‘y’ must be of the same size size. Repeated
 * indices in the packed vector are allowed.
 */
int acgvector_usdaxpy(
    double alpha,
    const struct acgvector * x,
    struct acgvector * y,
    int64_t * num_flops)
{
    if (!x->idx) return acgvector_daxpy(alpha, x, y, num_flops, NULL);
    if (y->idx) return ACG_ERR_VECTOR_EXPECTED_FULL;
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const acgidx_t * idx = x->idx;
    const double * xdata = x->x;
    double * ydata = y->x;
    for (acgidx_t k = 0; k < x->num_nonzeros; k++)
        ydata[idx[k]-y->idxbase] += alpha*xdata[k];
    if (num_flops) *num_flops += 2*x->num_nonzeros;
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_usga()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are allowed.
 */
int acgvector_usga(
    struct acgvector * x,
    const struct acgvector * y)
{
    if (!x->idx) return ACG_ERR_VECTOR_EXPECTED_PACKED;
    if (y->idx) return ACG_ERR_VECTOR_EXPECTED_FULL;
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const acgidx_t * idx = x->idx;
    double * xdata = x->x;
    const double * ydata = y->x;
    #pragma omp parallel for
    for (acgidx_t k = 0; k < x->num_nonzeros; k++)
        xdata[k] = ydata[idx[k]-y->idxbase];
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_usgz()’ performs a gather operation from a vector ‘y’
 * into a sparse vector ‘x’ in packed form, while zeroing the values
 * of the source vector ‘y’ that were copied to ‘x’. Repeated indices
 * in the packed vector are allowed.
 */
int acgvector_usgz(
    struct acgvector * x,
    struct acgvector * y)
{
    if (!x->idx) return ACG_ERR_VECTOR_EXPECTED_PACKED;
    if (y->idx) return ACG_ERR_VECTOR_EXPECTED_FULL;
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const acgidx_t * idx = x->idx;
    double * xdata = x->x;
    double * ydata = y->x;
    for (acgidx_t k = 0; k < x->num_nonzeros; k++) {
        xdata[k] = ydata[idx[k]-y->idxbase];
        ydata[idx[k]-y->idxbase] = 0;
    }
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_ussc()’ performs a scatter operation to a vector ‘y’
 * from a sparse vector ‘x’ in packed form. Repeated indices in the
 * packed vector are not allowed, otherwise the result is undefined.
 */
int acgvector_ussc(
    struct acgvector * y,
    const struct acgvector * x)
{
    if (!x->idx) return ACG_ERR_VECTOR_EXPECTED_PACKED;
    if (y->idx) return ACG_ERR_VECTOR_EXPECTED_FULL;
    if (x->size != y->size) return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    const acgidx_t * idx = x->idx;
    const double * xdata = x->x;
    double * ydata = y->x;
    for (acgidx_t k = 0; k < x->num_nonzeros; k++)
        ydata[idx[k]-y->idxbase] = xdata[k];
    return ACG_SUCCESS;
}

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
 * belongs to exactly one process.
 */
int acgvector_ddotmpi(
    const struct acgvector * x,
    const struct acgvector * y,
    double * dot,
    int64_t * num_flops,
    int64_t * num_bytes,
    MPI_Comm comm,
    int * outmpierrcode)
{
    int err = acgvector_ddot(x, y, dot, num_flops, num_bytes);
    if ((err = acgerrmpi(comm, err, NULL, NULL, NULL))) return err;
    int mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, dot, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (mpierrcode) err = ACG_ERR_MPI;
    if ((err = acgerrmpi(comm, err, NULL, NULL, &mpierrcode))) {
        if (outmpierrcode) *outmpierrcode = mpierrcode;
        return err;
    }
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_dnrm2mpi()’ computes the Euclidean norm of a
 * distributed vector in double precision floating point.
 *
 * The distributed vector is assumed to be partitioned among processes
 * in the MPI communicator ‘comm’, meaning that every vector element
 * must belong to exactly one process.
 */
int acgvector_dnrm2mpi(
    const struct acgvector * x,
    double * nrm2,
    int64_t * num_flops,
    MPI_Comm comm,
    int * outmpierrcode)
{
    const double * xdata = x->x;
    double c = 0;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) c += xdata[k]*xdata[k];
    if (num_flops) *num_flops += 2*nnzs;
    int mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &c, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (mpierrcode) {
        if (outmpierrcode) *outmpierrcode = mpierrcode;
        return ACG_ERR_MPI;
    }
    *nrm2 = sqrt(c);
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_dasummpi()’ computes the sum of absolute values
 * (1-norm) of a distributed vector in double precision floating
 * point.
 *
 * The distributed vector is assumed to be partitioned among processes
 * in the MPI communicator ‘comm’, meaning that every vector element
 * must belong to exactly one process.
 */
int acgvector_dasummpi(
    const struct acgvector * x,
    double * asum,
    int64_t * num_flops,
    MPI_Comm comm,
    int * outmpierrcode)
{
    const double * xdata = x->x;
    double c = 0;
    acgidx_t nnzs = x->idx ? (x->num_nonzeros-x->num_ghost_nonzeros) : x->size;
    for (acgidx_t k = 0; k < nnzs; k++) c += fabs(xdata[k]);
    if (num_flops) *num_flops += nnzs;
    int mpierrcode = MPI_Allreduce(
        MPI_IN_PLACE, &c, 1, MPI_DOUBLE, MPI_SUM, comm);
    int err = mpierrcode ? ACG_ERR_MPI : ACG_SUCCESS;
    if ((err = acgerrmpi(comm, err, NULL, NULL, &mpierrcode))) {
        if (outmpierrcode) *outmpierrcode = mpierrcode;
        return err;
    }
    *asum = c;
    return ACG_SUCCESS;
}
#endif

/*
 * MPI functions
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acgvector_send()’ sends vectors to another MPI process.
 *
 * This is analogous to ‘MPI_Send()’ and requires the receiving
 * process to perform a matching call to ‘acgvector_recv()’.
 */
int acgvector_send(
    const struct acgvector * vectors,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    for (int i = 0; i < count; i++) {
        const struct acgvector * x = &vectors[i];

        /* partitioning information */
        MPI_Send(&x->nparts, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&x->parttag, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&x->nprocs, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&x->npparts, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&x->ownerrank, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&x->ownerpart, 1, MPI_INT, recipient, tag, comm);

        /* vector elements */
        MPI_Send(&x->size, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&x->num_nonzeros, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&x->idxbase, 1, MPI_INT, recipient, tag, comm);
        bool idx = x->idx;
        MPI_Send(&idx, 1, MPI_C_BOOL, recipient, tag, comm);
        if (idx) {
            MPI_Send(x->idx, x->num_nonzeros, MPI_ACGIDX_T, recipient, tag, comm);
        }
        MPI_Send(x->x, x->num_nonzeros, MPI_DOUBLE, recipient, tag, comm);
        MPI_Send(&x->num_ghost_nonzeros, 1, MPI_ACGIDX_T, recipient, tag, comm);
    }
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_recv()’ receives vectors from another MPI process.
 *
 * This is analogous to ‘MPI_Recv()’ and requires the sending process
 * to perform a matching call to ‘acgvector_send()’.
 */
int acgvector_recv(
    struct acgvector * vectors,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    for (int i = 0; i < count; i++) {
        struct acgvector * x = &vectors[i];

        /* partitioning information */
        MPI_Recv(&x->nparts, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->parttag, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->nprocs, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->npparts, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->ownerrank, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->ownerpart, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);

        /* vector elements */
        MPI_Recv(&x->size, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->num_nonzeros, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->idxbase, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        bool idx;
        MPI_Recv(&idx, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (idx) {
            x->idx = malloc(x->num_nonzeros*sizeof(*x->idx));
            if (!x->idx) return ACG_ERR_ERRNO;
            MPI_Recv(x->idx, x->num_nonzeros, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { x->idx = NULL; }
        x->x = malloc(x->num_nonzeros*sizeof(*x->x));
        if (!x->x) { free(x->idx); return ACG_ERR_ERRNO; }
        MPI_Recv(x->x, x->num_nonzeros, MPI_DOUBLE, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&x->num_ghost_nonzeros, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
    }
    return ACG_SUCCESS;
}

/**
 * ‘acgvector_irecv()’ performs a non-blocking receive of a vector
 * from another MPI process.
 *
 * This is analogous to ‘MPI_Irecv()’ and requires the sending process
 * to perform a matching call to ‘acgvector_send()’.
 */
int acgvector_irecv(
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
int acgvector_scatter(
    struct acgvector * sendvectors,
    int sendcount,
    struct acgvector * recvvectors,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err = ACG_SUCCESS, commsize, rank, threadlevel;
    err = MPI_Comm_size(comm, &commsize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Comm_rank(comm, &rank);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Query_thread(&threadlevel);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    if (rank == root) {
        int nparts = 0;
        for (int p = 0; p < commsize; p++) {
            if (sendcount > 0) {
                const struct acgvector * x = &sendvectors[0];
                nparts = x->nparts;
                break;
            }
        }

        /* map subvectors to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcount; i++) {
                struct acgvector * x = &sendvectors[sendcount*p+i];
                x->nprocs = commsize;
                x->npparts = sendcount;
                x->ownerrank = p;
                x->ownerpart = i;
            }
        }

        /* send from root process */
        if (threadlevel == MPI_THREAD_MULTIPLE) {
            #pragma omp parallel for schedule(dynamic)
            for (int p = 0; p < commsize; p++) {
                if (err) continue;
                if (rank != p) {
                    int perr = acgvector_send(
                        &sendvectors[sendcount*p], sendcount, p, p+1,
                        comm, mpierrcode);
                    if (perr) { err = perr; continue; }
                }
            }
            if (err) return err;
            for (int i = 0; i < sendcount; i++) {
                err = acgvector_init_copy(
                    &recvvectors[i],
                    &sendvectors[sendcount*rank+i]);
                if (err) return err;
            }
        } else {
            for (int p = 0; p < commsize; p++) {
                if (err) continue;
                if (rank != p) {
                    int perr = acgvector_send(
                        &sendvectors[sendcount*p], sendcount, p, p+1,
                        comm, mpierrcode);
                    if (perr) { err = perr; continue; }
                }
            }
            if (err) return err;
            for (int i = 0; i < sendcount; i++) {
                err = acgvector_init_copy(
                    &recvvectors[i],
                    &sendvectors[sendcount*rank+i]);
                if (err) return err;
            }
        }
    } else {
        /* receive from root process */
        err = acgvector_recv(
            recvvectors, recvcount, root, rank+1,
            comm, mpierrcode);
        if (err) return err;
    }
    return ACG_SUCCESS;
}
#endif
