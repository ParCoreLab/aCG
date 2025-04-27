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

#include "acg/config.h"
#include "acg/symcsrmatrix.h"
#include "acg/error.h"
#include "acg/fmtspec.h"
#include "acg/graph.h"
#include "acg/halo.h"
#include "acg/mtxfile.h"
#include "acg/prefixsum.h"
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
 * ‘acgsymcsrmatrix_init_real_double()’ allocates and initialises a
 * matrix from nonzeros provided in coordinate format with real,
 * double precision coefficients.
 *
 * Only upper triangular entries of A should be provided.
 */
int acgsymcsrmatrix_init_real_double(
    struct acgsymcsrmatrix * A,
    acgidx_t N,
    int64_t nnzs,
    int idxbase,
    const acgidx_t * rowidx,
    const acgidx_t * colidx,
    const double * a)
{
#ifndef NDEBUG
    for (int64_t k = 0; k < nnzs; k++) {
        if (rowidx[k] < idxbase || rowidx[k] > N+idxbase ||
            colidx[k] < idxbase || colidx[k] > N+idxbase) {
            return ACG_ERR_INDEX_OUT_OF_BOUNDS;
        }
    }
#endif

    int64_t * rowptr = malloc((size_t)(N+1)*sizeof(*rowptr));
    if (!rowptr) return ACG_ERR_ERRNO;
    #pragma omp parallel for
    for (acgidx_t i = 0; i <= N; i++) rowptr[i] = 0;

    /* check if the nonzeros are already sorted row-wise, in which
     * case we can avoid allocating some temporary storage */
    int64_t unsorted = 0;
    if (nnzs > 0) rowptr[rowidx[0]-idxbase+1]++;
    #pragma omp parallel for reduction(+:unsorted)
    for (int64_t k = 1; k < nnzs; k++) {
        if (rowidx[k-1] > rowidx[k]) unsorted++;
        #pragma omp atomic
        rowptr[rowidx[k]-idxbase+1]++;
    }
    acgprefixsum_inplace_int64_t(N+1, rowptr, true);
    int sorted = unsorted == 0;

    if (sorted) {
        struct acggraph * graph = malloc(sizeof(*graph));
        if (!graph) return ACG_ERR_ERRNO;
        int err = acggraph_init(graph, N, NULL, nnzs, NULL, idxbase, rowptr, colidx);
        if (err) { free(rowptr); return err; }
        free(rowptr);
        A->graph = graph;
        A->nrows = graph->nnodes;
        A->nprows = graph->npnodes;
        A->nzrows = graph->parentnodeidx;
        A->nnzs = graph->nedges;
        A->npnzs = graph->npedges;
        A->rowidxbase = graph->nodeidxbase;
        A->rownnzs = graph->nodenedges;
        A->rowptr = graph->srcnodeptr;
        A->rowidx = graph->srcnodeidx;
        A->colidx = graph->dstnodeidx;
        A->nownedrows = graph->nownednodes;
        A->ninnerrows = graph->ninnernodes;
        A->nborderrows = graph->nbordernodes;
        A->borderrowoffset = graph->bordernodeoffset;
        A->nghostrows = graph->nghostnodes;
        A->ghostrowoffset = graph->ghostnodeoffset;
        A->ninnernzs = graph->ninneredges;
        A->ninterfacenzs = graph->ninterfaceedges;
        A->nborderrowinnernzs = graph->nbordernodeinneredges;
        A->nborderrowinterfacenzs = graph->nbordernodeinterfaceedges;
        A->a = malloc((size_t)A->npnzs*sizeof(*A->a));
        if (!A->a) { acggraph_free(graph); free(graph); free(rowptr); return ACG_ERR_ERRNO; }
        #pragma omp parallel for
        for (int64_t k = 0; k < nnzs; k++) A->a[k] = a[k];
        A->fnpnzs = A->onpnzs = 0;
        A->frowptr = A->orowptr = NULL;
        A->fcolidx = A->ocolidx = NULL;
        A->fa = A->oa = NULL;

    } else {
        acgidx_t * rcolidx = malloc((size_t)nnzs*sizeof(*rcolidx));
        if (!rcolidx) { free(rowptr); return ACG_ERR_ERRNO; }
        double * ra = malloc((size_t)nnzs*sizeof(*ra));
        if (!ra) { free(rcolidx); free(rowptr); return ACG_ERR_ERRNO; }
        for (int64_t k = 0; k < nnzs; k++) {
            int64_t l = rowptr[rowidx[k]-idxbase]++;
            rcolidx[l] = colidx[k]; ra[l] = a[k];
        }
        for (acgidx_t i = N; i > 0; i--) rowptr[i] = rowptr[i-1];
        rowptr[0] = 0;
        int err = acgsymcsrmatrix_init_rowwise_real_double(
            A, N, idxbase, rowptr, rcolidx, ra);
        if (err) { free(ra); free(rcolidx); free(rowptr); return err; }
        free(ra); free(rcolidx); free(rowptr);
    }
    return ACG_SUCCESS;
}

/**
 * ‘acgsymcsrmatrix_init_rowwise_real_double()’ allocates and
 * initialises a matrix in compressed row format with real, double
 * precision coefficients.
 *
 * Only upper triangular entries of A should be provided.
 */
int acgsymcsrmatrix_init_rowwise_real_double(
    struct acgsymcsrmatrix * A,
    acgidx_t N,
    int idxbase,
    const int64_t * rowptr,
    const acgidx_t * colidx,
    const double * a)
{
    int64_t nnzs = rowptr[N];
#ifndef NDEBUG
    for (acgidx_t i = 0; i < N; i++) {
        for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) {
            if (colidx[k] < idxbase || colidx[k] > N+idxbase)
                return ACG_ERR_INDEX_OUT_OF_BOUNDS;
        }
    }
#endif

    struct acggraph * graph = malloc(sizeof(*graph));
    if (!graph) return ACG_ERR_ERRNO;
    int err = acggraph_init(graph, N, NULL, nnzs, NULL, idxbase, rowptr, colidx);
    if (err) return err;
    A->graph = graph;
    A->nrows = graph->nnodes;
    A->nprows = graph->npnodes;
    A->nzrows = graph->parentnodeidx;
    A->nnzs = graph->nedges;
    A->npnzs = graph->npedges;
    A->rowidxbase = graph->nodeidxbase;
    A->rownnzs = graph->nodenedges;
    A->rowptr = graph->srcnodeptr;
    A->rowidx = graph->srcnodeidx;
    A->colidx = graph->dstnodeidx;
    A->nownedrows = graph->nownednodes;
    A->ninnerrows = graph->ninnernodes;
    A->nborderrows = graph->nbordernodes;
    A->borderrowoffset = graph->bordernodeoffset;
    A->nghostrows = graph->nghostnodes;
    A->ghostrowoffset = graph->ghostnodeoffset;
    A->ninnernzs = graph->ninneredges;
    A->ninterfacenzs = graph->ninterfaceedges;
    A->nborderrowinnernzs = graph->nbordernodeinneredges;
    A->nborderrowinterfacenzs = graph->nbordernodeinterfaceedges;
    A->a = malloc((size_t)A->npnzs*sizeof(*A->a));
    if (!A->a) { acggraph_free(graph); free(graph); return ACG_ERR_ERRNO; }
    #pragma omp parallel for
    for (int64_t k = 0; k < nnzs; k++) A->a[k] = a[k];
    A->fnpnzs = A->onpnzs = 0;
    A->frowptr = A->orowptr = NULL;
    A->fcolidx = A->ocolidx = NULL;
    A->fa = A->oa = NULL;
    return ACG_SUCCESS;
}

/**
 * ‘acgsymcsrmatrix_free()’ frees storage allocated for a matrix.
 */
void acgsymcsrmatrix_free(
    struct acgsymcsrmatrix * A)
{
    acggraph_free(A->graph);
    free(A->graph);
    free(A->a);
    free(A->frowptr);
    free(A->fcolidx);
    free(A->fa);
}

static void acgsymcsrmatrix_init_from_graph(
    struct acgsymcsrmatrix * A)
{
    struct acggraph * g = A->graph;
    A->nrows = g->nnodes;
    A->nprows = g->npnodes;
    A->nzrows = g->parentnodeidx;
    A->nnzs = g->nedges;
    A->npnzs = g->npedges;
    A->rowidxbase = g->nodeidxbase;
    A->rownnzs = g->nodenedges;
    A->rowptr = g->srcnodeptr;
    A->rowidx = g->srcnodeidx;
    A->colidx = g->dstnodeidx;
    A->nownedrows = g->nownednodes;
    A->ninnerrows = g->ninnernodes;
    A->nborderrows = g->nbordernodes;
    A->borderrowoffset = g->bordernodeoffset;
    A->nghostrows = g->nghostnodes;
    A->ghostrowoffset = g->ghostnodeoffset;
    A->ninnernzs = g->ninneredges;
    A->ninterfacenzs = g->ninterfaceedges;
    A->nborderrowinnernzs = g->nbordernodeinneredges;
    A->nborderrowinterfacenzs = g->nbordernodeinterfaceedges;
    A->fnpnzs = A->onpnzs = 0;
    A->frowptr = A->orowptr = NULL;
    A->fcolidx = A->ocolidx = NULL;
    A->fa = A->oa = NULL;
}

/**
 * ‘acgsymcsrmatrix_copy()’ copies a matrix.
 */
int acgsymcsrmatrix_copy(
    struct acgsymcsrmatrix * dst,
    const struct acgsymcsrmatrix * src)
{
    dst->graph = malloc(sizeof(*dst->graph));
    if (!dst->graph) return ACG_ERR_ERRNO;
    int err = acggraph_copy(dst->graph, src->graph);
    if (err) { free(dst->graph); return err; }
    acgsymcsrmatrix_init_from_graph(dst);
    dst->a = malloc((size_t)dst->npnzs*sizeof(*dst->a));
    if (!dst->a) { acggraph_free(dst->graph); free(dst->graph); return ACG_ERR_ERRNO; }
    for (int64_t k = 0; k < dst->npnzs; k++) dst->a[k] = src->a[k];
    dst->fnpnzs = src->fnpnzs;
    dst->onpnzs = src->onpnzs;
    if (src->frowptr) {
        dst->frowptr = malloc((size_t)(dst->nprows+1)*sizeof(*dst->frowptr));
        if (!dst->frowptr) {
            free(dst->a); acggraph_free(dst->graph); free(dst->graph);
            return ACG_ERR_ERRNO;
        }
        for (acgidx_t i = 0; i <= dst->nprows; i++) dst->frowptr[i] = src->frowptr[i];
    } else { dst->frowptr = NULL; }
    if (src->orowptr) {
        dst->orowptr = malloc((size_t)(dst->nborderrows+dst->nghostrows+1)*sizeof(*dst->orowptr));
        if (!dst->orowptr) {
            free(dst->a); acggraph_free(dst->graph); free(dst->graph);
            return ACG_ERR_ERRNO;
        }
        for (acgidx_t i = 0; i <= dst->nborderrows+dst->nghostrows; i++) dst->orowptr[i] = src->orowptr[i];
    } else { dst->orowptr = NULL; }
    if (src->fcolidx) {
        dst->fcolidx = malloc((size_t)dst->fnpnzs*sizeof(*dst->fcolidx));
        if (!dst->fcolidx) {
            free(dst->frowptr);
            free(dst->a); acggraph_free(dst->graph); free(dst->graph);
            return ACG_ERR_ERRNO;
        }
        for (int64_t k = 0; k < dst->fnpnzs; k++) dst->fcolidx[k] = src->fcolidx[k];
    } else { dst->fcolidx = NULL; }
    if (src->ocolidx) {
        dst->ocolidx = malloc((size_t)dst->onpnzs*sizeof(*dst->ocolidx));
        if (!dst->ocolidx) {
            free(dst->frowptr);
            free(dst->a); acggraph_free(dst->graph); free(dst->graph);
            return ACG_ERR_ERRNO;
        }
        for (int64_t k = 0; k < dst->fnpnzs; k++) dst->ocolidx[k] = src->ocolidx[k];
    } else { dst->ocolidx = NULL; }
    if (src->fa) {
        dst->fa = malloc((size_t)dst->fnpnzs*sizeof(*dst->fa));
        if (!dst->fa) {
            free(dst->fcolidx); free(dst->frowptr);
            free(dst->a); acggraph_free(dst->graph); free(dst->graph);
            return ACG_ERR_ERRNO;
        }
        for (int64_t k = 0; k < dst->fnpnzs; k++) dst->fa[k] = src->fa[k];
    } else { dst->fa = NULL; }
    if (src->oa) {
        dst->oa = malloc((size_t)dst->onpnzs*sizeof(*dst->oa));
        if (!dst->oa) {
            free(dst->fcolidx); free(dst->frowptr);
            free(dst->a); acggraph_free(dst->graph); free(dst->graph);
            return ACG_ERR_ERRNO;
        }
        for (int64_t k = 0; k < dst->fnpnzs; k++) dst->oa[k] = src->oa[k];
    } else { dst->oa = NULL; }
    return ACG_SUCCESS;
}

/*
 * modifying values
 */

/**
 * ‘acgsymcsrmatrix_setzero()’ sets every value of a matrix to zero.
 */
int acgsymcsrmatrix_setzero(
    struct acgsymcsrmatrix * A)
{
    for (int64_t k = 0; k < A->npnzs; k++) A->a[k] = 0;
    if (A->fa) { for (int64_t k = 0; k < A->fnpnzs; k++) A->fa[k] = 0; }
    if (A->oa) { for (int64_t k = 0; k < A->onpnzs; k++) A->oa[k] = 0; }
    return ACG_SUCCESS;
}

/*
 * output to Matrix Market format
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
 * ‘acgsymcsrmatrix_fwrite()’ writes a symmetric sparse matrix to a
 * stream as a Matrix Market file.
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
int acgsymcsrmatrix_fwrite(
    const struct acgsymcsrmatrix * A,
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
        "matrix", !A->rowptr ? "array" : "coordinate",
        "real", "symmetric");
    if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
    if (bytes_written) *bytes_written += ret;

    if (comments) {
        ret = fputs(comments, f);
        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (bytes_written) *bytes_written += strlen(comments);
    }

    if (!A->rowptr) {
        ret = fprintf(f, "%"PRIdx" %"PRIdx"\n", A->nrows, A->nrows);
        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (bytes_written) *bytes_written += ret;
    } else {
        ret = fprintf(f, "%"PRIdx" %"PRIdx" %"PRId64"\n",
                      A->nrows, A->nrows, A->nnzs);
        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
        if (bytes_written) *bytes_written += ret;
    }

    if (A->nzrows) {
        if (A->rowptr) {
            if (fmt) {
                for (acgidx_t i = 0; i < A->nprows; i++) {
                    for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                        ret = fprintf(f, "%"PRIdx" %"PRIdx" ",
                                      A->nzrows[i]-A->rowidxbase+1,
                                      A->nzrows[A->colidx[k]]-A->rowidxbase+1);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(f, fmt, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            } else {
                for (acgidx_t i = 0; i < A->nprows; i++) {
                    for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                        ret = fprintf(f, "%"PRIdx" %"PRIdx" ",
                                      A->nzrows[i]-A->rowidxbase+1,
                                      A->nzrows[A->colidx[k]]-A->rowidxbase+1);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(f, "%.*g", DBL_DIG, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            }
        } else {
            if (fmt) {
                for (acgidx_t i = 0; i < A->nprows; i++) {
                    for (acgidx_t j = 0; j < A->nprows; j++) {
                        int64_t k = i*A->nprows+j;
                        ret = fprintf(f, "%"PRIdx" %"PRIdx" ",
                                      A->nzrows[i]-A->rowidxbase+1,
                                      A->nzrows[j]-A->rowidxbase+1);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(f, fmt, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            } else {
                for (acgidx_t i = 0; i < A->nprows; i++) {
                    for (acgidx_t j = 0; j < A->nprows; j++) {
                        int64_t k = i*A->nprows+j;
                        ret = fprintf(f, "%"PRIdx" %"PRIdx" ",
                                      A->nzrows[i]-A->rowidxbase+1,
                                      A->nzrows[j]-A->rowidxbase+1);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(f, "%.*g", DBL_DIG, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            }
        }
    } else {
        if (A->rowptr) {
            if (fmt) {
                for (acgidx_t i = 0; i < A->nrows; i++) {
                    for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                        acgidx_t rowidx = A->colidx[k]+1;
                        acgidx_t colidx = i+1;
                        ret = fprintf(f, "%"PRIdx" %"PRIdx" ", rowidx, colidx);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(f, fmt, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            } else {
                for (acgidx_t i = 0; i < A->nrows; i++) {
                    for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
                        ret = fprintf(f, "%"PRIdx" %"PRIdx" ",
                                      i+1, A->colidx[k]+1);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fprintf(f, "%.*g", DBL_DIG, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            }
        } else {
            if (fmt) {
                int64_t k = 0;
                for (acgidx_t i = 0; i < A->nrows; i++) {
                    acgidx_t jbegin = i;
                    acgidx_t jend = A->nrows;
                    for (acgidx_t j = jbegin; j < jend; j++, k++) {
                        ret = fprintf(f, fmt, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
            } else {
                int64_t k = 0;
                for (acgidx_t i = 0; i < A->nrows; i++) {
                    acgidx_t jbegin = i;
                    acgidx_t jend = A->nrows;
                    for (acgidx_t j = jbegin; j < jend; j++, k++) {
                        ret = fprintf(f, "%.*g", DBL_DIG, A->a[k]);
                        if (ret < 0) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) *bytes_written += ret;
                        ret = fputc('\n', f);
                        if (ret == EOF) { err = ACG_ERR_ERRNO; goto fwrite_exit; }
                        if (bytes_written) (*bytes_written)++;
                    }
                }
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
int acgsymcsrmatrix_vector(
    const struct acgsymcsrmatrix * A,
    struct acgvector * x)
{
    if (!A->nzrows) { return acgvector_alloc(x, A->nrows); }
    int err = acgvector_alloc_packed(
        x, A->nrows, A->nprows, 0, A->nzrows);
    if (err) return err;
    x->num_ghost_nonzeros = A->nghostrows;
    return ACG_SUCCESS;
}

/*
 * matrix partitioning
 */

/**
 * ‘acgsymcsrmatrix_partition_rows()’ partitions the rows of a
 * symmetric matrix into the given number of parts.
 *
 * The rows are partitioned using the METIS graph partitioner.
 */
int acgsymcsrmatrix_partition_rows(
    struct acgsymcsrmatrix * A,
    int nparts,
    enum metis_partitioner partitioner,
    int * rowparts,
    acgidx_t * objval,
    acgidx_t seed,
    int verbose)
{
    if (!A->graph) {
        struct acggraph * graph = malloc(sizeof(*graph));
        if (!graph) return ACG_ERR_ERRNO;
        int err = acggraph_init(
            graph, A->nrows, NULL, A->nnzs, NULL,
            0, A->rowptr, A->colidx);
        if (err) return err;
        A->graph = graph;
    }
    return acggraph_partition_nodes(
        A->graph, nparts, partitioner, rowparts, objval, seed, verbose);
}

/**
 * ‘acgsymcsrmatrix_partition()’ partitions a matrix into a number of
 * submatrices based on a given partitioning of the matrix rows.
 *
 * ‘submatrices’ must point to an array of length ‘nparts’, which is
 * used to store the partitioned submatrices.
 */
int acgsymcsrmatrix_partition(
    const struct acgsymcsrmatrix * A,
    int nparts,
    const int * rowparts,
    struct acgsymcsrmatrix * submatrices,
    int verbose)
{
    /* 1. partition the underlying graph */
    struct acggraph * subgraphs = malloc((size_t)nparts*sizeof(*subgraphs));
    if (!subgraphs) return ACG_ERR_ERRNO;
    int err = acggraph_partition(
        A->graph, nparts, rowparts, NULL, subgraphs, verbose);
    if (err) { free(subgraphs); return err; }

    /* initialise each submatrix from its subgraph */
    for (int p = 0; p < nparts; p++) {
        struct acgsymcsrmatrix * Ap = &submatrices[p];
        Ap->graph = malloc(sizeof(*Ap->graph));
        if (!Ap->graph) {
            for (int q = 0; q < nparts; q++) acggraph_free(&subgraphs[q]);
            free(subgraphs);
            return ACG_ERR_ERRNO;
        }
        *(Ap->graph) = subgraphs[p];
        struct acggraph * subgraph = Ap->graph;
        Ap->nrows = subgraph->nnodes;
        Ap->nprows = subgraph->npnodes;
        Ap->nzrows = subgraph->parentnodeidx;
        Ap->nnzs = subgraph->nedges;
        Ap->npnzs = subgraph->npedges;
        Ap->rowidxbase = subgraph->nodeidxbase;
        Ap->rownnzs = subgraph->nodenedges;
        Ap->rowptr = subgraph->srcnodeptr;
        Ap->rowidx = subgraph->srcnodeidx;
        Ap->colidx = subgraph->dstnodeidx;
        Ap->nownedrows = subgraph->nownednodes;
        Ap->ninnerrows = subgraph->ninnernodes;
        Ap->nborderrows = subgraph->nbordernodes;
        Ap->borderrowoffset = subgraph->bordernodeoffset;
        Ap->nghostrows = subgraph->nghostnodes;
        Ap->ghostrowoffset = subgraph->ghostnodeoffset;
        Ap->ninnernzs = subgraph->ninneredges;
        Ap->ninterfacenzs = subgraph->ninterfaceedges;
        Ap->nborderrowinnernzs = subgraph->nbordernodeinneredges;
        Ap->nborderrowinterfacenzs = subgraph->nbordernodeinterfaceedges;

        /* allocate storage for nonzero matrix values */
        Ap->a = malloc((size_t)Ap->npnzs*sizeof(*Ap->a));
        if (!Ap->a) {
            for (int q = p-1; q >= 0; q--) {
                acggraph_free(submatrices[p].graph);
                free(submatrices[p].graph);
            }
            for (int q = p; q < nparts; q++) acggraph_free(&subgraphs[q]);
            free(subgraphs);
            return ACG_ERR_ERRNO;
        }

        /* copy nonzero matrix values to submatrix */
        for (int64_t l = 0; l < Ap->npnzs; l++) {
            int64_t k = subgraph->parentedgeidx[l] < 0
                ? (-subgraph->parentedgeidx[l]-1)
                : (subgraph->parentedgeidx[l]-1);
            Ap->a[l] = A->a[k];
        }

        Ap->fnpnzs = Ap->onpnzs = 0;
        Ap->frowptr = Ap->orowptr = NULL;
        Ap->fcolidx = Ap->ocolidx = NULL;
        Ap->fa = Ap->oa = NULL;
    }
    free(subgraphs);
    return ACG_SUCCESS;
}

int acgsymcsrmatrix_dsymv_init(
    struct acgsymcsrmatrix * A,
    double eps)
{
    /* allocate storage for nonzeros in full storage format */
    A->fnpnzs = 0;
    A->frowptr = malloc((size_t)(A->nprows+1)*sizeof(*A->frowptr));
    if (!A->frowptr) return ACG_ERR_ERRNO;
    #pragma omp parallel for
    for (acgidx_t i = 0; i <= A->nprows; i++) A->frowptr[i] = 0;
    #pragma omp for
    for (acgidx_t i = 0; i < A->nprows; i++) {
        for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
            acgidx_t j = A->colidx[k]-A->rowidxbase;
            if (j < A->ghostrowoffset) {
                #pragma omp atomic
                A->frowptr[i+1]++;
                if (i != j) {
                    #pragma omp atomic
                    A->frowptr[j+1]++;
                }
            }
        }
    }
    /* for (acgidx_t i = 1; i <= A->nprows; i++) A->frowptr[i] += A->frowptr[i-1]; */
    acgprefixsum_inplace_int64_t(A->nprows+1, A->frowptr, true);
    A->fnpnzs = A->frowptr[A->nprows];

    A->fcolidx = malloc((size_t)A->fnpnzs*sizeof(*A->fcolidx));
    if (!A->fcolidx) return ACG_ERR_ERRNO;
    A->fa = malloc((size_t)A->fnpnzs*sizeof(*A->fa));
    if (!A->fa) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < A->nprows; i++) {
        for (int64_t k = A->rowptr[i]; k < A->rowptr[i+1]; k++) {
            acgidx_t j = A->colidx[k]-A->rowidxbase;
            if (j < A->ghostrowoffset) {
                int64_t l = A->frowptr[i]++;
                A->fcolidx[l] = j+A->rowidxbase; A->fa[l] = A->a[k] + ((i == j) ? eps : 0.0);
                if (i != j) {
                    int64_t l = A->frowptr[j]++;
                    A->fcolidx[l] = i+A->rowidxbase; A->fa[l] = A->a[k];
                }
            } else {
                /* int64_t l = A->frowptr[i]++; */
                /* A->fcolidx[l] = j+A->rowidxbase; A->fa[l] = A->a[k] + ((i == j) ? eps : 0.0); */
                /* if (i != j) { */
                /*     int64_t l = A->frowptr[j]++; */
                /*     A->fcolidx[l] = i+A->rowidxbase; A->fa[l] = A->a[k]; */
                /* } */
            }
        }
    }
    for (acgidx_t i = A->nprows; i > 0; i--) A->frowptr[i] = A->frowptr[i-1];
    A->frowptr[0] = 0;

    /* allocate storage for offdiagonal nonzeros in full storage format */
    A->onpnzs = 0;
    A->orowptr = malloc((size_t)(A->nborderrows+A->nghostrows+1)*sizeof(*A->orowptr));
    if (!A->orowptr) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i <= A->nborderrows+A->nghostrows; i++) A->orowptr[i] = 0;
    for (acgidx_t i = 0; i < A->nborderrows+A->nghostrows; i++) {
        for (int64_t k = A->rowptr[A->borderrowoffset+i]; k < A->rowptr[A->borderrowoffset+i+1]; k++) {
            acgidx_t j = A->colidx[k]-A->rowidxbase;
            if (j >= A->ghostrowoffset) {
                A->orowptr[i+1]++; A->onpnzs++;
            }
        }
    }
    for (acgidx_t i = 1; i <= A->nborderrows+A->nghostrows; i++) A->orowptr[i] += A->orowptr[i-1];
    A->ocolidx = malloc((size_t)A->onpnzs*sizeof(*A->ocolidx));
    if (!A->ocolidx) return ACG_ERR_ERRNO;
    A->oa = malloc((size_t)A->onpnzs*sizeof(*A->oa));
    if (!A->oa) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < A->nborderrows+A->nghostrows; i++) {
        for (int64_t k = A->rowptr[A->borderrowoffset+i]; k < A->rowptr[A->borderrowoffset+i+1]; k++) {
            acgidx_t j = A->colidx[k]-A->rowidxbase;
            if (j >= A->ghostrowoffset) {
                int64_t l = A->orowptr[i]++;
                A->ocolidx[l] = j+A->rowidxbase-A->borderrowoffset; A->oa[l] = A->a[k] + ((i == j) ? eps : 0.0);
            }
        }
    }
    for (acgidx_t i = A->nborderrows+A->nghostrows; i > 0; i--) A->orowptr[i] = A->orowptr[i-1];
    A->orowptr[0] = 0;
    return ACG_SUCCESS;
}

/*
 * Level 2 BLAS operations (matrix-vector)
 */

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
int acgsymcsrmatrix_dsymv(
    double alpha,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * x,
    double beta,
    struct acgvector * y,
    int64_t * num_flops,
    int64_t * num_bytes)
{
    int err;
    if (A->nrows != y->size ||
        A->nprows != y->num_nonzeros ||
        A->nghostrows != y->num_ghost_nonzeros)
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;
    if (A->nrows != x->size ||
        A->nprows != x->num_nonzeros ||
        A->nghostrows != x->num_ghost_nonzeros)
        return ACG_ERR_VECTOR_INCOMPATIBLE_SIZE;

    if (beta != 1) {
        err = acgvector_dscal(beta, y, num_flops, num_bytes);
        if (err) return err;
    }

    if (A->frowptr && A->fcolidx && A->fa) {
        int idxbase = A->rowidxbase;
        const int64_t * rowptr = A->frowptr;
        const acgidx_t * j = A->fcolidx;
        const double * a = A->fa;
        const double * xdata = x->x;
        double * ydata = y->x;
#ifdef ACG_DSYMV_NO_UNROLL
        #pragma omp parallel for
        for (acgidx_t i = 0; i < A->nprows-A->nghostrows; i++) {
            double z = 0;
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                z += a[k]*xdata[j[k]-idxbase];
            ydata[i] += alpha*z;
        }
#else
        acgidx_t m = A->nprows-A->nghostrows;
        #pragma omp parallel for
        for (acgidx_t i = 0; i < m-m%4; i+=4) {
            double z0 = 0, z1 = 0, z2 = 0, z3 = 0;
            int64_t n0 = rowptr[i+1]-rowptr[i], n1 = rowptr[i+2]-rowptr[i+1], n2 = rowptr[i+3]-rowptr[i+2], n3 = rowptr[i+4]-rowptr[i+3];
            for (int64_t k = rowptr[i]; k < rowptr[i+1]-n0%2; k+=2)
                z0 += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+1]-n0%2; k < rowptr[i+1]; k++)
                z0 += a[k]*xdata[j[k]];
            for (int64_t k = rowptr[i+1]; k < rowptr[i+2]-n1%2; k+=2)
                z1 += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+2]-n1%2; k < rowptr[i+2]; k++)
                z1 += a[k]*xdata[j[k]];
            for (int64_t k = rowptr[i+2]; k < rowptr[i+3]-n2%2; k+=2)
                z2 += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+3]-n2%2; k < rowptr[i+3]; k++)
                z2 += a[k]*xdata[j[k]];
            for (int64_t k = rowptr[i+3]; k < rowptr[i+4]-n3%2; k+=2)
                z3 += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+4]-n3%2; k < rowptr[i+4]; k++)
                z3 += a[k]*xdata[j[k]];
            /* for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++) */
            /*     z0 += alpha*a[k]*xdata[j[k]]; */
            /* for (int64_t k = rowptr[i+1]; k < rowptr[i+2]; k++) */
            /*     z1 += alpha*a[k]*xdata[j[k]]; */
            /* for (int64_t k = rowptr[i+2]; k < rowptr[i+3]; k++) */
            /*     z2 += alpha*a[k]*xdata[j[k]]; */
            /* for (int64_t k = rowptr[i+3]; k < rowptr[i+4]; k++) */
            /*     z3 += alpha*a[k]*xdata[j[k]]; */
            ydata[i+0] += alpha*z0;
            ydata[i+1] += alpha*z1;
            ydata[i+2] += alpha*z2;
            ydata[i+3] += alpha*z3;
        }
        for (acgidx_t i = m-m%4; i < m-m%2; i+=2) {
            double z0 = 0, z1 = 0;
            int64_t n0 = rowptr[i+1]-rowptr[i], n1 = rowptr[i+2]-rowptr[i+1];
            for (int64_t k = rowptr[i]; k < rowptr[i+1]-n0%2; k+=2)
                z0 += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+1]-n0%2; k < rowptr[i+1]; k++)
                z0 += a[k]*xdata[j[k]];
            for (int64_t k = rowptr[i+1]; k < rowptr[i+2]-n1%2; k+=2)
                z1 += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+2]-n1%2; k < rowptr[i+2]; k++)
                z1 += a[k]*xdata[j[k]];
            ydata[i+0] += alpha*z0;
            ydata[i+1] += alpha*z1;
        }
        for (acgidx_t i = m-m%2; i < m; i++) {
            double z = 0;
            int64_t n = rowptr[i+1]-rowptr[i];
            for (int64_t k = rowptr[i]; k < rowptr[i+1]-n%2; k+=2)
                z += a[k]*xdata[j[k]]+a[k+1]*xdata[j[k+1]];
            for (int64_t k = rowptr[i+1]-n%2; k < rowptr[i+1]; k++)
                z += a[k]*xdata[j[k]];
            ydata[i] += alpha*z;
        }
#endif
        if (num_flops) *num_flops += 2*(A->fnpnzs+(A->nprows-A->nghostrows));
        if (num_bytes)
            *num_bytes +=
                A->fnpnzs*(sizeof(*a)+sizeof(*j))
                + (A->nprows-A->nghostrows)*(sizeof(*rowptr)+sizeof(*ydata))
                + x->num_nonzeros*sizeof(*xdata);
    } else {
        int idxbase = A->rowidxbase;
        const int64_t * rowptr = A->rowptr;
        const acgidx_t * j = A->colidx;
        const double * a = A->a;
        const double * xdata = x->x;
        double * ydata = y->x;
        #pragma omp parallel for
        for (acgidx_t i = 0; i < A->nprows-A->nghostrows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                ydata[i] += alpha*a[k]*xdata[j[k]-idxbase];
        }
        if (num_flops) *num_flops += 3*A->npnzs;
        if (num_bytes)
            *num_bytes +=
                A->fnpnzs*(sizeof(*a)+sizeof(*j))
                + (A->nprows-A->nghostrows)*(sizeof(*rowptr)+sizeof(*ydata))
                + x->num_nonzeros*sizeof(*xdata);
        for (acgidx_t i = 0; i < A->nprows-A->nghostrows; i++) {
            for (int64_t k = rowptr[i]; k < rowptr[i+1]; k++)
                if (i != j[k]-idxbase) ydata[j[k]-idxbase] += alpha*a[k]*xdata[i];
        }
        if (num_flops) *num_flops += 3*A->npnzs;
        if (num_bytes)
            *num_bytes +=
                A->fnpnzs*(sizeof(*a)+sizeof(*j))
                + (A->nprows-A->nghostrows)*(sizeof(*rowptr)+sizeof(*xdata))
                + y->num_nonzeros*sizeof(*ydata);
    }
    return ACG_SUCCESS;
}

/*
 * matrix distribution
 */

#ifdef ACG_HAVE_MPI

static int MPI_Send64(
    const void * buffer,
    int64_t count,
    MPI_Datatype datatype,
    int recipient,
    int tag,
    MPI_Comm comm)
{
    if (count <= INT_MAX)
        return MPI_Send(buffer, count, datatype, recipient, tag, comm);
    int typesize;
    int err = MPI_Type_size(datatype, &typesize);
    if (err) return err;
    size_t offset = 0;
    int64_t n = count;
    while (n > 0) {
        int r = n < INT_MAX ? n : INT_MAX;
        const void * p = &((const char *) buffer)[offset];
        err = MPI_Send(p, r, datatype, recipient, tag, comm);
        if (err) return err;
        offset += (size_t)r*typesize;
        n -= r;
    }
    return 0;
}

static int MPI_Recv64(
    void * buffer,
    int64_t count,
    MPI_Datatype datatype,
    int sender,
    int tag,
    MPI_Comm comm,
    MPI_Status * status)
{
    if (count <= INT_MAX)
        return MPI_Recv(buffer, count, datatype, sender, tag, comm, status);
    if (status != MPI_STATUS_IGNORE) {
        fprintf(stderr, "%s:%d: MPI_Recv64 - status not supported\n", __FILE__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int typesize;
    int err = MPI_Type_size(datatype, &typesize);
    if (err) return err;
    size_t offset = 0;
    int64_t n = count;
    while (n > 0) {
        int r = n < INT_MAX ? n : INT_MAX;
        void * p = &((char *) buffer)[offset];
        err = MPI_Recv(p, r, datatype, sender, tag, comm, MPI_STATUS_IGNORE);
        if (err) return err;
        offset += (size_t)r*typesize;
        n -= r;
    }
    return 0;
}

int acgsymcsrmatrix_send(
    const struct acgsymcsrmatrix * matrices,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    for (int i = 0; i < count; i++) {
        const struct acgsymcsrmatrix * A = &matrices[i];

        /* matrix sparsity pattern, partitioning and distribution */
        bool graph = A->graph;
        MPI_Send(&graph, 1, MPI_C_BOOL, recipient, tag, comm);
        if (graph) {
            err = acggraph_send(
                A->graph, 1, recipient, tag, comm, mpierrcode);
            if (err) return err;
        } else { return ACG_ERR_NOT_SUPPORTED; }

        /* matrix/submatrix nonzero values */
        MPI_Send64(A->a, A->npnzs, MPI_DOUBLE, recipient, tag, comm);

        /* matrix/submatrix nonzeros in full storage format */
        MPI_Send(&A->fnpnzs, 1, MPI_INT64_T, recipient, tag, comm);
        bool frowptr = A->frowptr;
        MPI_Send(&frowptr, 1, MPI_C_BOOL, recipient, tag, comm);
        if (frowptr) MPI_Send(A->frowptr, A->nprows+1, MPI_INT64_T, recipient, tag, comm);
        bool fcolidx = A->fcolidx;
        MPI_Send(&fcolidx, 1, MPI_C_BOOL, recipient, tag, comm);
        if (fcolidx) MPI_Send64(A->fcolidx, A->fnpnzs, MPI_ACGIDX_T, recipient, tag, comm);
        bool fa = A->fa;
        MPI_Send(&fa, 1, MPI_C_BOOL, recipient, tag, comm);
        if (fa) MPI_Send64(A->fa, A->fnpnzs, MPI_DOUBLE, recipient, tag, comm);

        MPI_Send(&A->onpnzs, 1, MPI_INT64_T, recipient, tag, comm);
        bool orowptr = A->orowptr;
        MPI_Send(&orowptr, 1, MPI_C_BOOL, recipient, tag, comm);
        if (orowptr) MPI_Send(A->orowptr, A->nborderrows+A->nghostrows+1, MPI_INT64_T, recipient, tag, comm);
        bool ocolidx = A->ocolidx;
        MPI_Send(&ocolidx, 1, MPI_C_BOOL, recipient, tag, comm);
        if (ocolidx) MPI_Send64(A->ocolidx, A->onpnzs, MPI_ACGIDX_T, recipient, tag, comm);
        bool oa = A->oa;
        MPI_Send(&oa, 1, MPI_C_BOOL, recipient, tag, comm);
        if (oa) MPI_Send64(A->oa, A->onpnzs, MPI_DOUBLE, recipient, tag, comm);
    }
    return ACG_SUCCESS;
}

int acgsymcsrmatrix_recv(
    struct acgsymcsrmatrix * matrices,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    for (int i = 0; i < count; i++) {
        struct acgsymcsrmatrix * A = &matrices[i];

        /* matrix sparsity pattern, partitioning and distribution */
        bool graph = A->graph;
        MPI_Recv(&graph, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (graph) {
            A->graph = malloc(sizeof(*A->graph));
            err = acggraph_recv(
                A->graph, 1, sender, tag, comm, mpierrcode);
            if (err) return err;
            struct acggraph * g = A->graph;
            A->nrows = g->nnodes;
            A->nprows = g->npnodes;
            A->nzrows = g->parentnodeidx;
            A->nnzs = g->nedges;
            A->npnzs = g->npedges;
            A->rowidxbase = g->nodeidxbase;
            A->rownnzs = g->nodenedges;
            A->rowptr = g->srcnodeptr;
            A->rowidx = g->srcnodeidx;
            A->colidx = g->dstnodeidx;
            A->nownedrows = g->nownednodes;
            A->ninnerrows = g->ninnernodes;
            A->nborderrows = g->nbordernodes;
            A->borderrowoffset = g->bordernodeoffset;
            A->nghostrows = g->nghostnodes;
            A->ghostrowoffset = g->ghostnodeoffset;
            A->ninnernzs = g->ninneredges;
            A->ninterfacenzs = g->ninterfaceedges;
            A->nborderrowinnernzs = g->nbordernodeinneredges;
            A->nborderrowinterfacenzs = g->nbordernodeinterfaceedges;
            A->fnpnzs = A->onpnzs = 0;
            A->frowptr = A->orowptr = NULL;
            A->fcolidx = A->ocolidx = NULL;
            A->fa = A->oa = NULL;
        } else { return ACG_ERR_NOT_SUPPORTED; }

        /* matrix/submatrix nonzero values */
        A->a = malloc((size_t)A->npnzs*sizeof(*A->a));
        if (!A->a) return ACG_ERR_ERRNO;
        MPI_Recv64(A->a, A->npnzs, MPI_DOUBLE, sender, tag, comm, MPI_STATUS_IGNORE);

        /* matrix/submatrix nonzeros in full storage format */
        MPI_Recv(&A->fnpnzs, 1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        bool frowptr;
        MPI_Recv(&frowptr, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (frowptr) {
            A->frowptr = malloc((size_t)(A->nprows+1)*sizeof(*A->frowptr));
            if (!A->frowptr) return ACG_ERR_ERRNO;
            MPI_Recv(A->frowptr, A->nprows+1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { A->frowptr = NULL; }
        bool fcolidx;
        MPI_Recv(&fcolidx, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (fcolidx) {
            A->fcolidx = malloc((size_t)A->fnpnzs*sizeof(*A->fcolidx));
            if (!A->fcolidx) return ACG_ERR_ERRNO;
            MPI_Recv64(A->fcolidx, A->fnpnzs, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { A->fcolidx = NULL; }
        bool fa;
        MPI_Recv(&fa, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (fa) {
            A->fa = malloc((size_t)A->fnpnzs*sizeof(*A->fa));
            if (!A->fa) return ACG_ERR_ERRNO;
            MPI_Recv64(A->fa, A->fnpnzs, MPI_DOUBLE, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { A->fa = NULL; }

        MPI_Recv(&A->onpnzs, 1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        bool orowptr;
        MPI_Recv(&orowptr, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (orowptr) {
            A->orowptr = malloc((size_t)(A->nborderrows+A->nghostrows+1)*sizeof(*A->orowptr));
            if (!A->orowptr) return ACG_ERR_ERRNO;
            MPI_Recv(A->orowptr, A->nborderrows+A->nghostrows+1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { A->orowptr = NULL; }
        bool ocolidx;
        MPI_Recv(&ocolidx, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (ocolidx) {
            A->ocolidx = malloc((size_t)A->onpnzs*sizeof(*A->ocolidx));
            if (!A->ocolidx) return ACG_ERR_ERRNO;
            MPI_Recv64(A->ocolidx, A->onpnzs, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { A->ocolidx = NULL; }
        bool oa;
        MPI_Recv(&oa, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (oa) {
            A->oa = malloc((size_t)A->onpnzs*sizeof(*A->oa));
            if (!A->oa) return ACG_ERR_ERRNO;
            MPI_Recv64(A->oa, A->onpnzs, MPI_DOUBLE, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { A->oa = NULL; }
    }
    return ACG_SUCCESS;
}

int acgsymcsrmatrix_scatter(
    struct acgsymcsrmatrix * sendmatrices,
    int sendcount,
    struct acgsymcsrmatrix * recvmatrices,
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
                const struct acgsymcsrmatrix * A = &sendmatrices[0];
                const struct acggraph * g = A->graph;
                nparts = g->nparts;
                break;
            }
        }

        /* map submatrices to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcount; i++) {
                struct acgsymcsrmatrix * A = &sendmatrices[sendcount*p+i];
                struct acggraph * g = A->graph;
                g->nprocs = commsize;
                g->npparts = sendcount;
                g->ownerrank = p;
                g->ownerpart = i;
            }
        }

        /* map neighbouring submatrices to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcount; i++) {
                struct acgsymcsrmatrix * A = &sendmatrices[sendcount*p+i];
                struct acggraph * g = A->graph;
                for (int j = 0; j < g->nneighbours; j++) {
                    struct acggraphneighbour * n = &g->neighbours[j];
                    int q = n->neighbourpart;
                    if (q < 0 || q >= nparts)
                        return ACG_ERR_INDEX_OUT_OF_BOUNDS;
                    struct acgsymcsrmatrix * B = &sendmatrices[q];
                    struct acggraph * h = B->graph;
                    n->neighbourrank = h->ownerrank;
                    n->neighbourpart = h->ownerpart;
                }
            }
        }

        /* send from root process */
        if (threadlevel == MPI_THREAD_MULTIPLE) {
            #pragma omp parallel for schedule(dynamic)
            for (int p = 0; p < commsize; p++) {
                if (err) continue;
                if (rank != p) {
                    int perr = acgsymcsrmatrix_send(
                        &sendmatrices[sendcount*p], sendcount, p, p+1,
                        comm, mpierrcode);
                    if (perr) { err = perr; continue; }
                }
            }
            if (err) return err;
            for (int i = 0; i < sendcount; i++) {
                err = acgsymcsrmatrix_copy(
                    &recvmatrices[i],
                    &sendmatrices[sendcount*rank+i]);
                if (err) return err;
            }
        } else {
            for (int p = 0; p < commsize; p++) {
                if (err) continue;
                if (rank != p) {
                    int perr = acgsymcsrmatrix_send(
                        &sendmatrices[sendcount*p], sendcount, p, p+1,
                        comm, mpierrcode);
                    if (perr) { err = perr; continue; }
                }
            }
            if (err) return err;
            for (int i = 0; i < sendcount; i++) {
                err = acgsymcsrmatrix_copy(
                    &recvmatrices[i],
                    &sendmatrices[sendcount*rank+i]);
                if (err) return err;
            }
        }

    } else {
        /* receive from root process */
        err = acgsymcsrmatrix_recv(
            recvmatrices, recvcount, root, rank+1,
            comm, mpierrcode);
        if (err) return err;
    }
    return ACG_SUCCESS;
}
#endif

/*
 * halo exchange/update for partitioned and distributed matrices
 */

/**
 * ‘acgsymcsrmatrix_halo()’ sets up a halo exchange communication
 * pattern to send and receive data associated with the “ghost”
 * rows/columns of partitioned and distributed vectors associated with
 * the row/column space of the given matrix.
 */
int acgsymcsrmatrix_halo(
    const struct acgsymcsrmatrix * A,
    struct acghalo * halo)
{
    return acggraph_halo(A->graph, halo);
}

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
int acgsymcsrmatrix_dsymvmpi(
    double alpha,
    const struct acgsymcsrmatrix * A,
    const struct acgvector * x,
    struct acgvector * y,
    struct acghalo * halo,
    int64_t * num_flops,
    int64_t * num_bytes,
    MPI_Comm comm,
    int tag,
    int * mpierrcode)
{
    int err;

    bool free_halo = !halo;
    if (!halo) {
        halo = malloc(sizeof(*halo));
        err = acgsymcsrmatrix_halo(A, halo);
        if (err) { free(halo); return err; }
    }

    /* on each process, update every ghost entry of the input vector
     * by receiving its value from the process with exclusive
     * ownership */
    err = acghalo_exchange(
        halo, x->num_nonzeros, x->x, MPI_DOUBLE,
        x->num_nonzeros, x->x, MPI_DOUBLE,
        0, NULL, NULL, 0, NULL, NULL, comm, tag, mpierrcode);
    if (err) {
        if (free_halo) { acghalo_free(halo); free(halo); }
        return err;
    }

    /* perform local matrix-vector multiply */
    err = acgsymcsrmatrix_dsymv(alpha, A, x, 1, y, num_flops, num_bytes);
    if (err) {
        if (free_halo) { acghalo_free(halo); free(halo); }
        return err;
    }
    /* if ((err = acgerrmpi(comm, err, NULL, NULL, NULL))) return err; */

    if (free_halo) { acghalo_free(halo); free(halo); }
    return ACG_SUCCESS;
}
#endif
