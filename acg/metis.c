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
 * METIS graph partitioner
 */

#include "acg/config.h"
#include "acg/metis.h"
#include "acg/error.h"
#include "acg/sort.h"

#ifdef ACG_HAVE_METIS
#include <metis.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

/**
 * ‘metis_partgraphsym()’ uses the METIS k-way graph partitioner to
 * partition an undirected graph given as a square, symmetric matrix
 * in coordinate format.
 *
 * Recursive bipartitioning or k-way partitioning is specified by
 * setting the ‘partitioner’ argument accordingly.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in coordinate (COO) format with ‘N’ rows and columns. There
 * are ‘nnzs’ nonzero matrix entries. The locations of the matrix
 * nonzeros are specified by the arrays ‘rowidx’ and ‘colidx’, both of
 * which are of length ‘nnzs’, and contain offsets in the range
 * ‘[0,N)’. Note that there should not be any duplicate nonzero
 * entries. The nonzeros may be located in the upper or lower triangle
 * of the adjacency matrix. However, if there is a nonzero entry at
 * row ‘i’ and column ‘j’, then there should not be a nonzero entry
 * row ‘j’ and column ‘i’.
 *
 * On success, the array ‘dstpart’ contains the part numbers assigned
 * by the partitioner to the graph vertices. Therefore, ‘dstpart’ must
 * be an array of length ‘N’.
 *
 * If it is not ‘NULL’, then ‘objval’ is used to store the value of
 * the objective function minimized by the partitioner, which, by
 * default, is the edge-cut of the partitioning solution.
 *
 * A seed for the random number generator can be specified through
 * ‘seed’. If ‘seed’ is set to -1, the default behaviour is used.
 */
int metis_partgraphsym(
    enum metis_partitioner partitioner,
    int nparts,
    acgidx_t N,
    int64_t nnzs,
    int rowidxstride,
    int rowidxbase,
    const acgidx_t * rowidx,
    int colidxstride,
    int colidxbase,
    const acgidx_t * colidx,
    int * dstpart,
    acgidx_t * outobjval,
    acgidx_t seed,
    int verbose)
{
#ifndef ACG_HAVE_METIS
    return ACG_ERR_METIS_NOT_SUPPORTED;
#else
    if (N > IDX_MAX-1) return ACG_ERR_METIS_EOVERFLOW;
    if (nnzs > IDX_MAX) return ACG_ERR_METIS_EOVERFLOW;
    if (nparts <= 0) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    /* perform some bounds checking on the input graph */
    for (int64_t k = 0; k < nnzs; k++) {
        acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
        acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
        if (i < 0 || i >= N || j < 0 || j >= N) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* handle edge case which causes METIS to crash */
    if (nparts == 1) {
        for (idx_t i = 0; i < N; i++) dstpart[i] = 0;
        *outobjval = 0;
        return ACG_SUCCESS;
    }

    /* configure partitioning options */
    idx_t options[METIS_NOPTIONS]; /* array of options */
    int err = METIS_SetDefaultOptions(options);
    if (err == METIS_ERROR_INPUT) { return ACG_ERR_METIS_INPUT; }
    else if (err == METIS_ERROR_MEMORY) { return ACG_ERR_METIS_MEMORY; }
    else if (err == METIS_ERROR) { return ACG_ERR_METIS; }

    /* METIS_OBJTYPE_CUT for edge-cut minimization (default) or
     * METIS_OBJTYPE_VOL for total communication volume
     * minimization */
    // options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;

    /* METIS_CTYPE_RM for random matching or METIS_CTYPE_SHEM for
     * sorted heavy-edge matching (default) */
    // options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;

    /*
     * METIS_IPTYPE_GROW - grows a bisection using a greedy strategy (default)
     * METIS_IPTYPE_RANDOM - computes a bisection at random followed
     *                       by a refinement
     * METIS_IPTYPE_EDGE - derives a separator from an edge cut
     * METIS_IPTYPE_NODE - grow a bisection using a greedy node-based
     *                     strategy
     */
    // options[METIS_OPTION_IPTYPE] = METIS_IPTYPE_GROW;

    /*
     * METIS_RTYPE_FM - FM-based cut refinement
     * METIS_RTYPE_GREEDY - greedy-based cut and volume refinement
     * METIS_RTYPE_SEP2SIDED - two-sided node FM refinement
     * METIS_RTYPE_SEP1SIDED - One-sided node FM refinement (default)
     */
    // options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;

    /* 0 (default) performs a 2–hop matching or 1 does not */
    // options[METIS_OPTION_NO2HOP] = 0;

    /* The number of different partitionings to compute. The final
     * partitioning is the one that achieves the best edgecut or
     * communication volume. Default is 1. */
    // options[METIS_OPTION_NCUTS] = 1;

    /* The number of iterations for the refinement algorithms at each
     * stage of the uncoarsening process. Default is 10. */
    // options[METIS_OPTION_NITER] = 10;

    /* Specifies the maximum allowed load imbalance among the
     * partitions. A value of x indicates that the allowed load
     * imbalance is (1 + x)/1000. The load imbalance for the jth
     * constraint is defined to be max_i(w[j,i])/t[j,i]), where w[j,i]
     * is the fraction of the overall weight of the jth constraint
     * that is assigned to the ith partition and t[j, i] is the
     * desired target weight of the jth constraint for the ith
     * partition (i.e., that specified via -tpwgts). For -ptype=rb,
     * the default value is 1 (i.e., load imbalance of 1.001) and for
     * -ptype=kway, the default value is 30 (i.e., load imbalance of
     * 1.03). */
    // options[METIS_OPTION_UFACTOR] = 30;

    /* 0 does not explicitly minimize maximum connectivity (default),
     * or 1 explicitly minimizes maximum connectivity. */
    // options[METIS_OPTION_MINCONN] = 0;

    /* 0 does not force contiguous partitions (default), or 1 forces
     * contiguous partitions. */
    // options[METIS_OPTION_CONTIG] = 0;

    /* seed for the random number generator */
    if (seed != -1) {
        if (seed < IDX_MIN || seed > IDX_MAX) return ACG_ERR_METIS_EOVERFLOW;
        options[METIS_OPTION_SEED] = seed;
    }

    /* 0 for C-style numbering that starts from 0 (default), or 1 for
     * Fortran-style numbering that starts from 1 */
    // options[METIS_OPTION_NUMBERING] = 0;

    /*
     * METIS_DBG_INFO (1)
     * METIS_DBG_TIME (2)
     * METIS_DBG_COARSEN (4)
     * METIS_DBG_REFINE (8)
     * METIS_DBG_IPART (16)
     * METIS_DBG_MOVEINFO (32)
     * METIS_DBG_SEPINFO (64)
     * METIS_DBG_CONNINFO (128)
     * METIS_DBG_CONTIGINFO (256)
     */
    options[METIS_OPTION_DBGLVL] = 0;
    if (verbose > 0) options[METIS_OPTION_DBGLVL] |= METIS_DBG_INFO | METIS_DBG_TIME;
    /* if (verbose > 1) { */
    /*     options[METIS_OPTION_DBGLVL] |= ( */
    /*         METIS_DBG_COARSEN | */
    /*         METIS_DBG_REFINE | */
    /*         METIS_DBG_IPART | */
    /*         METIS_DBG_MOVEINFO | */
    /*         METIS_DBG_SEPINFO | */
    /*         METIS_DBG_CONNINFO | */
    /*         METIS_DBG_CONTIGINFO); */
    /* } */

    /* the number of vertices in the graph */
    idx_t nvtxs = N;

    /* the number of balancing constraints, should be at least 1 */
    idx_t ncon = 1;

    /* adjacency structure of the graph (row pointers) */
    idx_t * xadj = malloc((size_t)(nvtxs+1)*sizeof(*xadj));
    if (!xadj) return ACG_ERR_ERRNO;
    for (idx_t i = 0; i <= nvtxs; i++) xadj[i] = 0;
    for (int64_t k = 0; k < nnzs; k++) {
        acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
        acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
        if (i != j) { xadj[i+1]++; xadj[j+1]++; }
    }
    for (idx_t i = 1; i <= nvtxs; i++) {
        if (xadj[i] > IDX_MAX - xadj[i-1]) { free(xadj); return ACG_ERR_METIS_EOVERFLOW; }
        xadj[i] += xadj[i-1];
    }

    /* adjacency structure of the graph (column offsets) */
    idx_t * adjncy = malloc((size_t)xadj[nvtxs]*sizeof(*adjncy));
    if (!adjncy) { free(xadj); return ACG_ERR_ERRNO; }

    for (int64_t k = 0; k < nnzs; k++) {
        acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
        acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
        if (i != j) { adjncy[xadj[i]++] = j; adjncy[xadj[j]++] = i; }
    }
    for (idx_t i = nvtxs; i > 0; i--) xadj[i] = xadj[i-1];
    xadj[0] = 0;

#if defined(DEBUG_METIS)
    if (verbose) {
        fprintf(stderr, "nvtxs=%"PRIDX", xadj=[", nvtxs);
        for (idx_t i = 0; i <= nvtxs; i++) fprintf(stderr, " %"PRIDX, xadj[i]);
        fprintf(stderr, "]\n");
        fprintf(stderr, "adjncy=[");
        for (idx_t i = 0; i < nvtxs; i++) {
            fprintf(stderr, " (%"PRIDX")", i);
            for (idx_t k = xadj[i]; k < xadj[i+1]; k++)
                fprintf(stderr, " %"PRIDX, adjncy[k]);
        }
        fprintf(stderr, "]\n");
    }
#endif

    idx_t * vwgt = NULL;    /* the weights of the vertices */
    idx_t * vsize = NULL;   /* the size of the vertices for computing
                             * the total communication volume */
    idx_t * adjwgt = NULL;  /* the weights of the edges */

    /* the number of parts to partition the graph */
    idx_t npartsi = nparts;

    /* This is an array of size nparts×ncon that specifies the desired
     * weight for each partition and constraint. The target partition
     * weight for the ith partition and jth constraint is specified at
     * tpwgts[i*ncon+j] (the numbering for both partitions and
     * constraints starts from 0). For each constraint, the sum of the
     * tpwgts[] entries must be 1.0. A NULL value can be passed to
     * indicate that the graph should be equally divided among the
     * partitions. */

    real_t * tpwgts = NULL;
    /* real_t * tpwgts = malloc((size_t)nparts*sizeof(*tpwgts)); */
    /* if (!tpwgts) { free(adjncy); free(xadj); return ACG_ERR_ERRNO; } */
    /* for (idx_t p = 0; p < nparts; p++) tpwgts[p] = 1.0 / nparts; */

    /* This is an array of size ncon that specifies the allowed load
     * imbalance tolerance for each constraint. For the ith partition
     * and jth constraint the allowed weight is the
     * ubvec[j]*tpwgts[i*ncon+j] fraction of the jth’s constraint
     * total weight. The load imbalances must be greater than 1.0.  A
     * NULL value can be passed indicating that the load imbalance
     * tolerance for each constraint should be 1.001 for ncon=1 or
     * 1.01 otherwise. */

    real_t * ubvec = NULL;
    /* real_t ubvec[1] = {1}; */

    /* edge-cut or the total communication volume of the partitioning
     * solution */
    idx_t objval = 0;

    /* This is a vector of size nvtxs that upon successful completion
     * stores the partition vector of the graph. The numbering of this
     * vector starts from either 0 or 1, depending on the value of
     * options[METIS OPTION NUMBERING]. */
    idx_t * part = NULL;
    bool free_part = false;
    if (sizeof(*dstpart) == sizeof(*part)) {
        part = (idx_t *) dstpart;
    } else {
        part = malloc((size_t)nvtxs*sizeof(*part));
        if (!part) { free(tpwgts); free(adjncy); free(xadj); return ACG_ERR_ERRNO; }
        free_part = true;
    }

    /* temporarily redirect stdandard output to standard error, so
     * that METIS prints its output to the standard error stream */
    int tmpfd = -1;
    if (verbose) {
        fflush(stdout);
        tmpfd = dup(STDOUT_FILENO);
        if (tmpfd != -1) dup2(STDERR_FILENO, STDOUT_FILENO);
    }

    /* invoke METIS to perform the partitioning */
    if (partitioner == metis_partgraphrecursive) {
        // int METIS_PartGraphRecursive(
        //     idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy,
        //     idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts,
        //     real_t *tpwgts, real_t ubvec,
        //     idx_t *options, idx_t *objval, idx_t *part);
        err = METIS_PartGraphRecursive(
            &nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
            &npartsi, tpwgts, ubvec, options, &objval, part);
    } else if (partitioner == metis_partgraphkway) {
        // int METIS_PartGraphKway(
        //     idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy,
        //     idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts,
        //     real_t *tpwgts, real_t ubvec,
        //     idx_t *options, idx_t *objval, idx_t *part);
        err = METIS_PartGraphKway(
            &nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
            &npartsi, tpwgts, ubvec, options, &objval, part);
    }

    if (tmpfd != -1) { fflush(stdout); dup2(tmpfd, STDOUT_FILENO); }

    if (err == METIS_ERROR_INPUT) { err = ACG_ERR_METIS_INPUT; }
    else if (err == METIS_ERROR_MEMORY) { err = ACG_ERR_METIS_MEMORY; }
    else if (err == METIS_ERROR) { err = ACG_ERR_METIS; }
    else { err = ACG_SUCCESS; }

    if (err) {
        if (free_part) free(part);
        free(tpwgts); free(adjncy); free(xadj);
        return err;
    }

    if (outobjval) *outobjval = objval;
    if (sizeof(*dstpart) != sizeof(*part)) {
        for (idx_t i = 0; i < N; i++) {
            if (part[i] > INT_MAX) {
                free(part); free(tpwgts); free(adjncy); free(xadj);
                errno = ERANGE;
                return ACG_ERR_ERRNO;
            }
            dstpart[i] = part[i];
        }
        free(part);
    }
    free(tpwgts); free(adjncy); free(xadj);

#if defined(DEBUG_METIS)
    if (verbose) {
        fprintf(stderr, "objval=%"PRIDX", part=[", objval);
        for (idx_t i = 0; i < nvtxs; i++) fprintf(stderr, " %"PRIDX, part[i]);
        fprintf(stderr, "]\n");
    }
#endif
    return ACG_SUCCESS;
#endif
}

/**
 * ‘metis_partgraph()’ uses the METIS k-way graph partitioner to
 * partition an undirected graph derived from a sparse matrix.
 *
 * Recursive bipartitioning or k-way partitioning is specified by
 * setting the ‘partitioner’ argument accordingly.
 *
 * The sparse matrix is provided in coordinate (COO) format with
 * dimensions given by ‘nrows’ and ‘ncols’. Furthermore, there are
 * ‘nnzs’ nonzero matrix entries, whose locations are specified by the
 * arrays ‘rowidx’ and ‘colidx’ (of length ‘nnzs’). The row offsets
 * are in the range ‘[0,nrows)’, whereas the column offsets are given
 * in the range are in the range ‘[0,ncols)’.
 *
 * The matrix may be unsymmetric or even non-square. Furthermore,
 * duplicate nonzero matrix entries are allowed, though they will be
 * removed when forming the undirected graph that is passed to the
 * METIS partitioner.
 *
 * If the matrix is square, then the graph to be partitioned is
 * obtained from the symmetrisation ‘A+A'’ of the matrix ‘A’ , where
 * ‘A'’ denotes the transpose of ‘A’.
 *
 * If the matrix is non-square, the partitioning algorithm is carried
 * out on a bipartite graph formed by the matrix rows and columns.
 * The adjacency matrix ‘B’ of the bipartite graph is square and
 * symmetric and takes the form of a 2-by-2 block matrix where ‘A’ is
 * placed in the upper right corner and ‘A'’ is placed in the lower
 * left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * As a result, the number of vertices in the graph is equal to
 * ‘nrows’ (and ‘ncols’) if the matrix is square. Otherwise, if the
 * matrix is non-square, then there are ‘nrows+ncols’ vertices.
 *
 * The array ‘dstrowpart’ must be of length ‘nrows’. This array is
 * used to store the part numbers assigned to the matrix rows. If the
 * matrix is non-square, then ‘dstcolpart’ must be an array of length
 * ‘ncols’, which is then similarly used to store the part numbers
 * assigned to the matrix columns.
 *
 * If it is not ‘NULL’, then ‘objval’ is used to store the value of
 * the objective function minimized by the partitioner, which, by
 * default, is the edge-cut of the partitioning solution.
 *
 * A seed for the random number generator can be specified through
 * ‘seed’. If ‘seed’ is set to -1, the default behaviour is used.
 */
int metis_partgraph(
    enum metis_partitioner partitioner,
    int nparts,
    acgidx_t nrows,
    acgidx_t ncols,
    acgidx_t nnzs,
    int rowidxstride,
    int rowidxbase,
    const acgidx_t * rowidx,
    int colidxstride,
    int colidxbase,
    const acgidx_t * colidx,
    int * dstrowpart,
    int * dstcolpart,
    acgidx_t * objval,
    acgidx_t seed,
    int verbose)
{
#ifndef ACG_HAVE_METIS
    return ACG_ERR_METIS_NOT_SUPPORTED;
#else
    bool square = nrows == ncols;
    acgidx_t N = square ? nrows : nrows + ncols;

    if (square) {
        /*
         * Handle unsymmetric matrices via symmetrisation: add all
         * nonzeros and their symmetric counterparts, then compact,
         * (i.e., sort and remove duplicates).
         */
        acgidx_t (* idx)[2] = malloc((size_t)nnzs*sizeof(acgidx_t[2]));
        if (!idx) return ACG_ERR_ERRNO;
        for (acgidx_t k = 0; k < nnzs; k++) {
            acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
            acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
            if (i <= j) { idx[k][0] = i; idx[k][1] = j; }
            else        { idx[k][0] = j; idx[k][1] = i; }
        }
        int err = acgradixsortpair_idx_t(
            nnzs, sizeof(*idx), &idx[0][0], sizeof(*idx), &idx[0][1],
            NULL, NULL);
        if (err) { free(idx); return err; }
        acgidx_t i = 0, k = 0;
        while (i < nnzs) {
            idx[k][0] = idx[i][0]; idx[k][1] = idx[i][1]; k++;
            for (i++; i < nnzs && idx[i][0] == idx[i-1][0] && idx[i][1] == idx[i-1][1]; i++) {}
        }

        err = metis_partgraphsym(
            partitioner, nparts, N, nnzs, sizeof(*idx), 0, &idx[0][0],
            sizeof(*idx), 0, &idx[0][1], dstrowpart, objval, seed, verbose);
        if (err) { free(idx); return err; }
        free(idx);
    } else {
        /*
         * Handle non-square matrices by partitioning the bipartite
         * graph whose vertices are the rows and columns of the
         * matrix. This requires shifting the column indices to the
         * right by an offset equal to the number of matrix rows.
         */
        int * dstpart = malloc((size_t)N*sizeof(*dstpart));
        if (!dstpart) return ACG_ERR_ERRNO;
        acgidx_t * tmpcolidx = malloc((size_t)nnzs*sizeof(*tmpcolidx));
        if (!tmpcolidx) { free(dstpart); return ACG_ERR_ERRNO; }
        for (acgidx_t k = 0; k < nnzs; k++) {
            acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
            tmpcolidx[k] = nrows+j;
        }
        int err = metis_partgraphsym(
            partitioner, nparts, N, nnzs, rowidxstride, rowidxbase, rowidx,
            sizeof(*tmpcolidx), 0, tmpcolidx, dstpart, objval, seed, verbose);
        if (err) { free(tmpcolidx); free(dstpart); return err; }
        free(tmpcolidx);
        for (acgidx_t i = 0; i < nrows; i++) dstrowpart[i] = dstpart[i];
        for (acgidx_t j = 0; j < ncols; j++) dstcolpart[j] = dstpart[nrows+j];
        free(dstpart);
    }
    return ACG_SUCCESS;
#endif
}

/**
 * ‘metis_ndsym()’ uses METIS to compute a multilevel nested
 * dissection ordering of an undirected graph given as a square,
 * symmetric matrix in coordinate format.
 *
 * The undirected graph is described in terms of a symmetric adjacency
 * matrix in coordinate (COO) format with ‘N’ rows and columns. There
 * are ‘nnzs’ nonzero matrix entries. The locations of the matrix
 * nonzeros are specified by the arrays ‘rowidx’ and ‘colidx’, both of
 * which are of length ‘nnzs’, and contain offsets in the range
 * ‘[0,N)’. Note that there should not be any duplicate nonzero
 * entries. The nonzeros may be located in the upper or lower triangle
 * of the adjacency matrix. However, if there is a nonzero entry at
 * row ‘i’ and column ‘j’, then there should not be a nonzero entry
 * row ‘j’ and column ‘i’. (Although both nonzeros are required in the
 * undirected graph data structure passed to METIS, as described in
 * Section 5.5 of the METIS manual, the required nonzeros will be
 * added by ‘metis_ndsym()’ before calling METIS.)
 *
 * The values ‘rowidxstride’ and ‘colidxstride’ may be used to specify
 * strides (in bytes) that are used when accessing the row and column
 * offsets in ‘rowidx’ and ‘colidx’, respectively. This is useful for
 * cases where the row and column offsets are not necessarily stored
 * contiguously in memory.
 *
 * On success, the arrays ‘perm’ and ‘perminv’ contain the permutation
 * and inverse permutation of the graph vertices. Therefore, ‘perm’
 * and ‘perminv’ must be arrays of length ‘N’.
 */
int metis_ndsym(
    acgidx_t N,
    acgidx_t nnzs,
    int rowidxstride,
    int rowidxbase,
    const acgidx_t * rowidx,
    int colidxstride,
    int colidxbase,
    const acgidx_t * colidx,
    int * dstperm,
    int * dstperminv,
    int verbose)
{
#ifndef ACG_HAVE_METIS
    return ACG_ERR_METIS_NOT_SUPPORTED;
#else
    if (N > IDX_MAX-1) return ACG_ERR_METIS_EOVERFLOW;
    if (nnzs > IDX_MAX) return ACG_ERR_METIS_EOVERFLOW;

    /* perform some bounds checking on the input graph */
    for (acgidx_t k = 0; k < nnzs; k++) {
        acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
        acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
        if (i < 0 || i >= N || j < 0 || j >= N) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* configure reordering options */
    idx_t options[METIS_NOPTIONS]; /* array of options */
    int err = METIS_SetDefaultOptions(options);
    if (err == METIS_ERROR_INPUT) { return ACG_ERR_METIS_INPUT; }
    else if (err == METIS_ERROR_MEMORY) { return ACG_ERR_METIS_MEMORY; }
    else if (err == METIS_ERROR) { return ACG_ERR_METIS; }

    /* METIS_CTYPE_RM for random matching or METIS_CTYPE_SHEM for
     * sorted heavy-edge matching (default) */
    // options[METIS_OPTION_CTYPE] = METIS_CTYPE_SHEM;

    /*
     * METIS_RTYPE_FM - FM-based cut refinement
     * METIS_RTYPE_GREEDY - greedy-based cut and volume refinement
     * METIS_RTYPE_SEP2SIDED - two-sided node FM refinement
     * METIS_RTYPE_SEP1SIDED - One-sided node FM refinement (default)
     */
    // options[METIS_OPTION_RTYPE] = METIS_RTYPE_SEP1SIDED;

    /* 0 (default) performs a 2–hop matching or 1 does not */
    // options[METIS_OPTION_NO2HOP] = 0;

    /* Specifies the number of different separators that it will
     * compute at each level of nested dissection. The final separator
     * that is used is the smallest one. Default is 1. */
    // options[METIS_OPTION_NSEPS] = 1;

    /* The number of iterations for the refinement algorithms at each
     * stage of the uncoarsening process. Default is 10. */
    // options[METIS_OPTION_NITER] = 10;

    /* Specifies the maximum allowed load imbalance among the
     * partitions. A value of x indicates that the allowed load
     * imbalance is (1 + x)/1000. The load imbalance for the jth
     * constraint is defined to be max_i(w[j,i])/t[j,i]), where w[j,i]
     * is the fraction of the overall weight of the jth constraint
     * that is assigned to the ith partition and t[j, i] is the
     * desired target weight of the jth constraint for the ith
     * partition (i.e., that specified via -tpwgts). For -ptype=rb,
     * the default value is 1 (i.e., load imbalance of 1.001) and for
     * -ptype=kway, the default value is 30 (i.e., load imbalance of
     * 1.03). */
    // options[METIS_OPTION_UFACTOR] = 30;

    /* Specifies that the graph should be compressed by combining
     * together vertices that have identical adjacency lists.
     *
     * 0 Does not try to compress the graph.
     * 1 Tries to compress the graph. (default) */
    // options[METIS_OPTION_COMPRESS] = 0;

    /* Specifies if the connected components of the graph should first
     * be identified and ordered separately.
     *
     *  0 Does not identify the connected components. (default)
     *  1 Identifies the connected components. */
    // options[METIS_OPTION_CCORDER] = 1;

    /* seed for the random number generator */
    // options[METIS_OPTION_SEED] = 0;

    /*
     * Specifies the minimum degree of the vertices that will be
     * ordered last. If the specified value is x > 0, then any
     * vertices with a degree greater than 0.1*x*(average degree) are
     * removed from the graph, an ordering of the rest of the vertices
     * is computed, and an overall ordering is computed by ordering
     * the removed vertices at the end of the overall ordering. For
     * example if x = 40, and the average degree is 5, then the
     * algorithm will remove all vertices with degree greater than
     * 20. The vertices that are removed are ordered last (i.e., they
     * are automatically placed in the top-level separator). Good
     * values are often in the range of 60 to 200 (i.e., 6 to 20 times
     * more than the average). Default value is 0, indicating that no
     * vertices are removed.
     *
     * Used to control whether or not the ordering algorithm should
     * remove any vertices with high degree (i.e., dense
     * columns). This is particularly helpful for certain classes of
     * LP matrices, in which there a few vertices that are connected
     * to many other vertices. By removing these vertices prior to
     * ordering, the quality and the amount of time required to do the
     * ordering improves.
     */
    // options[METIS_OPTION_PFACTOR] = 0;

    /* 0 for C-style numbering that starts from 0 (default), or 1 for
     * Fortran-style numbering that starts from 1 */
    // options[METIS_OPTION_NUMBERING] = 0;

    /*
     * METIS_DBG_INFO (1)
     * METIS_DBG_TIME (2)
     * METIS_DBG_COARSEN (4)
     * METIS_DBG_REFINE (8)
     * METIS_DBG_IPART (16)
     * METIS_DBG_MOVEINFO (32)
     * METIS_DBG_SEPINFO (64)
     * METIS_DBG_CONNINFO (128)
     * METIS_DBG_CONTIGINFO (256)
     */
    options[METIS_OPTION_DBGLVL] = 0;
    if (verbose > 0) options[METIS_OPTION_DBGLVL] |= METIS_DBG_INFO | METIS_DBG_TIME;
    if (verbose > 1) {
        options[METIS_OPTION_DBGLVL] |= (
            METIS_DBG_COARSEN |
            METIS_DBG_REFINE |
            METIS_DBG_IPART |
            METIS_DBG_MOVEINFO |
            METIS_DBG_SEPINFO |
            METIS_DBG_CONNINFO |
            METIS_DBG_CONTIGINFO);
    }

    /* the number of vertices in the graph */
    idx_t nvtxs = N;

    /* adjacency structure of the graph (row pointers) */
    idx_t * xadj = malloc((size_t)(nvtxs+1)*sizeof(*xadj));
    if (!xadj) return ACG_ERR_ERRNO;
    for (idx_t i = 0; i <= nvtxs; i++) xadj[i] = 0;
    for (acgidx_t k = 0; k < nnzs; k++) {
        acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
        acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
        if (i != j) { xadj[i+1]++; xadj[j+1]++; }
    }
    for (idx_t i = 1; i <= nvtxs; i++) {
        if (xadj[i] > IDX_MAX - xadj[i-1]) { free(xadj); return ACG_ERR_METIS_EOVERFLOW; }
        xadj[i] += xadj[i-1];
    }

    /* adjacency structure of the graph (column offsets) */
    idx_t * adjncy = malloc((size_t)xadj[nvtxs]*sizeof(*adjncy));
    if (!adjncy) { free(xadj); return ACG_ERR_ERRNO; }

    for (acgidx_t k = 0; k < nnzs; k++) {
        acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
        acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
        if (i != j) { adjncy[xadj[i]++] = j; adjncy[xadj[j]++] = i; }
    }
    for (idx_t i = nvtxs; i > 0; i--) xadj[i] = xadj[i-1];
    xadj[0] = 0;

#ifdef DEBUG_METIS
    fprintf(stderr, "nvtxs=%"PRIDX", xadj=[", nvtxs);
    for (idx_t i = 0; i <= nvtxs; i++) fprintf(stderr, " %"PRIDX, xadj[i]);
    fprintf(stderr, "]\n");
    fprintf(stderr, "adjncy=[");
    for (idx_t i = 0; i < nvtxs; i++) {
        fprintf(stderr, " (%"PRIDX")", i);
        for (idx_t k = xadj[i]; k < xadj[i+1]; k++)
            fprintf(stderr, " %"PRIDX, adjncy[k]);
    }
    fprintf(stderr, "]\n");
#endif

    idx_t * vwgt = NULL;    /* the weights of the vertices */

    /* These are vectors, each of size nvtxs that upon successful
     * completion store the permutation and inverse permutation
     * vectors of the fill-reducing graph ordering. The numbering of
     * this vector starts from either 0 or 1, depending on the value
     * of options[METIS OPTION NUMBERING]. */
    idx_t * perm = NULL;
    idx_t * perminv = NULL;
    bool free_perm = false;
    if (sizeof(*dstperm) == sizeof(*perm)) {
        perm = (idx_t *) dstperm;
        perminv = (idx_t *) dstperminv;
    } else {
        perm = malloc((size_t)nvtxs*sizeof(*perm));
        if (!perm) { free(adjncy); free(xadj); return ACG_ERR_ERRNO; }
        perminv = malloc((size_t)nvtxs*sizeof(*perminv));
        if (!perminv) { free(perm); free(adjncy); free(xadj); return ACG_ERR_ERRNO; }
        free_perm = true;
    }

    /* temporarily redirect stdandard output to standard error, so
     * that METIS prints its output to the standard error stream */
    int tmpfd = -1;
    if (verbose) {
        fflush(stdout);
        tmpfd = dup(STDOUT_FILENO);
        if (tmpfd != -1) dup2(STDERR_FILENO, STDOUT_FILENO);
    }

    // int METIS_NodeND(
    //     idx_t *nvtxs, idx_t *xadj, idx_t *adjncy,
    //     idx_t *vwgt, idx_t *options, idx_t *perm, idx_t *iperm);

    /* note that the meaning of METIS's perm and iperm are reversed
     * compared to the terminology we use for permutation and inverse
     * permutation of graph vertices */
    err = METIS_NodeND(
        &nvtxs, xadj, adjncy, vwgt, options, perminv, perm);

    if (tmpfd != -1) { fflush(stdout); dup2(tmpfd, STDOUT_FILENO); }

    if (err == METIS_ERROR_INPUT) { err = ACG_ERR_METIS_INPUT; }
    else if (err == METIS_ERROR_MEMORY) { err = ACG_ERR_METIS_MEMORY; }
    else if (err == METIS_ERROR) { err = ACG_ERR_METIS; }
    else { err = ACG_SUCCESS; }

    if (err) {
        if (free_perm) { free(perminv); free(perm); }
        free(adjncy); free(xadj);
        return err;
    }

    if (sizeof(*dstperm) != sizeof(*perm)) {
        for (idx_t i = 0; i < N; i++) {
            if (perm[i] > INT_MAX) {
                free(perminv); free(perm); free(adjncy); free(xadj);
                errno = ERANGE;
                return ACG_ERR_ERRNO;
            }
            dstperm[i] = perm[i];
            dstperminv[i] = perminv[i];
        }
        free(perminv); free(perm);
    }
    free(adjncy); free(xadj);
    return ACG_SUCCESS;
#endif
}

/**
 * ‘metis_nd()’ uses METIS to compute a multilevel nested dissection
 * ordering of an undirected graph derived from a sparse matrix.
 *
 * The sparse matrix is provided in coordinate (COO) format with
 * dimensions given by ‘nrows’ and ‘ncols’. Furthermore, there are
 * ‘nnzs’ nonzero matrix entries, whose locations are specified by the
 * arrays ‘rowidx’ and ‘colidx’ (of length ‘nnzs’). The row offsets
 * are in the range ‘[0,nrows)’, whereas the column offsets are given
 * in the range are in the range ‘[0,ncols)’.
 *
 * The matrix may be unsymmetric or even non-square. Furthermore,
 * duplicate nonzero matrix entries are allowed, though they will be
 * removed when forming the undirected graph that is passed to METIS.
 *
 * If the matrix is square, then the graph to be reordered is obtained
 * from the symmetrisation ‘A+A'’ of the matrix ‘A’ , where ‘A'’
 * denotes the transpose of ‘A’.
 *
 * If the matrix is non-square, the reordering algorithm is carried
 * out on a bipartite graph formed by the matrix rows and columns.
 * The adjacency matrix ‘B’ of the bipartite graph is square and
 * symmetric and takes the form of a 2-by-2 block matrix where ‘A’ is
 * placed in the upper right corner and ‘A'’ is placed in the lower
 * left corner:
 *
 *     ⎡  0   A ⎤
 * B = ⎢        ⎥.
 *     ⎣  A'  0 ⎦
 *
 * As a result, the number of vertices in the graph is equal to
 * ‘nrows’ (and ‘ncols’) if the matrix is square. Otherwise, if the
 * matrix is non-square, then there are ‘nrows+ncols’ vertices.
 *
 * The arrays ‘rowperm’ and ‘rowperminv’ must be of length
 * ‘nrows’. These arrays are used to store the permutation and inverse
 * permutation of the matrix rows. If the matrix is non-square, then
 * ‘colperm’ and ‘colperminv’ must be arrays of length ‘ncols’, which
 * are then similarly used to store the permutation and inverse
 * permutation of the matrix columns.
 */
int metis_nd(
    acgidx_t nrows,
    acgidx_t ncols,
    acgidx_t nnzs,
    int rowidxstride,
    int rowidxbase,
    const acgidx_t * rowidx,
    int colidxstride,
    int colidxbase,
    const acgidx_t * colidx,
    int * rowperm,
    int * rowperminv,
    int * colperm,
    int * colperminv,
    int verbose)
{
#ifndef ACG_HAVE_METIS
    return ACG_ERR_METIS_NOT_SUPPORTED;
#else
    bool square = nrows == ncols;
    acgidx_t N = square ? nrows : nrows + ncols;

    if (square) {
        /*
         * Handle unsymmetric matrices via symmetrisation: add all
         * nonzeros and their symmetric counterparts, then compact,
         * (i.e., sort and remove duplicates).
         */
        acgidx_t (* idx)[2] = malloc((size_t)nnzs*sizeof(acgidx_t[2]));
        if (!idx) return ACG_ERR_ERRNO;
        for (acgidx_t k = 0; k < nnzs; k++) {
            acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase;
            acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
            if (i <= j) { idx[k][0] = i; idx[k][1] = j; }
            else        { idx[k][0] = j; idx[k][1] = i; }
        }
        int err = acgradixsortpair_idx_t(
            nnzs, sizeof(*idx), &idx[0][0], sizeof(*idx), &idx[0][1],
            NULL, NULL);
        if (err) { free(idx); return err; }
        acgidx_t i = 0, k = 0;
        while (i < nnzs) {
            idx[k][0] = idx[i][0]; idx[k][1] = idx[i][1]; k++;
            for (i++; i < nnzs && idx[i][0] == idx[i-1][0] && idx[i][1] == idx[i-1][1]; i++) {}
        }

        /* /\* */
        /*  * Handle unsymmetric matrices via symmetrisation: add all */
        /*  * nonzeros and their symmetric counterparts, then compact, */
        /*  * (i.e., sort and remove duplicates). */
        /*  *\/ */
        /* acgidx_t (* idx)[2] = malloc((size_t)nnzs*sizeof(acgidx_t[2])); */
        /* if (!idx) return ACG_ERR_ERRNO; */
        /* for (acgidx_t k = 0; k < nnzs; k++) { */
        /*     acgidx_t i = *(const acgidx_t *) ((const char *) rowidx + (size_t)k*rowidxstride)-rowidxbase; */
        /*     acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase; */
        /*     if (i <= j) { idx[k][0] = i; idx[k][1] = j; } */
        /*     else        { idx[k][0] = j; idx[k][1] = i; } */
        /* } */
        /* int err = compact_unsorted_int64_pair( */
        /*     &nnzs, idx, nnzs, idx); */
        /* if (err) { free(idx); return ACG_ERR_ERRNO; } */
        err = metis_ndsym(
            N, nnzs, sizeof(*idx), 0, &idx[0][0],
            sizeof(*idx), 0, &idx[0][1], rowperm, rowperminv, verbose);
        if (err) { free(idx); return err; }
        free(idx);
    } else {
        /*
         * Handle non-square matrices by reordering the bipartite
         * graph whose vertices are the rows and columns of the
         * matrix. This requires shifting the column indices to the
         * right by an offset equal to the number of matrix rows.
         */
        int * perm = malloc((size_t)N*sizeof(*perm));
        if (!perm) return ACG_ERR_ERRNO;
        int * perminv = malloc((size_t)N*sizeof(*perminv));
        if (!perminv) { free(perm); return ACG_ERR_ERRNO; }
        acgidx_t * tmpcolidx = malloc((size_t)nnzs*sizeof(*tmpcolidx));
        if (!tmpcolidx) { free(perminv); free(perm); return ACG_ERR_ERRNO; }
        for (acgidx_t k = 0; k < nnzs; k++) {
            acgidx_t j = *(const acgidx_t *) ((const char *) colidx + (size_t)k*colidxstride)-colidxbase;
            tmpcolidx[k] = nrows+j;
        }
        int err = metis_ndsym(
            N, nnzs, rowidxstride, rowidxbase, rowidx,
            sizeof(*tmpcolidx), 0, tmpcolidx, perm, perminv, verbose);
        if (err) { free(tmpcolidx); free(perminv); free(perm); return err; }
        free(tmpcolidx);
        int k = 0, l = 0;
        for (acgidx_t i = 0; i < N; i++) {
            if (perm[i] < nrows) { rowperm[k++] = perm[i]; }
            else { colperm[l++] = perm[i]-nrows; }
        }
        for (acgidx_t i = 0; i < nrows; i++) rowperminv[rowperm[i]] = i;
        for (acgidx_t j = 0; j < ncols; j++) colperminv[colperm[j]] = j;
        free(perminv); free(perm);
    }
    return ACG_SUCCESS;
#endif
}
