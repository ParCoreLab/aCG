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

#ifndef ACG_METIS_H
#define ACG_METIS_H

#include "acg/config.h"

#include <stdint.h>

enum metis_partitioner
{
    metis_partgraphrecursive, /* METIS_PartGraphRecursive */
    metis_partgraphkway       /* METIS_PartGraphKway */
};

/**
 * ‘metis_partgraphsym()’ uses the METIS graph partitioner to
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
 * The values ‘rowidxstride’ and ‘colidxstride’ may be used to specify
 * strides (in bytes) that are used when accessing the row and column
 * offsets in ‘rowidx’ and ‘colidx’, respectively. This is useful for
 * cases where the row and column offsets are not necessarily stored
 * contiguously in memory.
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
    acgidx_t * objval,
    acgidx_t seed,
    int verbose);

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
    int verbose);

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
    int * perm,
    int * perminv,
    int verbose);

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
    int verbose);

#endif
