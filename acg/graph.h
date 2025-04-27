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
 * partitioned and distributed graphs
 */

#ifndef ACG_GRAPH_H
#define ACG_GRAPH_H

#include "acg/config.h"
#include "acg/metis.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#include <stdint.h>
#include <stdio.h>

/**
 * ‘acggraph’ represents an undirected graph in the form of one or
 * more partitioned subgraphs that may be distributed across one or
 * more processes.
 *
 * Each subgraph is stored in the form of a symmetric adjacency matrix
 * with the upper triangular entries stored in a compressed sparse row
 * format.
 */
struct acggraph
{
    /*
     * partitioning information
     */

    /**
     * ‘nparts’ is the total number of partitions of the graph (i.e.,
     * the total number of subgraphs).
     */
    int nparts;

    /**
     * ‘parttag’ is a tag that may be used to identify the subgraph.
     */
    int parttag;

    /**
     * ‘nprocs’ is the number of processes among which the graph is
     * distributed.
     */
    int nprocs;

    /**
     * ‘npparts’ is the number of subgraphs assigned to the current
     * process.
     */
    int npparts;

    /**
     * ‘ownerrank’ is the rank of the process that owns the subgraph,
     * ranging from ‘0’ up to ‘nprocs-1’.
     */
    int ownerrank;

    /**
     * ‘ownerpart’ is the partition number of the subgraph among the
     * subgraphs assigned to the current process rank, ranging from
     * ‘0’ up to ‘npparts-1’.
     */
    int ownerpart;

    /*
     * graph/subgraph nodes
     */

    /**
     * ‘nnodes’ is the total number of nodes in the graph.
     */
    acgidx_t nnodes;

    /**
     * ‘npnodes’ is the number of nodes in the subgraph.
     */
    acgidx_t npnodes;

    /**
     * ‘nodetags’ is an array of length ‘npnodes’ containing tags for
     * each node in the subgraph. If there are no node tags, then it
     * is set to ‘NULL’.
     */
    acgidx_t * nodetags;

    /**
     * ‘parentnodeidx’ is an array of length ‘npnodes’. Partitioning
     * the graph causes the nodes to be reordered, this array is used
     * to store the original index of each node in the graph prior to
     * partitioning.
     */
    acgidx_t * parentnodeidx;

    /*
     * graph/subgraph edges
     */

    /**
     * ‘nedges’ is the total number of edges in the graph.
     */
    int64_t nedges;

    /**
     * ‘npedges’ is the number of edges in the subgraph.
     */
    int64_t npedges;

    /**
     * ‘edgetags’ is an array of length ‘npedges’ containing tags for
     * each edge in the subgraph. If there are no edge tags, then it
     * is set to ‘NULL’.
     */
    int64_t * edgetags;

    /**
     * ‘parentedgeidx’ is an array of length ‘npedges’. Partitioning
     * the graph causes the edges to be reordered, this array is used
     * to store the original index of each edge in the graph prior to
     * partitioning.
     */
    int64_t * parentedgeidx;

    /*
     * incidence relation mapping edges to unordered pairs of nodes
     */

    /**
     * ‘nodeidxbase’ is the base (0 or 1) used for numbering nodes in
     * the incidence relation. More specifically, the node numbers
     * stored in ‘dstnodeidx’ range from ‘nodeidxbase’ up to
     * ‘npnodes+nodeidxbase-1’.
     */
    int nodeidxbase;

    /**
     * ‘nodenedges’ is an array of length ‘npnodes’ containing the
     * number of edges for each node.
     */
    int64_t * nodenedges;

    /**
     * ‘srcnodeptr’ is an array of length ‘npnodes+1’ containing the
     * offsets to the first edge of each node. More specifically, if
     * the i-th node, ‘uᵢ’, has one or more edges, ‘(uᵢ,v₁)’,
     * ‘(uᵢ,v₂)’, ..., then the node number of the second node in the
     * k-th edge pair is stored at ‘dstnodeidx[srcnodeptr[i]+k]’.
     */
    int64_t * srcnodeptr;

    /**
     * ‘srcnodeidx’ is an array of length ‘npedges’ containing the
     * node number of the first node ‘u’ in the pair ‘(u,v)’ for every
     * edge of the subgraph.
     */
    acgidx_t * srcnodeidx;

    /**
     * ‘dstnodeidx’ is an array of length ‘npedges’ containing the
     * node number of the second node ‘v’ in the pair ‘(u,v)’ for
     * every edge of the subgraph.
     */
    acgidx_t * dstnodeidx;

    /*
     * interior, border and ghost nodes
     */

    /**
     * ‘nownednodes’ is the number of nodes in the subgraph that are
     * owned by the subgraph according to the node partitioning, which
     * is equal to the number of interior and border nodes in the
     * subgraph.
     */
    acgidx_t nownednodes;

    /**
     * ‘ninnernodes’ is the number of interior nodes in the subgraph,
     * meaning that neighbourhing nodes are owned by subgraph.
     */
    acgidx_t ninnernodes;

    /**
     * ‘nbordernodes’ is the number of border nodes in the subgraph,
     * meaning that the nodes are owned by the subgraph, but their
     * neighbourhood contains one or more nodes that are owned by
     * another subgraph of the partitioned graph.
     */
    acgidx_t nbordernodes;

    /**
     * ‘bordernodeoffset’ is an offset to the first border node
     * according to the numbering of nodes in the subgraph, where
     * interior nodes are grouped before border nodes, which again are
     * ordered before ghost nodes.
     */
    acgidx_t bordernodeoffset;

    /**
     * ‘nghostnodes’ is the number of ghost nodes in the subgraph,
     * meaning that the nodes are not owned by the subgraph, but they
     * belong to the neighbourhood of one or more (border) nodes owned
     * by the subgraph.
     */
    acgidx_t nghostnodes;

    /**
     * ‘ghostnodeoffset’ is an offset to the first ghost node
     * according to the numbering of nodes in the subgraph, where
     * interior nodes are grouped before ghost nodes, which again are
     * ordered before ghost nodes.
     */
    acgidx_t ghostnodeoffset;

    /* interior and interface edges */

    /**
     * ‘ninneredges’ is the number of interior edges in the subgraph,
     * such that both nodes of the edge are owned by the subgraph.
     */
    int64_t ninneredges;

    /**
     * ‘nneighouredges’ is the number of border edges in the subgraph,
     * such that one of its nodes is owned by a neighbouring subgraph.
     */
    int64_t ninterfaceedges;

    /**
     * ‘nbordernodeinneredges’ is an array of length ‘nbordernodes’
     * containing the number of interior edges for every border node
     * in the subgraph.
     */
    int64_t * nbordernodeinneredges;

    /**
     * ‘nbordernodeinterfaceedges’ is an array of length
     * ‘nbordernodes’ containing the number of interface edges for
     * every border node in the subgraph.
     */
    int64_t * nbordernodeinterfaceedges;

    /*
     * neighbouring subgraphs
     */

    /**
     * ‘nneighbours’ is the number of neighbouring subgraphs, i.e.,
     * subgraphs with one or more ghost nodes owned by this subgraph.
     */
    int nneighbours;

    /**
     * ‘acggraphneighbour’ is a data structure for storing
     * information about the neighbouring subgraphs of a given
     * subgraph in a partitioned (and distributed) graph.
     */
    struct acggraphneighbour {
        /**
         * ‘neighbourranks’ is the rank of the process that owns the
         * neigbhouring subgraph.
         */
        int neighbourrank;

        /**
         * ‘neighbourpart’ is the part number of the neigbhouring
         * subgraph with respect to the numbering of subgraphs of the
         * owning process.
         */
        int neighbourpart;

        /**
         * ‘nbordernodes’ is the number of border nodes of the
         * subgraph with neighbouring nodes in the neighbouring
         * subgraph.
         */
        acgidx_t nbordernodes;

        /**
         * ‘bordernodes’ is an array of length ‘nbordernodes’
         * containing the index of every border node in the subgraph
         * with neighbouring nodes in the neighbouring subgraph.
         */
        acgidx_t * bordernodes;

        /**
         * ‘nghostnodes’ is the number of ghost nodes of the subgraph
         * with neighbouring nodes in the neighbouring subgraph.
         */
        acgidx_t nghostnodes;

        /**
         * ‘ghostnodes’ is an array of length ‘nghostnodes’ containing
         * the index of every ghost node in the subgraph with
         * neighbouring (border) nodes in the neibghbouring subgraph.
         */
        acgidx_t * ghostnodes;
    } *neighbours;
};

/**
 * ‘acggraph_init()’ creates a graph from a given adjacency matrix.
 * The graph initially consists of a single part belonging to a single
 * process, and is therefore not distributed.
 */
int acggraph_init(
    struct acggraph * graph,
    acgidx_t nnodes,
    const acgidx_t * nodetags,
    int64_t nedges,
    const int64_t * edgetags,
    int idxbase,
    const int64_t * srcnodeptr,
    const acgidx_t * dstnodeidx);

/**
 * ‘acggraph_free()’ frees resources associated with a graph.
 */
void acggraph_free(
    struct acggraph * graph);

/**
 * ‘acggraph_copy()’ copies a graph.
 */
int acggraph_copy(
    struct acggraph * dst,
    const struct acggraph * src);

/*
 * output (e.g., for debugging)
 */

int acggraph_fwrite(
    FILE * f,
    const struct acggraph * graph);

/*
 * graph partitioning
 */

/**
 * ‘acggraph_partition_nodes()’ partitions the nodes of a graph into
 * the given number of parts.
 *
 * The graph nodes are partitioned using the METIS partitioner.
 */
int acggraph_partition_nodes(
    const struct acggraph * graph,
    int nparts,
    enum metis_partitioner partitioner,
    int * nodeparts,
    acgidx_t * objval,
    acgidx_t seed,
    int verbose);

/**
 * ‘acggraph_partition()’ partitions a graph into a given number of
 * subgraphs based on a given partitioning of its nodes.
 */
int acggraph_partition(
    const struct acggraph * srcgraph,
    int nparts,
    const int * nodeparts,
    const int * parttags,
    struct acggraph * subgraphs,
    int verbose);

/*
 * distributing graphs
 */

#ifdef ACG_HAVE_MPI
int acggraph_send(
    const struct acggraph * graphs,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

int acggraph_recv(
    struct acggraph * graphs,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode);

int acggraph_scatter(
    struct acggraph * sendgraphs,
    int sendcount,
    struct acggraph * recvgraphs,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode);

int acggraph_scatterv(
    struct acggraph * sendgraphs,
    const int * sendcounts,
    const int * displs,
    struct acggraph * recvgraphs,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode);
#endif

/*
 * halo exchange/update for partitioned and distributed graphs
 */

struct acghalo;

/**
 * ‘acggraph_halo()’ sets up a halo exchange communication pattern
 * to send and receive data associated with the “ghost” nodes of a
 * partitioned and distributed graph.
 */
int acggraph_halo(
    const struct acggraph * graph,
    struct acghalo * halo);

#endif
