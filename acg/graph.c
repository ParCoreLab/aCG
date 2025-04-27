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

#include "acg/config.h"
#include "acg/graph.h"
#include "acg/error.h"
#include "acg/halo.h"
#include "acg/prefixsum.h"
#include "acg/sort.h"
#include "acg/time.h"

#ifdef ACG_HAVE_OPENMP
#include <omp.h>
#endif

#include <errno.h>
#include <unistd.h>

#include <stdbool.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

/**
 * ‘acggraph_init()’ creates a graph from a given adjacency matrix.
 * The graph initially consists of a single part belonging to a single
 * process, and is therefore not distributed.
 */
int acggraph_init(
    struct acggraph * g,
    acgidx_t nnodes,
    const acgidx_t * nodetags,
    int64_t nedges,
    const int64_t * edgetags,
    int idxbase,
    const int64_t * srcnodeptr,
    const acgidx_t * dstnodeidx)
{
    /* partitioning information */
    g->nparts = 1;
    g->parttag = 1;
    g->nprocs = 1;
    g->npparts = 1;
    g->ownerrank = 0;
    g->ownerpart = 0;

    /* graph/subgraph nodes */
    g->nnodes = nnodes;
    g->npnodes = nnodes;
    if (nodetags) {
        g->nodetags = malloc(nnodes*sizeof(*g->nodetags));
        if (!g->nodetags) return ACG_ERR_ERRNO;
        #pragma omp parallel for
        for (acgidx_t i = 0; i < nnodes; i++) g->nodetags[i] = nodetags[i];
    } else { g->nodetags = NULL; }
    g->parentnodeidx = NULL;

    /* graph/subgraph edges */
    g->nedges = nedges;
    g->npedges = nedges;
    if (edgetags) {
        g->edgetags = malloc(nedges*sizeof(*g->edgetags));
        if (!g->edgetags) return ACG_ERR_ERRNO;
        #pragma omp parallel for
        for (int64_t k = 0; k < nedges; k++) g->edgetags[k] = edgetags[k];
    } else { g->edgetags = NULL; }
    g->parentedgeidx = NULL;

    /* incidence relation - edges to unordered node pairs */
    g->nodeidxbase = idxbase;
    g->nodenedges = malloc(nnodes*sizeof(*g->nodenedges));
    if (!g->nodenedges) {
        free(g->parentedgeidx); free(g->edgetags);
        free(g->parentnodeidx); free(g->nodetags);
        return ACG_ERR_ERRNO;
    }
    #pragma omp parallel for
    for (acgidx_t i = 0; i < nnodes; i++)
        g->nodenedges[i] = srcnodeptr[i+1]-srcnodeptr[i];
    g->srcnodeptr = malloc((nnodes+1)*sizeof(*g->srcnodeptr));
    if (!g->srcnodeptr) {
        free(g->nodenedges);
        free(g->parentedgeidx); free(g->edgetags);
        free(g->parentnodeidx); free(g->nodetags);
        return ACG_ERR_ERRNO;
    }
    if (nnodes == 0 && !srcnodeptr) g->srcnodeptr[0] = 0;
    else {
        #pragma omp parallel for
        for (acgidx_t i = 0; i <= nnodes; i++)
            g->srcnodeptr[i] = srcnodeptr[i];
    }
    g->srcnodeidx = malloc(nedges*sizeof(*g->srcnodeidx));
    if (!g->srcnodeidx) {
        free(g->srcnodeptr); free(g->nodenedges);
        free(g->parentedgeidx); free(g->edgetags);
        free(g->parentnodeidx); free(g->nodetags);
        return ACG_ERR_ERRNO;
    }
    #pragma omp parallel for
    for (acgidx_t i = 0; i < nnodes; i++) {
        for (int64_t k = g->srcnodeptr[i]; k < g->srcnodeptr[i+1]; k++)
            g->srcnodeidx[k] = i+idxbase;
    }
    g->dstnodeidx = malloc(nedges*sizeof(*g->dstnodeidx));
    if (!g->dstnodeidx) {
        free(g->srcnodeidx); free(g->srcnodeptr); free(g->nodenedges);
        free(g->parentedgeidx); free(g->edgetags);
        free(g->parentnodeidx); free(g->nodetags);
        return ACG_ERR_ERRNO;
    }
    #pragma omp parallel for
    for (int64_t k = 0; k < nedges; k++) g->dstnodeidx[k] = dstnodeidx[k];

    /* interior, border and ghost nodes */
    g->nownednodes = nnodes;
    g->ninnernodes = nnodes;
    g->nbordernodes = 0;
    g->bordernodeoffset = nnodes;
    g->nghostnodes = 0;
    g->ghostnodeoffset = nnodes;

    /* interior, border and ghost edges */
    g->ninneredges = nedges;
    g->ninterfaceedges = 0;
    g->nbordernodeinneredges = NULL;
    g->nbordernodeinterfaceedges = NULL;

    /* neighbouring subgraphs */
    g->nneighbours = 0;
    g->neighbours = NULL;
    return ACG_SUCCESS;
}

/**
 * ‘acggraphneighbour_free()’ frees resources associated with a
 * subgraph neighbour.
 */
static void acggraphneighbour_free(
    struct acggraphneighbour * g)
{
    free(g->bordernodes);
    free(g->ghostnodes);
}

/**
 * ‘acggraph_free()’ frees resources associated with a graph.
 */
void acggraph_free(
    struct acggraph * g)
{
    /* graph/subgraph nodes */
    free(g->nodetags);
    free(g->parentnodeidx);

    /* graph/subgraph edges */
    free(g->edgetags);
    free(g->parentedgeidx);

    /* incidence relation */
    free(g->nodenedges);
    free(g->srcnodeptr);
    free(g->srcnodeidx);
    free(g->dstnodeidx);

    /* interior, border and ghost edges */
    free(g->nbordernodeinneredges);
    free(g->nbordernodeinterfaceedges);

    /* neighbouring subgraphs */
    for (int i = 0; i < g->nneighbours; i++)
        acggraphneighbour_free(&g->neighbours[i]);
    free(g->neighbours);
}

/**
 * ‘acggraphneighbour_copy()’ copies a subgraph neighbour.
 */
static int acggraphneighbour_copy(
    struct acggraphneighbour * dst,
    const struct acggraphneighbour * src)
{
    dst->neighbourrank = src->neighbourrank;
    dst->neighbourpart = src->neighbourpart;
    dst->nbordernodes = src->nbordernodes;
    dst->bordernodes = malloc(dst->nbordernodes*sizeof(*dst->bordernodes));
    if (!dst->bordernodes) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->nbordernodes; i++)
        dst->bordernodes[i] = src->bordernodes[i];
    dst->nghostnodes = src->nghostnodes;
    dst->ghostnodes = malloc(dst->nghostnodes*sizeof(*dst->ghostnodes));
    if (!dst->ghostnodes) return ACG_ERR_ERRNO;
    for (acgidx_t i = 0; i < dst->nghostnodes; i++)
        dst->ghostnodes[i] = src->ghostnodes[i];
    return ACG_SUCCESS;
}

/**
 * ‘acggraph_copy()’ copies a graph.
 */
int acggraph_copy(
    struct acggraph * dst,
    const struct acggraph * src)
{
    /* partitioning information */
    dst->nparts = src->nparts;
    dst->parttag = src->parttag;
    dst->nprocs = src->nprocs;
    dst->npparts = src->npparts;
    dst->ownerrank = src->ownerrank;
    dst->ownerpart = src->ownerpart;

    /* graph/subgraph nodes */
    dst->nnodes = src->nnodes;
    dst->npnodes = src->npnodes;
    if (src->nodetags) {
        dst->nodetags = malloc(dst->npnodes*sizeof(*dst->nodetags));
        if (!dst->nodetags) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dst->npnodes; i++)
            dst->nodetags[i] = src->nodetags[i];
    } else { dst->nodetags = NULL; }
    if (src->parentnodeidx) {
        dst->parentnodeidx = malloc(dst->npnodes*sizeof(*dst->parentnodeidx));
        if (!dst->parentnodeidx) return ACG_ERR_ERRNO;
        for (acgidx_t i = 0; i < dst->npnodes; i++)
            dst->parentnodeidx[i] = src->parentnodeidx[i];
    } else { dst->parentnodeidx = NULL; }

    /* graph/subgraph edges */
    dst->nedges = src->nedges;
    dst->npedges = src->npedges;
    if (src->edgetags) {
        dst->edgetags = malloc(dst->npedges*sizeof(*dst->edgetags));
        if (!dst->edgetags) return ACG_ERR_ERRNO;
        for (int64_t k = 0; k < dst->npedges; k++)
            dst->edgetags[k] = src->edgetags[k];
    } else { dst->edgetags = NULL; }
    if (src->parentedgeidx) {
        dst->parentedgeidx = malloc(dst->npedges*sizeof(*dst->parentedgeidx));
        if (!dst->parentedgeidx) return ACG_ERR_ERRNO;
        for (int64_t k = 0; k < dst->npedges; k++)
            dst->parentedgeidx[k] = src->parentedgeidx[k];
    } else { dst->parentedgeidx = NULL; }

    /* incidence relation - edges to unordered node pairs */
    dst->nodeidxbase = src->nodeidxbase;
    dst->nodenedges = malloc(dst->npnodes*sizeof(*dst->nodenedges));
    if (!dst->nodenedges) {
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0; i < dst->npnodes; i++)
        dst->nodenedges[i] = src->nodenedges[i];
    dst->srcnodeptr = malloc((dst->npnodes+1)*sizeof(*dst->srcnodeptr));
    if (!dst->srcnodeptr) {
        free(dst->nodenedges);
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    if (src->npnodes == 0 && !src->srcnodeptr) dst->srcnodeptr[0] = 0;
    else {
        for (acgidx_t i = 0; i <= dst->npnodes; i++)
            dst->srcnodeptr[i] = src->srcnodeptr[i];
    }
    dst->srcnodeidx = malloc(dst->npedges*sizeof(*dst->srcnodeidx));
    if (!dst->srcnodeidx) {
        free(dst->srcnodeptr); free(dst->nodenedges);
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    for (int64_t k = 0; k < dst->npedges; k++)
        dst->srcnodeidx[k] = src->srcnodeidx[k];
    dst->dstnodeidx = malloc(dst->npedges*sizeof(*dst->dstnodeidx));
    if (!dst->dstnodeidx) {
        free(dst->srcnodeidx); free(dst->srcnodeptr); free(dst->nodenedges);
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    for (int64_t k = 0; k < dst->npedges; k++)
        dst->dstnodeidx[k] = src->dstnodeidx[k];

    /* interior, border and ghost nodes */
    dst->nownednodes = src->nownednodes;
    dst->ninnernodes = src->ninnernodes;
    dst->nbordernodes = src->nbordernodes;
    dst->bordernodeoffset = src->bordernodeoffset;
    dst->nghostnodes = src->nghostnodes;
    dst->ghostnodeoffset = src->ghostnodeoffset;

    /* interior, border and ghost edges */
    dst->ninneredges = src->ninneredges;
    dst->ninterfaceedges = src->ninterfaceedges;
    dst->nbordernodeinneredges = malloc(dst->nbordernodes*sizeof(*dst->nbordernodeinneredges));
    if (!dst->nbordernodeinneredges) {
        free(dst->dstnodeidx); free(dst->srcnodeidx);
        free(dst->srcnodeptr); free(dst->nodenedges);
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t l = 0; l < dst->nbordernodes; l++)
        dst->nbordernodeinneredges[l] = src->nbordernodeinneredges[l];
    dst->nbordernodeinterfaceedges = malloc(dst->nbordernodes*sizeof(*dst->nbordernodeinterfaceedges));
    if (!dst->nbordernodeinterfaceedges) {
        free(dst->nbordernodeinneredges);
        free(dst->dstnodeidx); free(dst->srcnodeidx);
        free(dst->srcnodeptr); free(dst->nodenedges);
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t l = 0; l < dst->nbordernodes; l++)
        dst->nbordernodeinterfaceedges[l] = src->nbordernodeinterfaceedges[l];

    /* neighbouring subgraphs */
    dst->nneighbours = src->nneighbours;
    dst->neighbours = malloc(dst->nneighbours*sizeof(*dst->neighbours));
    if (!dst->neighbours) {
        free(dst->nbordernodeinterfaceedges);
        free(dst->nbordernodeinneredges);
        free(dst->dstnodeidx); free(dst->srcnodeidx);
        free(dst->srcnodeptr); free(dst->nodenedges);
        free(dst->parentedgeidx); free(dst->edgetags);
        free(dst->parentnodeidx); free(dst->nodetags);
        return ACG_ERR_ERRNO;
    }
    for (int p = 0; p < dst->nneighbours; p++) {
        int err = acggraphneighbour_copy(
            &dst->neighbours[p], &src->neighbours[p]);
        if (err) {
            for (int q = p-1; q >= 0; q--) acggraphneighbour_free(&dst->neighbours[q]);
            free(dst->neighbours);
            free(dst->nbordernodeinterfaceedges);
            free(dst->nbordernodeinneredges);
            free(dst->dstnodeidx); free(dst->srcnodeidx);
            free(dst->srcnodeptr); free(dst->nodenedges);
            free(dst->parentedgeidx); free(dst->edgetags);
            free(dst->parentnodeidx); free(dst->nodetags);
            return err;
        }
    }
    return ACG_SUCCESS;
}

/*
 * output (e.g., for debugging)
 */

static int acggraphneighbour_fwrite(
    FILE * f,
    const struct acggraphneighbour * g)
{
    fprintf(f, "neighbourrank: %d\n", g->neighbourrank);
    fprintf(f, "neighbourpart: %d\n", g->neighbourpart);
    fprintf(f, "nbordernodes: %"PRIdx"\n", g->nbordernodes);
    fprintf(f, "bordernodes:");
    if (g->bordernodes) {
        for (acgidx_t i = 0; i < g->nbordernodes; i++)
            fprintf(f, " %"PRIdx, g->bordernodes[i]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");
    fprintf(f, "nghostnodes: %"PRIdx"\n", g->nghostnodes);
    fprintf(f, "ghostnodes:");
    if (g->ghostnodes) {
        for (acgidx_t i = 0; i < g->nghostnodes; i++)
            fprintf(f, " %"PRIdx, g->ghostnodes[i]);
    } else { fprintf(f, " none"); }
    return ACG_SUCCESS;
}

int acggraph_fwrite(
    FILE * f,
    const struct acggraph * g)
{
    /* partitioning information */
    fprintf(f, "nparts: %d\n", g->nparts);
    fprintf(f, "parttag: %d\n", g->parttag);
    fprintf(f, "nprocs: %d\n", g->nprocs);
    fprintf(f, "npparts: %d\n", g->npparts);
    fprintf(f, "ownerrank: %d\n", g->ownerrank);
    fprintf(f, "ownerpart: %d\n", g->ownerpart);

    /* graph/subgraph nodes */
    fprintf(f, "nnodes: %"PRIdx"\n", g->nnodes);
    fprintf(f, "npnodes: %"PRIdx"\n", g->npnodes);
    fprintf(f, "nodetags:");
    if (g->nodetags) {
        for (acgidx_t i = 0; i < g->npnodes; i++)
            fprintf(f, " %"PRIdx, g->nodetags[i]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");
    fprintf(f, "parentnodeidx:");
    if (g->parentnodeidx) {
        for (acgidx_t i = 0; i < g->npnodes; i++)
            fprintf(f, " %"PRIdx, g->parentnodeidx[i]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");

    /* graph/subgraph edges */
    fprintf(f, "nedges: %"PRId64"\n", g->nedges);
    fprintf(f, "npedges: %"PRId64"\n", g->npedges);
    fprintf(f, "edgetags:");
    if (g->edgetags) {
        for (int64_t k = 0; k < g->npedges; k++)
            fprintf(f, " %"PRId64, g->edgetags[k]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");
    fprintf(f, "parentedgeidx:");
    if (g->parentedgeidx) {
        for (int64_t k = 0; k < g->npedges; k++)
            fprintf(f, " %"PRId64, g->parentedgeidx[k]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");

    /* incidence relation - edge to unordered node pairs */
    fprintf(f, "nodeidxbase: %d\n", g->nodeidxbase);
    fprintf(f, "nodenedges:");
    for (acgidx_t i = 0; i < g->npnodes; i++)
        fprintf(f, " %"PRId64, g->nodenedges[i]);
    fprintf(f, "\n");
    fprintf(f, "srcnodeptr:");
    for (acgidx_t i = 0; i <= g->npnodes; i++)
        fprintf(f, " %"PRId64, g->srcnodeptr[i]);
    fprintf(f, "\n");
    fprintf(f, "srcnodeidx:");
    for (int64_t k = 0; k < g->npedges; k++)
        fprintf(f, " %"PRIdx, g->srcnodeidx[k]);
    fprintf(f, "\n");
    fprintf(f, "dstnodeidx:");
    for (int64_t k = 0; k < g->npedges; k++)
        fprintf(f, " %"PRIdx, g->dstnodeidx[k]);
    fprintf(f, "\n");

    /* interior, border and ghost nodes */
    fprintf(f, "nownednodes: %"PRIdx"\n", g->nownednodes);
    fprintf(f, "ninnernodes: %"PRIdx"\n", g->ninnernodes);
    fprintf(f, "nbordernodes: %"PRIdx"\n", g->nbordernodes);
    fprintf(f, "bordernodeoffset: %"PRIdx"\n", g->bordernodeoffset);
    fprintf(f, "nghostnodes: %"PRIdx"\n", g->nghostnodes);
    fprintf(f, "ghostnodeoffset: %"PRIdx"\n", g->ghostnodeoffset);

    /* interior, border and ghost edges */
    fprintf(f, "ninneredges: %"PRId64"\n", g->ninneredges);
    fprintf(f, "ninterfaceedges: %"PRId64"\n", g->ninterfaceedges);
    fprintf(f, "nbordernodeinneredges:");
    if (g->nbordernodeinneredges) {
        for (int i = 0; i < g->nbordernodes; i++)
            fprintf(f, " %"PRId64, g->nbordernodeinneredges[i]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");
    fprintf(f, "nbordernodeinterfaceedges:");
    if (g->nbordernodeinterfaceedges) {
        for (int i = 0; i < g->nbordernodes; i++)
            fprintf(f, " %"PRId64, g->nbordernodeinterfaceedges[i]);
    } else { fprintf(f, " none"); }
    fprintf(f, "\n");

    /* neighbouring subgraphs */
    fprintf(f, "nneighbours: %d\n", g->nneighbours);
    for (int p = 0; p < g->nneighbours; p++) {
        acggraphneighbour_fwrite(f, &g->neighbours[p]);
        fprintf(f, "\n");
    }
    return ACG_SUCCESS;
}

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
    int verbose)
{
    int err;
    err = metis_partgraphsym(
        partitioner,
        nparts, graph->nnodes, graph->nedges,
        sizeof(*graph->srcnodeidx), graph->nodeidxbase, graph->srcnodeidx,
        sizeof(*graph->dstnodeidx), graph->nodeidxbase, graph->dstnodeidx,
        nodeparts, objval, seed, verbose);
    if (err) return err;
    return ACG_SUCCESS;
}

/**
 * ‘partitions’ is a data structure used to represent a set of
 * subgraphs together with a set of interfaces between those
 * subgraphs, which are made up of edges between nodes in different
 * subgraphs, according to a given node partitioning.
 */
struct partitions
{
    int ninterfaces;
    struct interface
    {
        int p, q;
        acgidx_t npnodes;
        acgidx_t * pnodes;
        acgidx_t nqnodes;
        acgidx_t * qnodes;
        int64_t nedges;
        int64_t * edges;
    } * interfaces;

    int ninteriors;
    struct interior
    {
        int ninterfaces;
        int * interfaces;
    } * interiors;
};

static void partitions_free(
    struct partitions * partitions)
{
    for (int i = 0; i < partitions->ninterfaces; i++) {
        struct interface * interface = &partitions->interfaces[i];
        free(interface->pnodes);
        free(interface->qnodes);
        free(interface->edges);
    }
    free(partitions->interfaces);
    for (int i = 0; i < partitions->ninteriors; i++) {
        struct interior * interior = &partitions->interiors[i];
        free(interior->interfaces);
    }
    free(partitions->interiors);
}

/**
 * ‘partitions_init()’ finds the interfaces between neighbouring
 * subgraphs that are induced by a given node partitioning.
 *
 * Each interface is made up of edges between nodes in different
 * subgraphs according to the given node partitioning.
 */
static int partitions_init(
    struct partitions * partitions,
    const struct acggraph * srcgraph,
    int nparts,
    const int * nodeparts)
{
    int err;

    /*
     * 1. Count the number of interface edges, i.e., edges with nodes
     *    in different partitions.
     */

    int64_t ninterfaceedges = 0;
    int nodeidxbase = srcgraph->nodeidxbase;
    #pragma omp parallel for reduction(+:ninterfaceedges)
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) {
        for (int64_t k = srcgraph->srcnodeptr[i];
             k < srcgraph->srcnodeptr[i+1];
             k++)
        {
            acgidx_t j = srcgraph->dstnodeidx[k]-nodeidxbase;
            if (nodeparts[i] != nodeparts[j]) ninterfaceedges++;
        }
    }
    int64_t * interfaceedges = malloc(ninterfaceedges*sizeof(*interfaceedges));
    if (!interfaceedges) return ACG_ERR_ERRNO;
    int64_t l = 0;
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) {
        for (int64_t k = srcgraph->srcnodeptr[i];
             k < srcgraph->srcnodeptr[i+1];
             k++)
        {
            acgidx_t j = srcgraph->dstnodeidx[k]-nodeidxbase;
            if (nodeparts[i] != nodeparts[j]) interfaceedges[l++] = k;
        }
    }

    /*
     * 2. Sort interface edges as pairs of node partition numbers.
     */

    int * srcnodepart = malloc(ninterfaceedges*sizeof(*srcnodepart));
    if (!srcnodepart) { free(interfaceedges); return ACG_ERR_ERRNO; }
    int * dstnodepart = malloc(ninterfaceedges*sizeof(*dstnodepart));
    if (!dstnodepart) { free(srcnodepart); free(interfaceedges); return ACG_ERR_ERRNO; }
    #pragma omp parallel for
    for (int64_t l = 0; l < ninterfaceedges; l++) {
        int64_t k = interfaceedges[l];
        acgidx_t i = srcgraph->srcnodeidx[k]-srcgraph->nodeidxbase;
        acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
        srcnodepart[l] = nodeparts[i] < nodeparts[j] ? nodeparts[i] : nodeparts[j];
        dstnodepart[l] = nodeparts[i] < nodeparts[j] ? nodeparts[j] : nodeparts[i];
    }
    int64_t * edgeinvperm = malloc(ninterfaceedges*sizeof(*edgeinvperm));
    if (!edgeinvperm) {
        free(dstnodepart); free(srcnodepart); free(interfaceedges);
        return ACG_ERR_ERRNO;
    }
    err = acgradixsortpair_int(
        ninterfaceedges, sizeof(*srcnodepart), srcnodepart,
        sizeof(*dstnodepart), dstnodepart, NULL, edgeinvperm);
    if (err) {
        free(edgeinvperm); free(dstnodepart); free(srcnodepart); free(interfaceedges);
        return err;
    }

    /*
     * 3. Count the number of interfaces and map each subgraph to the
     *    interfaces between itself and its neighbouring subgraphs.
     */

    struct interior * interiors = malloc(nparts*sizeof(*interiors));
    if (!interiors) {
        free(edgeinvperm); free(dstnodepart); free(srcnodepart); free(interfaceedges);
        return ACG_ERR_ERRNO;
    }
    for (int p = 0; p < nparts; p++) {
        struct interior * interior = &interiors[p];
        interior->ninterfaces = 0;
        interior->interfaces = NULL;
    }
    int ninterfaces = 0;
    for (int64_t k = 0; k < ninterfaceedges;) {
        int p = srcnodepart[k], q = dstnodepart[k];
        struct interior * g = &interiors[p];
        struct interior * h = &interiors[q];
        g->ninterfaces++; h->ninterfaces++;
        ninterfaces++;
        do { k++; } while (k < ninterfaceedges &&
                           srcnodepart[k] == srcnodepart[k-1] &&
                           dstnodepart[k] == dstnodepart[k-1]);
    }

    /* allocate storage for interfaces and for mapping subgraphs to
     * those interfaces */
    for (int p = 0; p < nparts; p++) {
        struct interior * interior = &interiors[p];
        interior->interfaces = malloc(interior->ninterfaces*sizeof(*interior->interfaces));
        if (!interior->interfaces) return ACG_ERR_ERRNO;
    }

    /* map each subgraph to adjacent interfaces */
    for (int p = 0; p < nparts; p++) interiors[p].ninterfaces = 0;
    for (int64_t k = 0, n = 0; k < ninterfaceedges;) {
        int p = srcnodepart[k], q = dstnodepart[k];
        struct interior * g = &interiors[p];
        struct interior * h = &interiors[q];
        g->interfaces[g->ninterfaces++] = n;
        h->interfaces[h->ninterfaces++] = n;
        n++;
        do { k++; } while (k < ninterfaceedges &&
                           srcnodepart[k] == srcnodepart[k-1] &&
                           dstnodepart[k] == dstnodepart[k-1]);
    }

    /*
     * 4. Count and extract edges for each interface.
     */

    /* allocate storage for each interface */
    struct interface * interfaces = malloc(ninterfaces*sizeof(*interfaces));
    if (!interfaces) {
        for (int p = 0; p < nparts; p++) { free(interiors[p].interfaces); } free(interiors);
        free(edgeinvperm); free(dstnodepart); free(srcnodepart); free(interfaceedges);
        return ACG_ERR_ERRNO;
    }
    for (int n = 0; n < ninterfaces; n++) {
        struct interface * interface = &interfaces[n];
        interface->p = interface->q = 0;
        interface->npnodes = interface->nqnodes = interface->nedges = 0;
        interface->pnodes = interface->qnodes = NULL;
        interface->edges = NULL;
    }

    /* count the number of edges for each interface */
    for (int64_t k = 0, n = 0; k < ninterfaceedges;) {
        int p = srcnodepart[k], q = dstnodepart[k];
        struct interface * interface = &interfaces[n++];
        interface->p = srcnodepart[k];
        interface->q = dstnodepart[k];
        do { k++; interface->nedges++; }
        while (k < ninterfaceedges &&
               srcnodepart[k] == srcnodepart[k-1] &&
               dstnodepart[k] == dstnodepart[k-1]);
    }

    /* allocate storage for the edges of each interface */
    for (int n = 0; n < ninterfaces; n++) {
        struct interface * interface = &interfaces[n];
        interface->edges = malloc(interface->nedges*sizeof(*interface->edges));
        if (!interface->edges) return ACG_ERR_ERRNO;
    }

    /* extract edges for each interface */
    for (int64_t k = 0, n = 0; k < ninterfaceedges;) {
        int p = srcnodepart[k], q = dstnodepart[k];
        struct interface * interface = &interfaces[n++];
        int64_t l = 0;
        do {
            interface->edges[l++] = interfaceedges[edgeinvperm[k]];
            k++;
        } while (k < ninterfaceedges &&
                 srcnodepart[k] == srcnodepart[k-1] &&
                 dstnodepart[k] == dstnodepart[k-1]);
    }
    free(edgeinvperm); free(dstnodepart); free(srcnodepart); free(interfaceedges);

    /*
     * 5. Extract nodes for each interface by sorting and compacting
     *    the nodes of every interface edge. The nodes are arranged in
     *    two groups based on the partition they belong to.
     */

    for (int n = 0; n < ninterfaces; n++) {
        struct interface * interface = &interfaces[n];

        /* obtain lists of nodes, possibly containing duplicates */
        acgidx_t * pnodes = malloc(interface->nedges*sizeof(*pnodes));
        acgidx_t * qnodes = malloc(interface->nedges*sizeof(*qnodes));
        #pragma omp parallel for
        for (int64_t l = 0; l < interface->nedges; l++) {
            int64_t k = interface->edges[l];
            acgidx_t i = srcgraph->srcnodeidx[k]-srcgraph->nodeidxbase;
            acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
            if (nodeparts[i] == interface->p) {
                pnodes[l] = i, qnodes[l] = j;
            } else { pnodes[l] = j, qnodes[l] = i; }
        }

        /* sort and compact nodes to remove duplicates */
        err = acgradixsort_idx_t(
            interface->nedges, sizeof(*pnodes), pnodes, NULL, NULL);
        if (err) return err;
        for (int64_t l = 0; l < interface->nedges;) {
            interface->npnodes++;
            do { l++; } while (l < interface->nedges && pnodes[l] == pnodes[l-1]);
        }
        interface->pnodes = malloc(interface->npnodes*sizeof(*interface->pnodes));
        for (int64_t l = 0, i = 0; l < interface->nedges;) {
            interface->pnodes[i++] = pnodes[l];
            do { l++; } while (l < interface->nedges && pnodes[l] == pnodes[l-1]);
        }
        free(pnodes);
        err = acgradixsort_idx_t(
            interface->nedges, sizeof(*qnodes), qnodes, NULL, NULL);
        if (err) return err;
        for (int64_t l = 0; l < interface->nedges;) {
            interface->nqnodes++;
            do { l++; } while (l < interface->nedges && qnodes[l] == qnodes[l-1]);
        }
        interface->qnodes = malloc(interface->nqnodes*sizeof(*interface->qnodes));
        for (int64_t l = 0, i = 0; l < interface->nedges;) {
            interface->qnodes[i++] = qnodes[l];
            do { l++; } while (l < interface->nedges && qnodes[l] == qnodes[l-1]);
        }
        free(qnodes);
    }

    /* set up partitions */
    partitions->ninteriors = nparts;
    partitions->interiors = interiors;
    partitions->ninterfaces = ninterfaces;
    partitions->interfaces = interfaces;
    return ACG_SUCCESS;
}

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
    int verbose)
{
    int err;
    acgtime_t t0, t1;

    /* check the node partitioning */
    acgidx_t nnodes = srcgraph->nnodes;
    for (acgidx_t i = 0; i < nnodes; i++) {
        if (nodeparts[i] < 0 || nodeparts[i] >= nparts)
            return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    }

    /* initialise empty subgraphs for each partition */
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];

        /* partitioning information */
        g->nparts = nparts;
        g->parttag = parttags ? parttags[p] : p+1;
        g->nprocs = 1;
        g->npparts = nparts;
        g->ownerrank = 0;
        g->ownerpart = p;

        /* graph/subgraph nodes */
        g->nnodes = srcgraph->nnodes;
        g->npnodes = 0;
        g->nodetags = NULL;
        g->parentnodeidx = NULL;

        /* graph/subgraph edges */
        g->nedges = srcgraph->nedges;
        g->npedges = 0;
        g->edgetags = NULL;
        g->parentedgeidx = NULL;

        /* incidence relation - edges to unordered node pairs */
        g->nodeidxbase = srcgraph->nodeidxbase;
        g->nodenedges = NULL;
        g->srcnodeptr = NULL;
        g->srcnodeidx = NULL;
        g->dstnodeidx = NULL;

        /* interior, border and ghost nodes */
        g->nownednodes = 0;
        g->ninnernodes = 0;
        g->nbordernodes = 0;
        g->bordernodeoffset = 0;
        g->nghostnodes = 0;
        g->ghostnodeoffset = 0;

        /* interior, border and ghost edges */
        g->ninneredges = 0;
        g->ninterfaceedges = 0;
        g->nbordernodeinneredges = NULL;
        g->nbordernodeinterfaceedges = NULL;

        /* neighbouring subgraphs */
        g->nneighbours = 0;
        g->neighbours = NULL;
    }

    /*
     * 1. Find the interfaces between neighbouring subgraphs.
     */

    if (verbose > 0) {
        fprintf(stderr, "finding interfaces between neighbouring subgraphs:");
        gettime(&t0);
    }

    struct partitions partitions;
    err = partitions_init(
        &partitions, srcgraph, nparts, nodeparts);
    if (err) return err;

    /* allocate storage for neighbours of each subgraph */
    for (int p = 0; p < nparts; p++) {
        const struct interior * interior = &partitions.interiors[p];
        struct acggraph * g = &subgraphs[p];
        g->nneighbours = interior->ninterfaces;
        g->neighbours = malloc(g->nneighbours*sizeof(*g->neighbours));
        for (int r = 0; r < g->nneighbours; r++) {
            int interfaceidx = interior->interfaces[r];
            const struct interface * interface = &partitions.interfaces[interfaceidx];
            struct acggraphneighbour * neighbour = &g->neighbours[r];
            neighbour->neighbourrank = 0;
            if (p == interface->p) {
                neighbour->neighbourpart = interface->q;
                neighbour->nbordernodes = interface->npnodes;
                neighbour->nghostnodes = interface->nqnodes;
            } else if (p == interface->q) {
                neighbour->neighbourpart = interface->p;
                neighbour->nbordernodes = interface->nqnodes;
                neighbour->nghostnodes = interface->npnodes;
            } else {
                fprintf(stderr, "%s:%d:"
                        " subgraph %'d of %'d"
                        " has incorrect interface %'d of %'d"
                        " for subgraphs %'d and %'d\n",
                        __FILE__, __LINE__,
                        p, nparts, interfaceidx,
                        interior->ninterfaces,
                        interface->p, interface->q);
                return ACG_ERR_INVALID_VALUE;
            }
            neighbour->bordernodes = malloc(neighbour->nbordernodes*sizeof(*neighbour->bordernodes));
            if (!neighbour->bordernodes) return ACG_ERR_ERRNO;
            neighbour->ghostnodes = malloc(neighbour->nghostnodes*sizeof(*neighbour->ghostnodes));
            if (!neighbour->ghostnodes) return ACG_ERR_ERRNO;
        }
    }

    if (verbose > 0) {
        gettime(&t1);
        fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
    }

    /*
     * 2. Count and extract ghost nodes for every part.
     */

    if (verbose > 0) {
        fprintf(stderr, "counting and extracting ghost nodes:");
        gettime(&t0);
    }

    /* count the number of nodes exclusively owned by each subgraph */
    #pragma omp parallel for
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) {
        int p = nodeparts[i];
        struct acggraph * g = &subgraphs[p];
        #pragma omp atomic
        g->nownednodes++;
    }

    /* count and extract ghost nodes for each subgraph */
    for (int p = 0; p < nparts; p++) {
        const struct interior * interior = &partitions.interiors[p];
        struct acggraph * g = &subgraphs[p];
        g->ghostnodeoffset = g->nownednodes;
        g->nghostnodes = 0;
        for (int r = 0; r < g->nneighbours; r++) {
            struct acggraphneighbour * neighbour = &g->neighbours[r];
            g->nghostnodes += neighbour->nghostnodes;
        }
        g->npnodes = g->nownednodes + g->nghostnodes;
        g->parentnodeidx = malloc(g->npnodes*sizeof(*g->parentnodeidx));
        if (!g->parentnodeidx) return ACG_ERR_ERRNO;
    }

    #pragma omp parallel for
    for (int p = 0; p < nparts; p++) {
        const struct interior * interior = &partitions.interiors[p];
        struct acggraph * g = &subgraphs[p];
        acgidx_t l = 0;
        for (int r = 0; r < g->nneighbours; r++) {
            struct acggraphneighbour * neighbour = &g->neighbours[r];
            int interfaceidx = interior->interfaces[r];
            const struct interface * interface = &partitions.interfaces[interfaceidx];
            const acgidx_t * ghostnodes = NULL;
            if (p == interface->p) ghostnodes = interface->qnodes;
            else ghostnodes = interface->pnodes;
            for (acgidx_t i = 0; i < neighbour->nghostnodes; i++, l++) {
                neighbour->ghostnodes[i] = l;
                g->parentnodeidx[g->ghostnodeoffset+l] = ghostnodes[i];
            }
        }
    }

    if (verbose > 0) {
        gettime(&t1);
        fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
    }

    /*
     * 3. Extract interior and border nodes for every subgraph.
     */

    if (verbose > 0) {
        fprintf(stderr, "extracting interior and border nodes for subgraphs:");
        gettime(&t0);
    }

    /* for each node, count its occurences as a ghost node */
    int * nodenghosts = malloc(srcgraph->nnodes*sizeof(*nodenghosts));
    if (!nodenghosts) return ACG_ERR_ERRNO;
    #pragma omp parallel for
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) nodenghosts[i] = 0;
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];
        for (int r = 0; r < g->nneighbours; r++) {
            struct acggraphneighbour * neighbour = &g->neighbours[r];
            for (acgidx_t l = 0; l < neighbour->nghostnodes; l++) {
                acgidx_t ghostnodeidx = neighbour->ghostnodes[l];
                acgidx_t i = g->parentnodeidx[g->ghostnodeoffset+ghostnodeidx];
                nodenghosts[i]++;
            }
        }
    }

    /* count interior and border nodes for every subgraph */
    #pragma omp parallel for
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) {
        int p = nodeparts[i];
        struct acggraph * g = &subgraphs[p];
        if (nodenghosts[i] == 0) {
            #pragma omp atomic
            g->ninnernodes++;
        } else {
            #pragma omp atomic
            g->nbordernodes++;
        }
    }
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];
        g->bordernodeoffset = g->ninnernodes;
        g->ninnernodes = g->nbordernodes = 0;
    }

    /* extract interior and border nodes for every subgraph */
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) {
        int p = nodeparts[i];
        struct acggraph * g = &subgraphs[p];
        if (nodenghosts[i] == 0) g->parentnodeidx[g->ninnernodes++] = i;
        else g->parentnodeidx[g->bordernodeoffset+g->nbordernodes++] = i;
    }

    /* copy node tags */
    if (srcgraph->nodetags) {
        for (int p = 0; p < nparts; p++) {
            struct acggraph * g = &subgraphs[p];
            g->nodetags = malloc(g->npnodes*sizeof(*g->nodetags));
            if (!g->nodetags) return ACG_ERR_ERRNO;
            #pragma omp parallel for
            for (acgidx_t l = 0; l < g->npnodes; l++) {
                acgidx_t i = g->parentnodeidx[l];
                g->nodetags[l] = srcgraph->nodetags[i];
            }
        }
    }

    /* map border nodes of interfaces to subgraph border nodes */
    #pragma omp parallel for
    for (int p = 0; p < nparts; p++) {
        const struct interior * interior = &partitions.interiors[p];
        struct acggraph * g = &subgraphs[p];
        for (int r = 0; r < g->nneighbours; r++) {
            struct acggraphneighbour * neighbour = &g->neighbours[r];
            int interfaceidx = interior->interfaces[r];
            const struct interface * interface = &partitions.interfaces[interfaceidx];
            const acgidx_t * neighbourbordernodes;
            if (p == interface->p) neighbourbordernodes = interface->pnodes;
            else neighbourbordernodes = interface->qnodes;
            acgidx_t k = 0, l = 0;
            while (k < g->nbordernodes && l < neighbour->nbordernodes) {
                acgidx_t i = g->parentnodeidx[g->bordernodeoffset+k];
                acgidx_t j = neighbourbordernodes[l];
                if (i < j) k++;
                else neighbour->bordernodes[l++] = k;
            }
        }
    }

    if (verbose > 0) {
        gettime(&t1);
        fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
    }

    /*
     * 4. For every subgraph, extract edges for each of its nodes.
     */

    if (verbose > 0) {
        fprintf(stderr, "extracting subgraph edges:");
        gettime(&t0);
    }

    /* count the total number of edges in every subgraph, which
     * amounts to the edges of all nodes owned by the subgraph and all
     * edges of any interfaces to neighbouring subgraphs */
    #pragma omp parallel for
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];
        g->npedges = 0;

        /* count every edge for interior nodes */
        for (acgidx_t l = 0; l < g->ninnernodes; l++) {
            acgidx_t i = g->parentnodeidx[l];
            g->npedges += srcgraph->nodenedges[i];
        }

        /* count only interior edges for border nodes, otherwise we
         * end up double counting edges on the interface */
        for (acgidx_t l = 0; l < g->nbordernodes; l++) {
            acgidx_t i = g->parentnodeidx[g->bordernodeoffset+l];
            for (int64_t k = srcgraph->srcnodeptr[i];
                 k < srcgraph->srcnodeptr[i+1];
                 k++)
            {
                acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodeparts[j] == p) g->npedges++;
            }
        }

        /* count edges from border nodes to ghost nodes */
        const struct interior * interior = &partitions.interiors[p];
        for (int r = 0; r < g->nneighbours; r++) {
            int interfaceidx = interior->interfaces[r];
            const struct interface * interface = &partitions.interfaces[interfaceidx];
            g->npedges += interface->nedges;
        }
    }
    partitions_free(&partitions);

    /* allocate storage for subgraph edges */
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];
        g->parentedgeidx = malloc(g->npedges*sizeof(*g->parentedgeidx));
        if (!g->parentedgeidx) return ACG_ERR_ERRNO;
        g->nbordernodeinneredges = malloc(g->nbordernodes*sizeof(*g->nbordernodeinneredges));
        if (!g->nbordernodeinneredges) return ACG_ERR_ERRNO;
        g->nbordernodeinterfaceedges = malloc(g->nbordernodes*sizeof(*g->nbordernodeinterfaceedges));
        if (!g->nbordernodeinterfaceedges) return ACG_ERR_ERRNO;
    }

    /* for each node, count and extract incident edges that need to be
     * reversed after renumbering subgraph nodes, including edges from
     * border nodes to interior nodes and edges from ghost nodes to
     * border nodes */
    int * nodenreversededges = malloc(srcgraph->nnodes*sizeof(*nodenreversededges));
    #pragma omp parallel for
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) nodenreversededges[i] = 0;
    #pragma omp parallel for
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];
        for (acgidx_t m = 0; m < g->nbordernodes; m++) {
            acgidx_t j = g->parentnodeidx[g->bordernodeoffset+m];
            for (int64_t k = srcgraph->srcnodeptr[j];
                 k < srcgraph->srcnodeptr[j+1];
                 k++)
            {
                acgidx_t i = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodeparts[i] == p && nodenghosts[i] == 0) {
                    #pragma omp atomic
                    nodenreversededges[i]++;
                }
            }
        }
        for (acgidx_t m = 0; m < g->nghostnodes; m++) {
            acgidx_t j = g->parentnodeidx[g->ghostnodeoffset+m];
            for (int64_t k = srcgraph->srcnodeptr[j];
                 k < srcgraph->srcnodeptr[j+1];
                 k++)
            {
                acgidx_t i = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodeparts[i] == p) {
                    #pragma omp atomic
                    nodenreversededges[i]++;
                }
            }
        }
    }
    int64_t * nodereversededgeptr = malloc((srcgraph->nnodes+1)*sizeof(*nodereversededgeptr));
    if (!nodereversededgeptr) return ACG_ERR_ERRNO;
    /* nodereversededgeptr[0] = 0; */
    /* for (acgidx_t l = 1; l <= srcgraph->nnodes; l++) */
    /*     nodereversededgeptr[l] = nodereversededgeptr[l-1]+nodenreversededges[l-1]; */
    #pragma omp parallel for
    for (acgidx_t l = 0; l < srcgraph->nnodes; l++) nodereversededgeptr[l] = nodenreversededges[l];
    acgprefixsum_inplace_int64_t(srcgraph->nnodes+1, nodereversededgeptr, false);

    int64_t nnodereversededges = nodereversededgeptr[srcgraph->nnodes];
    int64_t * nodereversededges = malloc(nnodereversededges*sizeof(*nodereversededges));
    if (!nodereversededges) return ACG_ERR_ERRNO;
    #pragma omp parallel for
    for (acgidx_t i = 0; i < srcgraph->nnodes; i++) nodenreversededges[i] = 0;
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];
        for (acgidx_t m = 0; m < g->nbordernodes; m++) {
            acgidx_t j = g->parentnodeidx[g->bordernodeoffset+m];
            for (int64_t k = srcgraph->srcnodeptr[j];
                 k < srcgraph->srcnodeptr[j+1];
                 k++)
            {
                acgidx_t i = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodeparts[i] == p && nodenghosts[i] == 0)
                    nodereversededges[nodereversededgeptr[i]+nodenreversededges[i]++] = k;
            }
        }
        for (acgidx_t m = 0; m < g->nghostnodes; m++) {
            acgidx_t j = g->parentnodeidx[g->ghostnodeoffset+m];
            for (int64_t k = srcgraph->srcnodeptr[j];
                 k < srcgraph->srcnodeptr[j+1];
                 k++)
            {
                acgidx_t i = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodeparts[i] == p)
                    nodereversededges[nodereversededgeptr[i]+nodenreversededges[i]++] = k;
            }
        }
    }

    /* for every subgraph, extract edges in the following order:
     *
     *   1. for every interior node:
     *      1a) edges between interior nodes
     *      1b) edges from interior to border nodes
     *      1c) edges from border to interior nodes
     *
     *   2. for every border node:
     *      2a) edges between border nodes
     *      2b) edges from border to ghost nodes
     *      2c) edges ghost to border nodes
     */
    #pragma omp parallel for
    for (int p = 0; p < nparts; p++) {
        struct acggraph * g = &subgraphs[p];

        /* for every interior node, extract edges to interior nodes
         * then border nodes */
        int64_t dstedgeidx = 0;
        for (acgidx_t l = 0; l < g->ninnernodes; l++) {
            acgidx_t i = g->parentnodeidx[l];
            for (int64_t k = srcgraph->srcnodeptr[i];
                 k < srcgraph->srcnodeptr[i+1];
                 k++)
            {
                acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodenghosts[j] == 0) g->parentedgeidx[dstedgeidx++] = k+1;
            }
            for (int64_t k = srcgraph->srcnodeptr[i];
                 k < srcgraph->srcnodeptr[i+1];
                 k++)
            {
                acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodenghosts[j] > 0) g->parentedgeidx[dstedgeidx++] = k+1;
            }

            /* Look for edges from border nodes to interior nodes,
             * store the edge index with a negative sign to indicate
             * that the orientation is reversed. Always add 1, making
             * the index nonzero, to be able to distinguish the
             * orientation even for an edge with index zero. */
            for (int64_t l = nodereversededgeptr[i];
                 l < nodereversededgeptr[i+1];
                 l++)
            {
                int64_t k = nodereversededges[l];
                g->parentedgeidx[dstedgeidx++] = -k-1;
            }
        }

        /* for every border node, extract edges to border nodes then
         * ghost nodes */
        for (acgidx_t l = 0; l < g->nbordernodes; l++) {
            g->nbordernodeinneredges[l] = g->nbordernodeinterfaceedges[l] = 0;
            acgidx_t i = g->parentnodeidx[g->bordernodeoffset+l];
            for (int64_t k = srcgraph->srcnodeptr[i];
                 k < srcgraph->srcnodeptr[i+1];
                 k++)
            {
                acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodenghosts[j] > 0 && nodeparts[j] == p) {
                    g->nbordernodeinneredges[l]++;
                    g->parentedgeidx[dstedgeidx++] = k+1;
                }
            }
            for (int64_t k = srcgraph->srcnodeptr[i];
                 k < srcgraph->srcnodeptr[i+1];
                 k++)
            {
                acgidx_t j = srcgraph->dstnodeidx[k]-srcgraph->nodeidxbase;
                if (nodeparts[j] != p) {
                    g->nbordernodeinterfaceedges[l]++;
                    g->parentedgeidx[dstedgeidx++] = k+1;
                }
            }

            /* Look for edges from ghost nodes to border nodes, store
             * the edge index with a negative sign to indicate that
             * the orientation is reversed. Always add 1, making the
             * index nonzero, to be able to distinguish the
             * orientation even for an edge with index zero. */
            for (int64_t l = nodereversededgeptr[i];
                 l < nodereversededgeptr[i+1];
                 l++)
            {
                int64_t k = nodereversededges[l];
                g->parentedgeidx[dstedgeidx++] = -k-1;
            }
        }
    }
    free(nodereversededges);
    free(nodereversededgeptr);
    free(nodenreversededges);
    free(nodenghosts);

    /* copy edge tags */
    if (srcgraph->edgetags) {
        for (int p = 0; p < nparts; p++) {
            struct acggraph * g = &subgraphs[p];
            g->edgetags = malloc(g->npedges*sizeof(*g->edgetags));
            if (!g->edgetags) return ACG_ERR_ERRNO;
            #pragma omp parallel for
            for (int64_t l = 0; l < g->npedges; l++) {
                int64_t k = g->parentedgeidx[l] > 0 ? (g->parentedgeidx[l]-1) : (-g->parentedgeidx[l]-1);
                g->edgetags[l] = srcgraph->edgetags[k];
            }
        }
    }

    if (verbose > 0) {
        gettime(&t1);
        fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
    }

    /*
     * 5. Set up the incidence relation mapping edges to (unordered)
     *    pairs of nodes for every subgraph. This involves renumbering
     *    nodes to use the new ordering within each part.
     */

    if (verbose > 0) {
        fprintf(stderr, "setting up edge-node incidence relations for each subgraph:");
        gettime(&t0);
    }

    #pragma omp parallel for
    for (int p = 0; p < nparts; p++) {
        int terr;
        if (err) continue;

        struct acggraph * g = &subgraphs[p];
        g->srcnodeidx = malloc(g->npedges*sizeof(*g->srcnodeidx));
        if (!g->srcnodeidx) { err = ACG_ERR_ERRNO; continue; }
        g->dstnodeidx = malloc(g->npedges*sizeof(*g->dstnodeidx));
        if (!g->dstnodeidx) { err = ACG_ERR_ERRNO; continue; }

        /* sort all nodes in the subgraph */
        acgidx_t * sortednodes = malloc(g->npnodes*sizeof(*sortednodes));
        if (!sortednodes) { err = ACG_ERR_ERRNO; continue; }
        for (acgidx_t m = 0; m < g->npnodes; m++)
            sortednodes[m] = g->parentnodeidx[m];
        int64_t * nodeinvperm = malloc(g->npnodes*sizeof(*nodeinvperm));
        if (!nodeinvperm) { err = ACG_ERR_ERRNO; continue; }
        terr = acgradixsort_idx_t(
            g->npnodes, sizeof(*sortednodes), sortednodes, NULL, nodeinvperm);
        if (terr) { err = terr; continue; }

        /* sort subgraph edges by the source node (i.e., first node in the pair) */
        acgidx_t * sortededgenodes = malloc(g->npedges*sizeof(*sortededgenodes));
        if (!sortededgenodes) { err = ACG_ERR_ERRNO; continue; }
        for (int64_t l = 0; l < g->npedges; l++) {
            int64_t k = g->parentedgeidx[l];
            sortededgenodes[l] = k > 0
                ? (srcgraph->srcnodeidx[k-1]-srcgraph->nodeidxbase)
                : (srcgraph->dstnodeidx[-k-1]-srcgraph->nodeidxbase);
        }
        int64_t * edgenodeinvperm = malloc(g->npedges*sizeof(*edgenodeinvperm));
        if (!edgenodeinvperm) { err = ACG_ERR_ERRNO; continue; }
        terr = acgradixsort_idx_t(
            g->npedges, sizeof(*sortededgenodes), sortededgenodes, NULL, edgenodeinvperm);
        if (terr) { err = terr; continue; }

        /* merge the lists of sorted nodes and source nodes of every edge */
        for (int64_t k = 0, l = 0; k < g->npnodes && l < g->npedges;) {
            if (sortednodes[k] < sortededgenodes[l]) k++;
            else if (sortednodes[k] > sortededgenodes[l]) {
                fprintf(stderr, "%s: Warning -"
                        " edge %'"PRId64" of %'"PRId64
                        " references node %'"PRIdx", "
                        " which is missing from the subgraph\n",
                        __func__, edgenodeinvperm[l], g->npedges,
                        sortededgenodes[l]);
                err = ACG_ERR_INVALID_VALUE;
            } else {
                g->srcnodeidx[edgenodeinvperm[l]] = nodeinvperm[k]+g->nodeidxbase; l++;
            }
        }

        /* sort subgraph edges by the sink node (i.e., second node in the pair) */
        for (int64_t l = 0; l < g->npedges; l++) {
            int64_t k = g->parentedgeidx[l];
            sortededgenodes[l] = k > 0
                ? (srcgraph->dstnodeidx[k-1]-srcgraph->nodeidxbase)
                : (srcgraph->srcnodeidx[-k-1]-srcgraph->nodeidxbase);
        }
        terr = acgradixsort_idx_t(
            g->npedges, sizeof(*sortededgenodes), sortededgenodes, NULL, edgenodeinvperm);
        if (terr) { err = terr; continue; }

        /* merge the lists of sorted nodes and sink nodes of every edge */
        for (int64_t k = 0, l = 0; k < g->npnodes && l < g->npedges;) {
            if (sortednodes[k] < sortededgenodes[l]) k++;
            else if (sortednodes[k] > sortededgenodes[l]) {
                fprintf(stderr, "%s: Warning -"
                        " edge %'"PRId64" of %'"PRId64
                        " references node %'"PRIdx", "
                        " which is missing from the subgraph\n",
                        __func__, edgenodeinvperm[l], g->npedges,
                        sortededgenodes[l]);
                err = ACG_ERR_INVALID_VALUE;
            } else { g->dstnodeidx[edgenodeinvperm[l]] = nodeinvperm[k]+g->nodeidxbase; l++; }
        }
        free(edgenodeinvperm); free(sortededgenodes);
        free(nodeinvperm); free(sortednodes);

        g->nodenedges = malloc(g->npnodes*sizeof(*g->nodenedges));
        if (!g->nodenedges) { err = ACG_ERR_ERRNO; continue; }
        for (acgidx_t i = 0; i < g->npnodes; i++)
            g->nodenedges[i] = 0;
        for (int64_t k = 0; k < g->npedges; k++) {
            acgidx_t i = g->srcnodeidx[k]-g->nodeidxbase;
            g->nodenedges[i]++;
        }
        g->srcnodeptr = malloc((g->npnodes+1)*sizeof(*g->srcnodeptr));
        if (!g->srcnodeptr) { err = ACG_ERR_ERRNO; continue; }
        for (acgidx_t l = 0; l < g->npnodes; l++) g->srcnodeptr[l] = g->nodenedges[l];
        g->srcnodeptr[0] = 0;
        for (acgidx_t l = 1; l <= g->npnodes; l++)
            g->srcnodeptr[l] = g->srcnodeptr[l-1] + g->nodenedges[l-1];
    }
    if (err) return err;

    if (verbose > 0) {
        gettime(&t1);
        fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
    }
    return ACG_SUCCESS;
}

#ifdef ACG_HAVE_MPI
static int acggraphneighbour_send(
    const struct acggraphneighbour * neighbours,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    for (int i = 0; i < count; i++) {
        const struct acggraphneighbour * n = &neighbours[i];
        MPI_Send(&n->neighbourrank, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&n->neighbourpart, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&n->nbordernodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(n->bordernodes, n->nbordernodes, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&n->nghostnodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(n->ghostnodes, n->nghostnodes, MPI_ACGIDX_T, recipient, tag, comm);
    }
    return ACG_SUCCESS;
}

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

int acggraph_send(
    const struct acggraph * graphs,
    int count,
    int recipient,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    for (int i = 0; i < count; i++) {
        const struct acggraph * g = &graphs[i];

        /* partitioning information */
        MPI_Send(&g->nparts, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&g->parttag, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&g->nprocs, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&g->npparts, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&g->ownerrank, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(&g->ownerpart, 1, MPI_INT, recipient, tag, comm);

        /* graph/subgraph nodes */
        MPI_Send(&g->nnodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&g->npnodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        bool nodetags = g->nodetags;
        MPI_Send(&nodetags, 1, MPI_C_BOOL, recipient, tag, comm);
        if (nodetags) {
            MPI_Send(g->nodetags, g->npnodes, MPI_ACGIDX_T, recipient, tag, comm);
        }
        bool parentnodeidx = g->parentnodeidx;
        MPI_Send(&parentnodeidx, 1, MPI_C_BOOL, recipient, tag, comm);
        if (parentnodeidx) {
            MPI_Send(g->parentnodeidx, g->npnodes, MPI_ACGIDX_T, recipient, tag, comm);
        }

        /* graph/subgraph edges */
        MPI_Send(&g->nedges, 1, MPI_INT64_T, recipient, tag, comm);
        MPI_Send(&g->npedges, 1, MPI_INT64_T, recipient, tag, comm);
        bool edgetags = g->edgetags;
        MPI_Send(&edgetags, 1, MPI_C_BOOL, recipient, tag, comm);
        if (edgetags) {
            MPI_Send64(g->edgetags, g->npedges, MPI_INT64_T, recipient, tag, comm);
        }
        bool parentedgeidx = g->parentedgeidx;
        MPI_Send(&parentedgeidx, 1, MPI_C_BOOL, recipient, tag, comm);
        if (parentedgeidx) {
            MPI_Send64(g->parentedgeidx, g->npedges, MPI_INT64_T, recipient, tag, comm);
        }

        /* incidence relation (edges to unordered node pairs) */
        MPI_Send(&g->nodeidxbase, 1, MPI_INT, recipient, tag, comm);
        MPI_Send(g->nodenedges, g->npnodes, MPI_INT64_T, recipient, tag, comm);
        MPI_Send(g->srcnodeptr, g->npnodes+1, MPI_INT64_T, recipient, tag, comm);
        MPI_Send64(g->srcnodeidx, g->npedges, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send64(g->dstnodeidx, g->npedges, MPI_ACGIDX_T, recipient, tag, comm);

        /* interior, border and ghost nodes */
        MPI_Send(&g->nownednodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&g->ninnernodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&g->nbordernodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&g->bordernodeoffset, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&g->nghostnodes, 1, MPI_ACGIDX_T, recipient, tag, comm);
        MPI_Send(&g->ghostnodeoffset, 1, MPI_ACGIDX_T, recipient, tag, comm);

        /* interior and interface edges */
        MPI_Send(&g->ninneredges, 1, MPI_INT64_T, recipient, tag, comm);
        MPI_Send(&g->ninterfaceedges, 1, MPI_INT64_T, recipient, tag, comm);
        MPI_Send(g->nbordernodeinneredges, g->nbordernodes, MPI_INT64_T, recipient, tag, comm);
        MPI_Send(g->nbordernodeinterfaceedges, g->nbordernodes, MPI_INT64_T, recipient, tag, comm);

        /* neighbouring subgraphs */
        MPI_Send(&g->nneighbours, 1, MPI_INT, recipient, tag, comm);
        err = acggraphneighbour_send(
            g->neighbours, g->nneighbours, recipient, tag, comm, mpierrcode);
        if (err) return err;
    }
    return ACG_SUCCESS;
}

static int acggraphneighbour_recv(
    struct acggraphneighbour * neighbours,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    for (int i = 0; i < count; i++) {
        struct acggraphneighbour * n = &neighbours[i];
        MPI_Recv(&n->neighbourrank, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&n->neighbourpart, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&n->nbordernodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        n->bordernodes = malloc(n->nbordernodes*sizeof(*n->bordernodes));
        if (!n->bordernodes) return ACG_ERR_ERRNO;
        MPI_Recv(n->bordernodes, n->nbordernodes, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&n->nghostnodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        n->ghostnodes = malloc(n->nghostnodes*sizeof(*n->ghostnodes));
        if (!n->ghostnodes) return ACG_ERR_ERRNO;
        MPI_Recv(n->ghostnodes, n->nghostnodes, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
    }
    return ACG_SUCCESS;
}

int acggraph_recv(
    struct acggraph * graphs,
    int count,
    int sender,
    int tag,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err;
    for (int i = 0; i < count; i++) {
        struct acggraph * g = &graphs[i];

        /* partitioning information */
        MPI_Recv(&g->nparts, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->parttag, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->nprocs, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->npparts, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->ownerrank, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->ownerpart, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);

        /* graph/subgraph nodes */
        MPI_Recv(&g->nnodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->npnodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        bool nodetags;
        MPI_Recv(&nodetags, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (nodetags) {
            g->nodetags = malloc(g->npnodes*sizeof(*g->nodetags));
            if (!g->nodetags) return ACG_ERR_ERRNO;
            MPI_Recv(g->nodetags, g->npnodes, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { g->nodetags = NULL; }
        bool parentnodeidx;
        MPI_Recv(&parentnodeidx, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (parentnodeidx) {
            g->parentnodeidx = malloc(g->npnodes*sizeof(*g->parentnodeidx));
            if (!g->parentnodeidx) return ACG_ERR_ERRNO;
            MPI_Recv(g->parentnodeidx, g->npnodes, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { g->parentnodeidx = NULL; }

        /* graph/subgraph edges */
        MPI_Recv(&g->nedges, 1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->npedges, 1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        bool edgetags;
        MPI_Recv(&edgetags, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (edgetags) {
            g->edgetags = malloc(g->npedges*sizeof(*g->edgetags));
            if (!g->edgetags) return ACG_ERR_ERRNO;
            MPI_Recv64(g->edgetags, g->npedges, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { g->edgetags = NULL; }
        bool parentedgeidx;
        MPI_Recv(&parentedgeidx, 1, MPI_C_BOOL, sender, tag, comm, MPI_STATUS_IGNORE);
        if (parentedgeidx) {
            g->parentedgeidx = malloc(g->npedges*sizeof(*g->parentedgeidx));
            if (!g->parentedgeidx) return ACG_ERR_ERRNO;
            MPI_Recv64(g->parentedgeidx, g->npedges, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        } else { g->parentedgeidx = NULL; }

        /* incidence relation (edges to unordered node pairs) */
        MPI_Recv(&g->nodeidxbase, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        g->nodenedges = malloc(g->npnodes*sizeof(*g->nodenedges));
        if (!g->nodenedges) return ACG_ERR_ERRNO;
        MPI_Recv(g->nodenedges, g->npnodes, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        g->srcnodeptr = malloc((g->npnodes+1)*sizeof(*g->srcnodeptr));
        if (!g->srcnodeptr) { free(g->nodetags); return ACG_ERR_ERRNO; }
        MPI_Recv(g->srcnodeptr, g->npnodes+1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        g->srcnodeidx = malloc(g->npedges*sizeof(*g->srcnodeidx));
        if (!g->srcnodeidx) return ACG_ERR_ERRNO;
        MPI_Recv64(g->srcnodeidx, g->npedges, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        g->dstnodeidx = malloc(g->npedges*sizeof(*g->dstnodeidx));
        if (!g->dstnodeidx) return ACG_ERR_ERRNO;
        MPI_Recv64(g->dstnodeidx, g->npedges, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);

        /* interior, border and ghost nodes */
        MPI_Recv(&g->nownednodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->ninnernodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->nbordernodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->bordernodeoffset, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->nghostnodes, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->ghostnodeoffset, 1, MPI_ACGIDX_T, sender, tag, comm, MPI_STATUS_IGNORE);

        /* interior and interface edges */
        MPI_Recv(&g->ninneredges, 1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&g->ninterfaceedges, 1, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        g->nbordernodeinneredges = malloc(g->nbordernodes*sizeof(*g->nbordernodeinneredges));
        if (!g->nbordernodeinneredges) return ACG_ERR_ERRNO;
        MPI_Recv(g->nbordernodeinneredges, g->nbordernodes, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);
        g->nbordernodeinterfaceedges = malloc(g->nbordernodes*sizeof(*g->nbordernodeinterfaceedges));
        if (!g->nbordernodeinterfaceedges) return ACG_ERR_ERRNO;
        MPI_Recv(g->nbordernodeinterfaceedges, g->nbordernodes, MPI_INT64_T, sender, tag, comm, MPI_STATUS_IGNORE);

        /* neighbouring subgraphs */
        MPI_Recv(&g->nneighbours, 1, MPI_INT, sender, tag, comm, MPI_STATUS_IGNORE);
        g->neighbours = malloc(g->nneighbours*sizeof(*g->neighbours));
        if (!g->neighbours) return ACG_ERR_ERRNO;
        err = acggraphneighbour_recv(
            g->neighbours, g->nneighbours, sender, tag, comm, mpierrcode);
        if (err) return err;
    }
    return ACG_SUCCESS;
}

int acggraph_scatter(
    struct acggraph * sendgraphs,
    int sendcount,
    struct acggraph * recvgraphs,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err = ACG_SUCCESS, commsize, rank;
    err = MPI_Comm_size(comm, &commsize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Comm_rank(comm, &rank);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    if (rank == root) {
        int nparts = 0;
        for (int p = 0; p < commsize; p++) {
            if (sendcount > 0) {
                const struct acggraph * g = &sendgraphs[0];
                nparts = g->nparts;
                break;
            }
        }

        /* map subgraphs to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcount; i++) {
                struct acggraph * g = &sendgraphs[sendcount*p+i];
                g->nprocs = commsize;
                g->npparts = sendcount;
                g->ownerrank = p;
                g->ownerpart = i;
            }
        }

        /* map neighbouring subgraphs to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcount; i++) {
                struct acggraph * g = &sendgraphs[sendcount*p+i];
                for (int j = 0; j < g->nneighbours; j++) {
                    struct acggraphneighbour * n = &g->neighbours[j];
                    int q = n->neighbourpart;
                    if (q < 0 || q >= nparts)
                        return ACG_ERR_INDEX_OUT_OF_BOUNDS;
                    struct acggraph * h = &sendgraphs[q];
                    n->neighbourrank = h->ownerrank;
                    n->neighbourpart = h->ownerpart;
                }
            }
        }

        /* send from root process */
        for (int p = 0; p < commsize; p++) {
            if (rank != p) {
                err = acggraph_send(
                    &sendgraphs[sendcount*p], sendcount, p, p+1,
                    comm, mpierrcode);
                if (err) return err;
            } else {
                for (int i = 0; i < sendcount; i++) {
                    err = acggraph_copy(
                        &recvgraphs[i],
                        &sendgraphs[sendcount*p+i]);
                    if (err) return err;
                }
            }
        }
    } else {
        /* receive from root process */
        err = acggraph_recv(
            recvgraphs, recvcount, root, rank+1,
            comm, mpierrcode);
        if (err) return err;
    }
    return ACG_SUCCESS;
}

int acggraph_scatterv(
    struct acggraph * sendgraphs,
    const int * sendcounts,
    const int * displs,
    struct acggraph * recvgraphs,
    int recvcount,
    int root,
    MPI_Comm comm,
    int * mpierrcode)
{
    int err = ACG_SUCCESS, commsize, rank;
    err = MPI_Comm_size(comm, &commsize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Comm_rank(comm, &rank);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    if (rank == root) {
        int nparts = 0;
        for (int p = 0; p < commsize; p++) {
            if (sendcounts[p] > 0) {
                const struct acggraph * g = &sendgraphs[displs[p]+0];
                nparts = g->nparts;
                break;
            }
        }

        /* map subgraphs to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcounts[p]; i++) {
                struct acggraph * g = &sendgraphs[displs[p]+i];
                g->nprocs = commsize;
                g->npparts = sendcounts[p];
                g->ownerrank = p;
                g->ownerpart = i;
            }
        }

        /* map neighbouring subgraphs to ranks and part numbers */
        for (int p = 0; p < commsize; p++) {
            for (int i = 0; i < sendcounts[p]; i++) {
                struct acggraph * g = &sendgraphs[displs[p]+i];
                for (int j = 0; j < g->nneighbours; j++) {
                    struct acggraphneighbour * n = &g->neighbours[j];
                    int q = n->neighbourpart;
                    if (q < 0 || q >= nparts)
                        return ACG_ERR_INDEX_OUT_OF_BOUNDS;
                    struct acggraph * h = &sendgraphs[q];
                    n->neighbourrank = h->ownerrank;
                    n->neighbourpart = h->ownerpart;
                }
            }
        }

        /* send from root process */
        for (int p = 0; p < commsize; p++) {
            if (rank != p) {
                err = acggraph_send(
                    &sendgraphs[displs[p]], sendcounts[p], p, p+1,
                    comm, mpierrcode);
                if (err) return err;
            } else {
                for (int i = 0; i < sendcounts[p]; i++) {
                    err = acggraph_copy(
                        &recvgraphs[i],
                        &sendgraphs[displs[p]+i]);
                    if (err) return err;
                }
            }
        }
    } else {
        /* receive from root process */
        err = acggraph_recv(
            recvgraphs, recvcount, root, rank+1,
            comm, mpierrcode);
        if (err) return err;
    }
    return ACG_SUCCESS;
}
#endif

/*
 * halo exchange/update for partitioned and distributed graphs
 */

/**
 * ‘acggraph_halo()’ sets up a halo exchange communication pattern
 * to send and receive data associated with the “ghost” nodes of a
 * partitioned and distributed graph.
 */
int acggraph_halo(
    const struct acggraph * graph,
    struct acghalo * halo)
{
    /* sender */
    halo->nrecipients = graph->nneighbours;
    halo->recipients = malloc(halo->nrecipients*sizeof(*halo->recipients));
    if (!halo->recipients) return ACG_ERR_ERRNO;
    halo->sendcounts = malloc(halo->nrecipients*sizeof(*halo->sendcounts));
    if (!halo->sendcounts) { free(halo->recipients); return ACG_ERR_ERRNO; }
    halo->sdispls = malloc(halo->nrecipients*sizeof(*halo->sdispls));
    if (!halo->sdispls) {
        free(halo->sendcounts); free(halo->recipients);
        return ACG_ERR_ERRNO;
    }
    halo->sendsize = 0;
    for (int i = 0; i < graph->nneighbours; i++) {
        const struct acggraphneighbour * n = &graph->neighbours[i];
        halo->sendsize += n->nbordernodes;
    }
    halo->sendbufidx = malloc(halo->sendsize*sizeof(*halo->sendbufidx));
    if (!halo->sendbufidx) {
        free(halo->sdispls); free(halo->sendcounts); free(halo->recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < graph->nneighbours; i++) {
        const struct acggraphneighbour * n = &graph->neighbours[i];
        int q = n->neighbourrank;
        halo->recipients[i] = q;
        halo->sendcounts[i] = n->nbordernodes;
        halo->sdispls[i] = i > 0 ? halo->sdispls[i-1] + halo->sendcounts[i-1] : 0;
        for (int j = 0; j < n->nbordernodes; j++)
            halo->sendbufidx[halo->sdispls[i]+j] = graph->bordernodeoffset+n->bordernodes[j];
    }

    /* recipient */
    halo->nsenders = graph->nneighbours;
    halo->senders = malloc(halo->nsenders*sizeof(*halo->senders));
    if (!halo->senders) {
        free(halo->sendbufidx); free(halo->sdispls); free(halo->sendcounts); free(halo->recipients);
        return ACG_ERR_ERRNO;
    }
    halo->recvcounts = malloc(halo->nsenders*sizeof(*halo->recvcounts));
    if (!halo->recvcounts) {
        free(halo->senders);
        free(halo->sendbufidx); free(halo->sdispls); free(halo->sendcounts); free(halo->recipients);
        return ACG_ERR_ERRNO;
    }
    halo->rdispls = malloc(halo->nsenders*sizeof(*halo->rdispls));
    if (!halo->rdispls) {
        free(halo->recvcounts); free(halo->senders);
        free(halo->sendbufidx); free(halo->sdispls); free(halo->sendcounts); free(halo->recipients);
        return ACG_ERR_ERRNO;
    }
    halo->recvsize = 0;
    for (int i = 0; i < graph->nneighbours; i++) {
        const struct acggraphneighbour * n = &graph->neighbours[i];
        halo->recvsize += n->nghostnodes;
    }
    halo->recvbufidx = malloc(halo->recvsize*sizeof(*halo->recvbufidx));
    if (!halo->recvbufidx) {
        free(halo->rdispls); free(halo->recvcounts); free(halo->senders);
        free(halo->sendbufidx); free(halo->sdispls); free(halo->sendcounts); free(halo->recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < graph->nneighbours; i++) {
        const struct acggraphneighbour * n = &graph->neighbours[i];
        int q = n->neighbourrank;
        halo->senders[i] = q;
        halo->recvcounts[i] = n->nghostnodes;
        halo->rdispls[i] = i > 0 ? halo->rdispls[i-1] + halo->recvcounts[i-1] : 0;
        for (int j = 0; j < n->nghostnodes; j++)
            halo->recvbufidx[halo->rdispls[i]+j] = graph->ghostnodeoffset+n->ghostnodes[j];
    }
    halo->nexchanges = 0;
    halo->texchange = 0;
    halo->tpack = halo->tunpack = halo->tsendrecv = halo->tmpiirecv = halo->tmpisend = halo->tmpiwaitall = 0;
    halo->npack = halo->nunpack = halo->nmpiirecv = halo->nmpisend = 0;
    halo->Bpack = halo->Bunpack = halo->Bmpiirecv = halo->Bmpisend = 0;
    halo->maxexchangestats = ACG_HALO_MAX_EXCHANGE_STATS;
    halo->thaloexchangestats = malloc(halo->maxexchangestats*sizeof(*halo->thaloexchangestats));
    if (!halo->thaloexchangestats) return ACG_ERR_ERRNO;
    return ACG_SUCCESS;
}
