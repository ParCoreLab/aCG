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
 * halo exchange communication pattern
 */

#include "acg/config.h"
#include "acg/error.h"
#include "acg/halo.h"
#include "acg/sort.h"
#include "acg/time.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif

#ifdef ACG_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <errno.h>

#include <stdbool.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘acghalo_init()’ sets up a halo exchange pattern based on a
 * partitioned, unstructured computational mesh.
 */
int acghalo_init(
    struct acghalo * halo,
    int nsendnodes,
    const acgidx_t * sendnodetags,
    const int * sendnodenneighbours,
    const int * sendnodeneighbours,
    acgidx_t nrecvnodes,
    const acgidx_t * recvnodetags,
    const int * recvnodeparts)
{
    int err;

    /* 1. sort sending nodes by the part number of the recipient and
     * node number */
    acgidx_t sendsize = 0;
    for (acgidx_t i = 0; i < nsendnodes; i++) sendsize += sendnodenneighbours[i];
    int * sendnoderecipients = malloc(sendsize*sizeof(*sendnoderecipients));
    if (!sendnoderecipients) return ACG_ERR_ERRNO;
    int * sendbufidx = malloc(sendsize*sizeof(*sendbufidx));
    if (!sendbufidx) { free(sendnoderecipients); return ACG_ERR_ERRNO; }
    for (int i = 0, k = 0; i < nsendnodes; i++) {
        for (acgidx_t j = 0; j < sendnodenneighbours[i]; j++, k++) {
            sendnoderecipients[k] = sendnodeneighbours[k];
            sendbufidx[k] = i;
        }
    }
    err = acgradixsortpair_int(
        sendsize, sizeof(*sendnoderecipients), sendnoderecipients,
        sizeof(*sendbufidx), sendbufidx, NULL, NULL);
    if (err) { free(sendbufidx); free(sendnoderecipients); return err; }

    /* 2. count number of recipients */
    int nrecipients = 0;
    for (acgidx_t i = 0; i < sendsize; ) {
        for (i++; i < sendsize &&
                 sendnoderecipients[i] == sendnoderecipients[i-1]; i++) {}
        nrecipients++;
    }

    /* 3. obtain a list of recipients together with message sizes and
     * displacements for each recipient */
    int * recipients = malloc(nrecipients*sizeof(*recipients));
    if (!recipients) {
        free(sendbufidx); free(sendnoderecipients);
        return ACG_ERR_ERRNO;
    }
    int * sendcounts = malloc(nrecipients*sizeof(*sendcounts));
    if (!sendcounts) {
        free(recipients); free(sendbufidx); free(sendnoderecipients);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0, j = 0; i < sendsize; ) {
        recipients[j] = sendnoderecipients[i];
        sendcounts[j] = 1;
        for (i++; i < sendsize &&
                 sendnoderecipients[i] == sendnoderecipients[i-1]; i++)
        { sendcounts[j]++; }
        j++;
    }
    free(sendnoderecipients); 
    int * sdispls = malloc(nrecipients*sizeof(*sdispls));
    if (!sdispls) {
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    if (nrecipients > 0) sdispls[0] = 0;
    for (int i = 1; i < nrecipients; i++) sdispls[i] = sdispls[i-1] + sendcounts[i-1];

    /* 4. sort receiving nodes by the part number of the sender and
     * node number */
    acgidx_t recvsize = nrecvnodes;
    acgidx_t * recvnodesenders = malloc(recvsize*sizeof(*recvnodesenders));
    if (!recvnodesenders) {
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0; i < recvsize; i++) recvnodesenders[i] = recvnodeparts[i];
    acgidx_t * recvnodetagssorted = malloc(recvsize*sizeof(*recvnodetagssorted));
    if (!recvnodetagssorted) {
        free(recvnodesenders);
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0; i < recvsize; i++) recvnodetagssorted[i] = recvnodetags[i];
    int64_t * recvnodeinvperm = malloc(recvsize*sizeof(*recvnodeinvperm));
    if (!recvnodeinvperm) {
        free(recvnodetagssorted); free(recvnodesenders);
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    err = acgradixsortpair_idx_t(
        recvsize, sizeof(*recvnodesenders), recvnodesenders,
        sizeof(*recvnodetagssorted), recvnodetagssorted,
        NULL, recvnodeinvperm);
    if (err) {
        free(recvnodeinvperm); free(recvnodetagssorted); free(recvnodesenders);
        free(sendcounts); free(recipients); free(sendbufidx);
        return err;
    }
    free(recvnodetagssorted);
    int * recvbufidx = malloc(recvsize*sizeof(*recvbufidx));
    if (!recvbufidx) {
        free(recvnodeinvperm); free(recvnodesenders);
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0; i < recvsize; i++) {
        if (recvnodeinvperm[i] > INT_MAX) {
            free(recvbufidx); free(recvnodeinvperm); free(recvnodesenders);
            free(sendcounts); free(recipients); free(sendbufidx);
            return ACG_ERR_INDEX_OUT_OF_BOUNDS;
        }
        recvbufidx[i] = recvnodeinvperm[i];
    }
    free(recvnodeinvperm);

    /* 2. count number of senders */
    int nsenders = 0;
    for (acgidx_t i = 0; i < recvsize; ) {
        for (i++; i < recvsize && recvnodesenders[i] == recvnodesenders[i-1]; i++) {}
        nsenders++;
    }

    /* 3. obtain a list of senders together with message sizes and
     * displacements for each sender */
    int * senders = malloc(nsenders*sizeof(*senders));
    if (!senders) {
        free(recvbufidx); free(recvnodesenders);
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    int * recvcounts = malloc(nsenders*sizeof(*recvcounts));
    if (!recvcounts) {
        free(senders); free(recvbufidx); free(recvnodesenders);
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    for (acgidx_t i = 0, j = 0; i < recvsize; ) {
        senders[j] = recvnodesenders[i];
        recvcounts[j] = 1;
        for (i++; i < recvsize &&
                 recvnodesenders[i] == recvnodesenders[i-1]; i++)
        { recvcounts[j]++; }
        j++;
    }
    free(recvnodesenders);
    int * rdispls = malloc(nsenders*sizeof(*rdispls));
    if (!rdispls) {
        free(recvcounts); free(senders); free(recvbufidx);
        free(sendcounts); free(recipients); free(sendbufidx);
        return ACG_ERR_ERRNO;
    }
    if (nsenders > 0) rdispls[0] = 0;
    for (int i = 1; i < nsenders; i++) rdispls[i] = rdispls[i-1] + recvcounts[i-1];

    halo->nrecipients = nrecipients;
    halo->recipients = recipients;
    halo->sendcounts = sendcounts;
    halo->sdispls = sdispls;
    halo->sendsize = sendsize;
    halo->sendbufidx = sendbufidx;
    halo->nsenders = nsenders;
    halo->senders = senders;
    halo->recvcounts = recvcounts;
    halo->rdispls = rdispls;
    halo->recvsize = recvsize;
    halo->recvbufidx = recvbufidx;
    halo->nexchanges = 0;
    halo->texchange = 0;
    halo->tpack = halo->tunpack = halo->tsendrecv = halo->tmpiirecv = halo->tmpisend = halo->tmpiwaitall = 0;
    halo->npack = halo->nunpack = halo->nmpiirecv = halo->nmpisend = 0;
    halo->Bpack = halo->Bunpack = halo->Bmpiirecv = halo->Bmpisend = 0;
    halo->maxexchangestats = ACG_HALO_MAX_EXCHANGE_STATS;
    halo->thaloexchangestats = malloc(halo->maxexchangestats*sizeof(*halo->thaloexchangestats));
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_free()’ frees resources associated with a halo exchange.
 */
void acghalo_free(
    struct acghalo * halo)
{
    free(halo->recvbufidx);
    free(halo->rdispls);
    free(halo->recvcounts);
    free(halo->senders);
    free(halo->sendbufidx);
    free(halo->sdispls);
    free(halo->sendcounts);
    free(halo->recipients);
    free(halo->thaloexchangestats);
}

/**
 * ‘acghalo_copy()’ creates a copy of a halo exchange data structure.
 */
int acghalo_copy(
    struct acghalo * dst,
    const struct acghalo * src)
{
    int nrecipients = src->nrecipients;
    int * recipients = malloc(nrecipients*sizeof(*recipients));
    if (!recipients) return ACG_ERR_ERRNO;
    for (int i = 0; i < nrecipients; i++) recipients[i] = src->recipients[i];
    int * sendcounts = malloc(nrecipients*sizeof(*sendcounts));
    if (!sendcounts) { free(recipients); return ACG_ERR_ERRNO; }
    for (int i = 0; i < nrecipients; i++) sendcounts[i] = src->sendcounts[i];
    int * sdispls = malloc(nrecipients*sizeof(*sdispls));
    if (!sdispls) { free(sendcounts); free(recipients); return ACG_ERR_ERRNO; }
    for (int i = 0; i < nrecipients; i++) sdispls[i] = src->sdispls[i];
    int sendsize = src->sendsize;
    int * sendbufidx = malloc(sendsize*sizeof(*sendbufidx));
    if (!sendbufidx) { free(sdispls); free(sendcounts); free(recipients); return ACG_ERR_ERRNO; }
    for (int i = 0; i < sendsize; i++) sendbufidx[i] = src->sendbufidx[i];
    int nsenders = src->nsenders;
    int * senders = malloc(nsenders*sizeof(*senders));
    if (!senders) {
        free(sendbufidx); free(sdispls); free(sendcounts); free(recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < nsenders; i++) senders[i] = src->senders[i];
    int * recvcounts = malloc(nsenders*sizeof(*recvcounts));
    if (!recvcounts) {
        free(senders);
        free(sendbufidx); free(sdispls); free(sendcounts); free(recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < nsenders; i++) recvcounts[i] = src->recvcounts[i];
    int * rdispls = malloc(nsenders*sizeof(*rdispls));
    if (!rdispls) {
        free(recvcounts); free(senders);
        free(sendbufidx); free(sdispls); free(sendcounts); free(recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < nsenders; i++) rdispls[i] = src->rdispls[i];
    int recvsize = src->recvsize;
    int * recvbufidx = malloc(recvsize*sizeof(*recvbufidx));
    if (!recvbufidx) {
        free(rdispls); free(recvcounts); free(senders);
        free(sendbufidx); free(sdispls); free(sendcounts); free(recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < recvsize; i++) recvbufidx[i] = src->recvbufidx[i];
    int maxexchangestats = src->maxexchangestats;
    double (* thaloexchangestats)[4] = malloc(maxexchangestats*sizeof(*thaloexchangestats));
    if (!thaloexchangestats) {
        free(recvbufidx); free(rdispls); free(recvcounts); free(senders);
        free(sendbufidx); free(sdispls); free(sendcounts); free(recipients);
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < src->nexchanges; i++) {
        thaloexchangestats[i][0] = src->thaloexchangestats[i][0];
        thaloexchangestats[i][1] = src->thaloexchangestats[i][1];
        thaloexchangestats[i][2] = src->thaloexchangestats[i][2];
        thaloexchangestats[i][3] = src->thaloexchangestats[i][3];
    }

    dst->nrecipients = nrecipients;
    dst->recipients = recipients;
    dst->sendcounts = sendcounts;
    dst->sdispls = sdispls;
    dst->sendsize = sendsize;
    dst->sendbufidx = sendbufidx;
    dst->nsenders = nsenders;
    dst->senders = senders;
    dst->recvcounts = recvcounts;
    dst->rdispls = rdispls;
    dst->recvsize = recvsize;
    dst->recvbufidx = recvbufidx;
    dst->nexchanges = src->nexchanges;
    dst->texchange = src->texchange;
    dst->tpack = src->tpack;
    dst->tunpack = src->tunpack;
    dst->tsendrecv = src->tsendrecv;
    dst->tmpiirecv = src->tmpiirecv;
    dst->tmpisend = src->tmpisend;
    dst->tmpiwaitall = src->tmpiwaitall;
    dst->npack = src->npack;
    dst->nunpack = src->nunpack;
    dst->nmpiirecv = src->nmpiirecv;
    dst->nmpisend = src->nmpisend;
    dst->Bpack = src->Bpack;
    dst->Bunpack = src->Bunpack;
    dst->Bmpiirecv = src->Bmpiirecv;
    dst->Bmpisend = src->Bmpisend;
    dst->maxexchangestats = src->maxexchangestats;
    dst->thaloexchangestats = thaloexchangestats;
    return ACG_SUCCESS;
}

/*
 * output (e.g., for debugging)
 */

int acghalo_fwrite(
    FILE * f,
    const struct acghalo * halo)
{
    fprintf(f, "nrecipients: %d\n", halo->nrecipients);
    fprintf(f, "recipients: [");
    for (int i = 0; i < halo->nrecipients; i++) fprintf(f, " %d", halo->recipients[i]);
    fprintf(f, " ]\n");
    fprintf(f, "sendcounts: [");
    for (int i = 0; i < halo->nrecipients; i++) fprintf(f, " %d", halo->sendcounts[i]);
    fprintf(f, " ]\n");
    fprintf(f, "sdispls: [");
    for (int i = 0; i < halo->nrecipients; i++) fprintf(f, " %d", halo->sdispls[i]);
    fprintf(f, " ]\n");
    fprintf(f, "sendsize: %d\n", halo->sendsize);
    fprintf(f, "sendbufidx: [");
    for (int i = 0; i < halo->sendsize; i++) fprintf(f, " %d", halo->sendbufidx[i]);
    fprintf(f, " ]\n");
    fprintf(f, "nsenders: %d\n", halo->nsenders);
    fprintf(f, "senders: [");
    for (int i = 0; i < halo->nsenders; i++) fprintf(f, " %d", halo->senders[i]);
    fprintf(f, " ]\n");
    fprintf(f, "recvcounts: [");
    for (int i = 0; i < halo->nsenders; i++) fprintf(f, " %d", halo->recvcounts[i]);
    fprintf(f, " ]\n");
    fprintf(f, "rdispls: [");
    for (int i = 0; i < halo->nsenders; i++) fprintf(f, " %d", halo->rdispls[i]);
    fprintf(f, " ]\n");
    fprintf(f, "recvsize: %d\n", halo->recvsize);
    fprintf(f, "recvbufidx: [");
    for (int i = 0; i < halo->recvsize; i++) fprintf(f, " %d", halo->recvbufidx[i]);
    fprintf(f, " ]\n");
    return ACG_SUCCESS;
}

/*
 * packing and unpacking messages
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acghalo_pack()’ packs messages for sending in a halo exchange.
 *
 * Data is copied from the array ‘srcbuf’, which is of length
 * ‘srcbufsize’ and contains elements of the type specified by
 * ‘datatype’, to the ‘sendbuf’ array, which must be of length
 * ‘sendbufsize’. The number of elements to be copied is given by
 * ‘sendbufsize’, and the ‘i’th element in ‘sendbuf’ is copied from
 * the position ‘srcbufidx[i]’ in ‘srcbuf’.
 *
 * The arrays ‘sendbuf’ and ‘srcbuf’ must not overlap.
 */
int acghalo_pack(
    int sendbufsize,
    void * restrict sendbuf,
    MPI_Datatype datatype,
    int srcbufsize,
    const void * restrict srcbuf,
    const int * restrict srcbufidx,
    int64_t * nbytes,
    int * mpierrcode)
{
    if (datatype == MPI_DOUBLE) {
        for (int i = 0; i < sendbufsize; i++) {
#ifdef ACG_DEBUG_HALO
            fprintf(stderr, "%s: packing value from location %d to %d\n", __func__, i, srcbufidx[i]);
#endif
            ((double *) sendbuf)[i] = ((const double *) srcbuf)[srcbufidx[i]];
        }
        if (nbytes) *nbytes += sendbufsize*(2*sizeof(double)+sizeof(*srcbufidx));
    } else {
        int datatypesize;
        int err = MPI_Type_size(datatype, &datatypesize);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        for (int i = 0; i < sendbufsize; i++) {
#ifdef ACG_DEBUG_HALO
            fprintf(stderr, "%s: packing value from location %d to %d\n", __func__, i, srcbufidx[i]);
#endif
            void * restrict dst = (char *) sendbuf+datatypesize*i;
            const void * restrict src = (char*) srcbuf+datatypesize*srcbufidx[i];
            memcpy(dst, src, datatypesize);
        }
        if (nbytes) *nbytes += sendbufsize*(2*datatypesize+sizeof(*srcbufidx));
    }
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_unpack()’ unpacks messages received in a halo exchange.
 *
 * Data is copied to the array ‘dstbuf’, which is of length
 * ‘dstbufsize’ and contains elements of the type specified by
 * ‘datatype’, from the ‘recvbuf’ array, which must be of length
 * ‘recvbufsize’. The number of elements to be copied is given by
 * ‘recvbufsize’, and the ‘i’th element in ‘recvbuf’ is copied to the
 * position ‘dstbufidx[i]’ in ‘dstbuf’.
 *
 * The arrays ‘dstbuf’ and ‘recvbuf’ must not overlap.
 */
int acghalo_unpack(
    int recvbufsize,
    const void * restrict recvbuf,
    MPI_Datatype datatype,
    int dstbufsize,
    void * restrict dstbuf,
    const int * restrict dstbufidx,
    int64_t * nbytes,
    int * mpierrcode)
{
    if (datatype == MPI_DOUBLE) {
        for (int i = 0; i < recvbufsize; i++) {
#ifdef ACG_DEBUG_HALO
            fprintf(stderr, "%s: unpacking value from location %d to %d\n", __func__, i, dstbufidx[i]);
#endif
            ((double *) dstbuf)[dstbufidx[i]] = ((const double *) recvbuf)[i];
        }
        if (nbytes) *nbytes += recvbufsize*(2*sizeof(double)+sizeof(*dstbufidx));
    } else {
        int datatypesize;
        int err = MPI_Type_size(datatype, &datatypesize);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        for (int i = 0; i < recvbufsize; i++) {
#ifdef ACG_DEBUG_HALO
            fprintf(stderr, "%s: unpacking value from location %d to %d\n", __func__, i, dstbufidx[i]);
#endif
            void * restrict dst = (char *) dstbuf+datatypesize*dstbufidx[i];
            const void * restrict src = (char*) recvbuf+datatypesize*i;
            memcpy(dst, src, datatypesize);
        }
        if (nbytes) *nbytes += recvbufsize*(2*datatypesize+sizeof(*dstbufidx));
    }
    return ACG_SUCCESS;
}

/*
 * halo communication routines
 */

/**
 * ‘halo_alltoallv_isend()’ posts (non-blocking) sends for a halo exchange.
 *
 * The array ‘recipients’, which is of length ‘nrecipients’, specifies
 * the processes to send messages to. Moreover, the number of elements
 * to send to each process is given by the array ‘sendcounts’. For
 * every sender, a send is posted using ‘MPI_Isend()’ with the given
 * tag, ‘tag’.
 *
 * On each process, ‘sendbuf’ is an array containing data to send to
 * neighbouring processes. More specifically, data sent to the process
 * with rank ‘recipients[p]’ must be stored contiguously in ‘sendbuf’,
 * starting at the index ‘sdispls[p]’. Thus, the length of ‘sendbuf’
 * must be at least equal to the maximum of ‘sdispls[p]+sendcounts[p]’
 * for any recieving neighbouring process ‘p’.
 */
static int halo_alltoallv_isend(
    const void * sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    MPI_Datatype sendtype,
    int tag,
    MPI_Comm comm,
    MPI_Request * sendreqs,
    int * mpierrcode,
    int64_t * nmsgs,
    int64_t * nbytes)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int sendtypesize;
    int err = MPI_Type_size(sendtype, &sendtypesize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    for (int p = 0; p < nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting MPI_Isend %d of %d from rank %d of size %d at offset %d for recipient %d with tag %d\n", __func__, p+1, nrecipients, rank, sendcounts[p], sdispls[p], recipients[p], tag);
#endif
        void * sendbufp = (char *) sendbuf + sendtypesize*sdispls[p];
        err = MPI_Isend(
            sendbufp, sendcounts[p], sendtype, recipients[p],
            tag, comm, &sendreqs[p]);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        if (nbytes) *nbytes += sendcounts[p]*sendtypesize;
    }
    if (nmsgs) *nmsgs += nrecipients;
    return ACG_SUCCESS;
}

/**
 * ‘halo_alltoallv_irecv()’ posts nonblocking receives for a halo
 * exchange.
 *
 * The array ‘senders’ is of length ‘nsenders’, and specifies the
 * processes from which to receive messages. Moreover, the number of
 * elements to receive from each process is given by the array
 * ‘recvcounts’. For every sender, a receive is posted using
 * ‘MPI_Irecv()’ with the given tag, ‘tag’.
 *
 * On each process, ‘recvbuf’ is a buffer used to receive data from
 * neighbouring processes. More specifically, data received from the
 * process with rank ‘senders[p]’ will be stored contiguously in
 * ‘recvbuf’, starting at the index ‘rdispls[p]’. Thus, the length of
 * ‘recvbuf’ must be at least equal to the maximum of
 * ‘rdispls[p]+recvcounts[p]’ for any sending neighbour ‘p’.
 *
 * The array ‘recvreqs’ must be of length ‘nsenders’, and it is used
 * to store MPI requests associated with the nonblocking
 * receives. These requests can be used to check if the corresponding
 * message has been received (e.g., using ‘MPI_Test()’) or to wait for
 * a message to be received (e.g., using ‘MPI_Wait()’).
 */
static int halo_alltoallv_irecv(
    void * recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    MPI_Datatype recvtype,
    int tag,
    MPI_Comm comm,
    MPI_Request * recvreqs,
    int * mpierrcode,
    int64_t * nmsgs,
    int64_t * nbytes)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int recvtypesize;
    int err = MPI_Type_size(recvtype, &recvtypesize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    for (int p = 0; p < nsenders; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting MPI_Irecv %d of %d for rank %d of size %d at offset %d from sender %d with tag %d\n", __func__, p+1, nsenders, rank, recvcounts[p], rdispls[p], senders[p], tag);
#endif
        void * recvbufp = (char *) recvbuf + recvtypesize*rdispls[p];
        err = MPI_Irecv(
            recvbufp, recvcounts[p], recvtype, senders[p],
            tag, comm, &recvreqs[p]);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        if (nbytes) *nbytes += recvcounts[p]*recvtypesize;
    }
    if (nmsgs) *nmsgs += nsenders;
    return ACG_SUCCESS;
}

/**
 * ‘halo_alltoallv_mpi()’ performs an neighbour all-to-all halo exchange.
 *
 * This assumes that messages have already been packed into a sending
 * buffer and will be unpacked from a receiving buffer afterwards.
 *
 * The array ‘recipients’, which is of length ‘nrecipients’, specifies
 * the processes to send messages to (i.e., recipients). Moreover, the
 * number of elements to send to each recipient is given by the array
 * ‘sendcounts’. For every recipient, a send is posted using
 * ‘MPI_Isend()’ with the given tag, ‘tag’.
 *
 * On each process, ‘sendbuf’ is an array containing data to send to
 * neighbouring processes. More specifically, data sent to the process
 * with rank ‘recipients[p]’ must be stored contiguously in ‘sendbuf’,
 * starting at the index ‘sdispls[p]’. Thus, the length of ‘sendbuf’
 * must be at least equal to the maximum of ‘sdispls[p]+sendcounts[p]’
 * for any recieving neighbouring process ‘p’.
 *
 * The array ‘senders’ is of length ‘nsenders’, and specifies the
 * processes from which to receive messages. Moreover, the number of
 * elements to receive from each process is given by the array
 * ‘recvcounts’. For every sender, a receive is posted using
 * ‘MPI_Irecv()’ with the given tag, ‘tag’.
 *
 * On each process, ‘recvbuf’ is a buffer used to receive data from
 * neighbouring processes. More specifically, data received from the
 * process with rank ‘senders[p]’ will be stored contiguously in
 * ‘recvbuf’, starting at the index ‘rdispls[p]’. Thus, the length of
 * ‘recvbuf’ must be at least equal to the maximum of
 * ‘rdispls[p]+recvcounts[p]’ for any sending neighbour ‘p’.
 */
static int halo_alltoallv_mpi(
    const void * sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    MPI_Datatype sendtype,
    MPI_Request * sendreqs,
    int recvsize,
    void * recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    MPI_Datatype recvtype,
    MPI_Request * recvreqs,
    int tag,
    MPI_Comm comm,
    int * mpierrcode,
    int64_t * nsendmsgs,
    int64_t * nsendbytes,
    int64_t * nrecvmsgs,
    int64_t * nrecvbytes,
    bool wait)
{
    int err = ACG_SUCCESS, errnocode = 0;

    /* 1. post non-blocking message receives */
    err = halo_alltoallv_irecv(
        recvbuf, nsenders, senders, recvcounts, rdispls, recvtype,
        tag, comm, recvreqs, mpierrcode, nrecvmsgs, nrecvbytes);
    if (err) return err;

    /* 2. post non-blocking message sends */
    err = halo_alltoallv_isend(
        sendbuf, nrecipients, recipients, sendcounts, sdispls, sendtype,
        tag, comm, sendreqs, mpierrcode, nsendmsgs, nsendbytes);
    if (err) return err;

    /* 3. wait for non-blocking sends & receives to complete */
    if (wait) {
        MPI_Waitall(nrecipients, sendreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(nsenders, recvreqs, MPI_STATUSES_IGNORE);
    }
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_exchange()’ performs a halo exchange.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange(
    struct acghalo * halo,
    int srcbufsize,
    const void * srcbuf,
    MPI_Datatype sendtype,
    int dstbufsize,
    void * dstbuf,
    MPI_Datatype recvtype,
    int sendbufsize,
    void * sendbuf,
    MPI_Request * sendreqs,
    int recvbufsize,
    void * recvbuf,
    MPI_Request * recvreqs,
    MPI_Comm comm,
    int tag,
    int * mpierrcode)
{
    int err;
    acgtime_t t0, t1;
    acgtime_t tpack0, tpack1;
    acgtime_t tunpack0, tunpack1;
    acgtime_t talltoallv0, talltoallv1;
    halo->nexchanges++;
    gettime(&t0);

    int sendtypesize, recvtypesize;
    err = MPI_Type_size(sendtype, &sendtypesize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    err = MPI_Type_size(recvtype, &recvtypesize);
    if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }

    if (sendbuf && sendbufsize < halo->sendsize) return ACG_ERR_INDEX_OUT_OF_BOUNDS;
    if (recvbuf && recvbufsize < halo->recvsize) return ACG_ERR_INDEX_OUT_OF_BOUNDS;

    /* 1. if needed, allocate storage for intermediate buffers */
    void * tmpsendbuf = sendbuf ? NULL : malloc(halo->sendsize*sendtypesize);
    if (!sendbuf && !tmpsendbuf) return ACG_ERR_ERRNO;
    void * tmprecvbuf = recvbuf ? NULL : malloc(halo->recvsize*recvtypesize);
    if (!recvbuf && !tmprecvbuf) { free(tmpsendbuf); return ACG_ERR_ERRNO; }
    MPI_Request * tmpsendreqs =
        sendreqs ? NULL : malloc(halo->nrecipients*sizeof(*tmpsendreqs));
    if (!sendreqs && !tmpsendreqs) return ACG_ERR_ERRNO;
    MPI_Request * tmprecvreqs =
        recvreqs ? NULL : malloc(halo->nsenders*sizeof(*tmprecvreqs));
    if (!recvreqs && !tmprecvreqs) return ACG_ERR_ERRNO;

    /* 2. pack data for sending */
    gettime(&tpack0);
    err = acghalo_pack(
        halo->sendsize, sendbuf ? sendbuf : tmpsendbuf, sendtype,
        srcbufsize, srcbuf, halo->sendbufidx, &halo->Bpack, mpierrcode);
    if (err) { gettime(&t1); halo->texchange += elapsed(t0,t1); return err; }
    gettime(&tpack1); halo->tpack += elapsed(tpack0,tpack1); halo->npack++;

    /* 3. exchange messages */
    err = halo_alltoallv_mpi(
        sendbuf ? sendbuf : tmpsendbuf, halo->nrecipients, halo->recipients,
        halo->sendcounts, halo->sdispls, sendtype,
        sendreqs ? sendreqs : tmpsendreqs,
        halo->recvsize, recvbuf ? recvbuf : tmprecvbuf, halo->nsenders, halo->senders,
        halo->recvcounts, halo->rdispls, recvtype,
        recvreqs ? recvreqs : tmprecvreqs,
        tag, comm, mpierrcode,
        &halo->nmpisend, &halo->Bmpisend, &halo->nmpiirecv, &halo->Bmpiirecv, true);
    if (err) { gettime(&t1); halo->texchange += elapsed(t0,t1); return err; }

    /* 4. unpack received data */
    gettime(&tunpack0);
    err = acghalo_unpack(
        halo->recvsize, recvbuf ? recvbuf : tmprecvbuf, recvtype,
        dstbufsize, dstbuf, halo->recvbufidx, &halo->Bunpack, mpierrcode);
    if (err) { gettime(&t1); halo->texchange += elapsed(t0,t1); return err; }
    gettime(&tunpack1); halo->tunpack += elapsed(tunpack0,tunpack1); halo->nunpack++;

    /* 5. clean up intermediate buffers */
    if (tmprecvbuf) free(tmprecvbuf);
    if (tmpsendbuf) free(tmpsendbuf);
    if (tmpsendreqs) free(tmpsendreqs);
    if (tmprecvreqs) free(tmprecvreqs);
    gettime(&t1); halo->texchange += elapsed(t0,t1);
    return ACG_SUCCESS;
}
#endif

/*
 * halo communication with CUDA-aware MPI
 */

/**
 * ‘acghaloexchange_init()’ allocate additional storage needed to
 * perform a halo exchange.
 */
int acghaloexchange_init(
    struct acghaloexchange * haloexchange,
    const struct acghalo * halo,
    enum acgdatatype sendtype,
    enum acgdatatype recvtype,
    const struct acgcomm * comm)
{
    /* allocate storage for intermediate send/receive buffers */
    int sendtypesize, recvtypesize;
    int err = acgdatatype_size(sendtype, &sendtypesize);
    if (err) return err;
    err = acgdatatype_size(recvtype, &recvtypesize);
    if (err) return err;
    void * sendbuf = malloc(halo->sendsize*sendtypesize);
    if (!sendbuf) return ACG_ERR_ERRNO;
    void * recvbuf = malloc(halo->recvsize*recvtypesize);
    if (!recvbuf) { free(sendbuf); return ACG_ERR_ERRNO; }

    /* allocate storage for requests */
    void * sendreqs = NULL, * recvreqs = NULL;
    if (comm->type == acgcomm_mpi || comm->type == acgcomm_nvshmem) {
#if defined(ACG_HAVE_MPI)
        sendreqs = malloc(halo->nrecipients*sizeof(MPI_Request));
        if (!sendreqs) { free(recvbuf); free(sendbuf); return ACG_ERR_ERRNO; }
        recvreqs = malloc(halo->nsenders*sizeof(MPI_Request));
        if (!recvreqs) { free(sendreqs); free(recvbuf); free(sendbuf); return ACG_ERR_ERRNO; }
#else
        return ACG_ERR_MPI_NOT_SUPPORTED;
#endif
    }

    haloexchange->sendtype = sendtype;
    haloexchange->recvtype = recvtype;
    haloexchange->sendbuf = sendbuf;
    haloexchange->recvbuf = recvbuf;
    haloexchange->sendreqs = sendreqs;
    haloexchange->recvreqs = recvreqs;
#if defined(ACG_HAVE_CUDA) || defined(ACG_HAVE_HIP)
    haloexchange->d_sendbuf = NULL;
    haloexchange->d_recvbuf = NULL;
    haloexchange->d_sendbufidx = NULL;
    haloexchange->d_recvbufidx = NULL;
    haloexchange->d_recipients = NULL;
    haloexchange->d_sendcounts = NULL;
    haloexchange->d_sdispls = NULL;
    haloexchange->d_senders = NULL;
    haloexchange->d_recvcounts = NULL;
    haloexchange->d_rdispls = NULL;
    haloexchange->d_putdispls = NULL;
    haloexchange->d_putranks = NULL;
    haloexchange->d_getranks = NULL;
#if defined(ACG_HAVE_CUDA)
    haloexchange->cudastream = NULL;
#endif
#if defined(ACG_HAVE_HIP)
    haloexchange->hipstream = NULL;
#endif
    haloexchange->d_received = NULL;
    haloexchange->d_readytoreceive = NULL;
    haloexchange->putdispls = NULL;
    haloexchange->putranks = NULL;
    haloexchange->getranks = NULL;
    haloexchange->use_nvshmem = 0;
    haloexchange->maxevents = 0;
    haloexchange->nevents = 0;
#if defined(ACG_HAVE_CUDA)
    haloexchange->cudaevents = NULL;
#endif
#if defined(ACG_HAVE_HIP)
    haloexchange->hipevents = NULL;
#endif
#endif
    return ACG_SUCCESS;
}

#if defined(ACG_HAVE_CUDA)
/**
 * ‘acghaloexchange_init_cuda()’ allocate additional storage needed
 * to perform a halo exchange for data residing on a CUDA device.
 */
int acghaloexchange_init_cuda(
    struct acghaloexchange * haloexchange,
    const struct acghalo * halo,
    enum acgdatatype sendtype,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    cudaStream_t stream)
{
    int err = acghaloexchange_init(haloexchange, halo, sendtype, recvtype, comm);
    if (err) return err;

    /* allocate storage for device-side send/receive buffers */
    int sendtypesize, recvtypesize;
    err = acgdatatype_size(sendtype, &sendtypesize);
    if (err) return err;
    err = acgdatatype_size(recvtype, &recvtypesize);
    if (err) return err;
    void * d_sendbuf = NULL, * d_recvbuf = NULL;
    int use_nvshmem = comm->type == acgcomm_nvshmem;
    if (use_nvshmem) {
        int commsize, rank;
        MPI_Comm_size(comm->mpicomm, &commsize);
        MPI_Comm_rank(comm->mpicomm, &rank);
        int maxsendsize = halo->sendsize;
        MPI_Allreduce(MPI_IN_PLACE, &maxsendsize, 1, MPI_INT, MPI_MAX, comm->mpicomm);
        int maxrecvsize = halo->recvsize;
        MPI_Allreduce(MPI_IN_PLACE, &maxrecvsize, 1, MPI_INT, MPI_MAX, comm->mpicomm);
        int errcode;
        err = acgcomm_nvshmem_malloc(&d_sendbuf, maxsendsize*sendtypesize, &errcode);
        if (err) return err;
        err = acgcomm_nvshmem_malloc(&d_recvbuf, maxrecvsize*recvtypesize, &errcode);
        if (err) { acgcomm_nvshmem_free(d_sendbuf); return err; }
    } else {
        err = cudaMalloc((void **) &d_sendbuf, halo->sendsize*sendtypesize);
        if (err) return ACG_ERR_CUDA;
        err = cudaMalloc((void **) &d_recvbuf, halo->recvsize*recvtypesize);
        if (err) { cudaFree(d_sendbuf); return ACG_ERR_CUDA; }
    }

    /* if NVSHMEM will be used, let each sender know the offset in the
    * receive buffer to use for its put operations */
    uint64_t * d_received = NULL, * d_readytoreceive = NULL;
    int * putdispls = NULL, * putranks = NULL, * getranks = NULL;
    if (use_nvshmem) {
        putdispls = malloc(halo->nrecipients*sizeof(*putdispls));
        if (!putdispls) return ACG_ERR_ERRNO;
        for (int i = 0; i < halo->nrecipients; i++) putdispls[i] = 0;
        int tag = 1;
        for (int p = 0; p < halo->nsenders; p++) {
            err = MPI_Isend(
                &halo->rdispls[p], 1, MPI_INT, halo->senders[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->recvreqs)[p]);
        }
        for (int p = 0; p < halo->nrecipients; p++) {
            err = MPI_Irecv(
                &putdispls[p], 1, MPI_INT, halo->recipients[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->sendreqs)[p]);
        }
        MPI_Waitall(halo->nrecipients, haloexchange->sendreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(halo->nsenders, haloexchange->recvreqs, MPI_STATUSES_IGNORE);

        putranks = malloc(halo->nrecipients*sizeof(*putranks));
        if (!putranks) return ACG_ERR_ERRNO;
        for (int i = 0; i < halo->nrecipients; i++) putranks[i] = 0;
        for (int p = 0; p < halo->nsenders; p++) {
            err = MPI_Isend(
                &p, 1, MPI_INT, halo->senders[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->recvreqs)[p]);
        }
        for (int p = 0; p < halo->nrecipients; p++) {
            err = MPI_Irecv(
                &putranks[p], 1, MPI_INT, halo->recipients[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->sendreqs)[p]);
        }
        MPI_Waitall(halo->nrecipients, haloexchange->sendreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(halo->nsenders, haloexchange->recvreqs, MPI_STATUSES_IGNORE);

        getranks = malloc(halo->nsenders*sizeof(*getranks));
        if (!getranks) return ACG_ERR_ERRNO;
        for (int i = 0; i < halo->nsenders; i++) getranks[i] = 0;
        for (int p = 0; p < halo->nrecipients; p++) {
            err = MPI_Isend(
                &p, 1, MPI_INT, halo->recipients[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->sendreqs)[p]);
        }
        for (int p = 0; p < halo->nsenders; p++) {
            err = MPI_Irecv(
                &getranks[p], 1, MPI_INT, halo->senders[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->recvreqs)[p]);
        }
        MPI_Waitall(halo->nsenders, haloexchange->recvreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(halo->nrecipients, haloexchange->sendreqs, MPI_STATUSES_IGNORE);

        /* allocate device-side storage for signals */
        int maxsenders = halo->nsenders;
        MPI_Allreduce(MPI_IN_PLACE, &maxsenders, 1, MPI_INT, MPI_MAX, comm->mpicomm);
        int maxrecipients = halo->nrecipients;
        MPI_Allreduce(MPI_IN_PLACE, &maxrecipients, 1, MPI_INT, MPI_MAX, comm->mpicomm);
        if (maxsenders > 0) {
            err = acgcomm_nvshmem_calloc((void **) &d_received, maxsenders, sizeof(*d_received), NULL);
            if (err) return err;
        }
        if (maxrecipients > 0) {
            err = acgcomm_nvshmem_calloc((void **) &d_readytoreceive, maxrecipients, sizeof(*d_readytoreceive), NULL);
            if (err) return err;
        }
    }

    /* copy buffers needed for packing/unpacking to device */
    void * d_sendbufidx, * d_recvbufidx;
    err = cudaMalloc((void **) &d_sendbufidx, halo->sendsize*sizeof(*halo->sendbufidx));
    if (err) {
        if (use_nvshmem) { acgcomm_nvshmem_free(d_recvbuf); acgcomm_nvshmem_free(d_sendbuf); }
        else { cudaFree(d_recvbuf); cudaFree(d_sendbuf); }
        return ACG_ERR_CUDA;
    }
    err = cudaMemcpy(d_sendbufidx, halo->sendbufidx, halo->sendsize*sizeof(*halo->sendbufidx), cudaMemcpyHostToDevice);
    if (err) {
        cudaFree(d_sendbufidx);
        if (use_nvshmem) { acgcomm_nvshmem_free(d_recvbuf); acgcomm_nvshmem_free(d_sendbuf); }
        else { cudaFree(d_recvbuf); cudaFree(d_sendbuf); }
        return ACG_ERR_CUDA;
    }
    err = cudaMalloc((void **) &d_recvbufidx, halo->recvsize*sizeof(*halo->recvbufidx));
    if (err) {
        cudaFree(d_sendbufidx);
        if (use_nvshmem) { acgcomm_nvshmem_free(d_recvbuf); acgcomm_nvshmem_free(d_sendbuf); }
        else { cudaFree(d_recvbuf); cudaFree(d_sendbuf); }
        return ACG_ERR_CUDA;
    }
    err = cudaMemcpy(d_recvbufidx, halo->recvbufidx, halo->recvsize*sizeof(*halo->recvbufidx), cudaMemcpyHostToDevice);
    if (err) {
        cudaFree(d_recvbufidx); cudaFree(d_sendbufidx);
        if (use_nvshmem) { acgcomm_nvshmem_free(d_recvbuf); acgcomm_nvshmem_free(d_sendbuf); }
        else { cudaFree(d_recvbuf); cudaFree(d_sendbuf); }
        return ACG_ERR_CUDA;
    }

    int * d_recipients;
    err = cudaMalloc((void **) &d_recipients, halo->nrecipients*sizeof(*halo->recipients));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_recipients, halo->recipients, halo->nrecipients*sizeof(*halo->recipients), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    int * d_sendcounts;
    err = cudaMalloc((void **) &d_sendcounts, halo->nrecipients*sizeof(*halo->sendcounts));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_sendcounts, halo->sendcounts, halo->nrecipients*sizeof(*halo->sendcounts), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    int * d_sdispls;
    err = cudaMalloc((void **) &d_sdispls, halo->nrecipients*sizeof(*halo->sdispls));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_sdispls, halo->sdispls, halo->nrecipients*sizeof(*halo->sdispls), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    int * d_senders;
    err = cudaMalloc((void **) &d_senders, halo->nsenders*sizeof(*halo->senders));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_senders, halo->senders, halo->nsenders*sizeof(*halo->senders), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    int * d_recvcounts;
    err = cudaMalloc((void **) &d_recvcounts, halo->nsenders*sizeof(*halo->recvcounts));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_recvcounts, halo->recvcounts, halo->nsenders*sizeof(*halo->recvcounts), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    int * d_rdispls;
    err = cudaMalloc((void **) &d_rdispls, halo->nsenders*sizeof(*halo->rdispls));
    if (err) return ACG_ERR_CUDA;
    err = cudaMemcpy(d_rdispls, halo->rdispls, halo->nsenders*sizeof(*halo->rdispls), cudaMemcpyHostToDevice);
    if (err) return ACG_ERR_CUDA;
    int * d_putdispls = NULL, * d_putranks = NULL, * d_getranks = NULL;
    if (use_nvshmem) {
        err = cudaMalloc((void **) &d_putdispls, halo->nrecipients*sizeof(*putdispls));
        if (err) fprintf(stderr, "%s:%d\n",__FILE__,__LINE__);
        if (err) return ACG_ERR_CUDA;
        err = cudaMemcpy(d_putdispls, putdispls, halo->nrecipients*sizeof(*putdispls), cudaMemcpyHostToDevice);
        if (err) fprintf(stderr, "%s:%d\n",__FILE__,__LINE__);
        if (err) return ACG_ERR_CUDA;
        err = cudaMalloc((void **) &d_putranks, halo->nrecipients*sizeof(*putranks));
        if (err) fprintf(stderr, "%s:%d\n",__FILE__,__LINE__);
        if (err) return ACG_ERR_CUDA;
        err = cudaMemcpy(d_putranks, putranks, halo->nrecipients*sizeof(*putranks), cudaMemcpyHostToDevice);
        if (err) fprintf(stderr, "%s:%d\n",__FILE__,__LINE__);
        if (err) return ACG_ERR_CUDA;
        err = cudaMalloc((void **) &d_getranks, halo->nsenders*sizeof(*getranks));
        if (err) fprintf(stderr, "%s:%d\n",__FILE__,__LINE__);
        if (err) return ACG_ERR_CUDA;
        err = cudaMemcpy(d_getranks, getranks, halo->nsenders*sizeof(*getranks), cudaMemcpyHostToDevice);
        if (err) fprintf(stderr, "%s:%d\n",__FILE__,__LINE__);
        if (err) return ACG_ERR_CUDA;
    }

    /* allocate storage for performance monitoring events */
    int maxevents = ACG_HALO_MAX_PERF_EVENTS;
    cudaEvent_t (* events)[4] = malloc(maxevents*sizeof(*events));
    if (!events) {
        cudaFree(d_recvbufidx); cudaFree(d_sendbufidx);
        if (use_nvshmem) { acgcomm_nvshmem_free(d_recvbuf); acgcomm_nvshmem_free(d_sendbuf); }
        else { cudaFree(d_recvbuf); cudaFree(d_sendbuf); }
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < maxevents; i++) {
        cudaEventCreate(&events[i][0]);
        cudaEventCreate(&events[i][1]);
        cudaEventCreate(&events[i][2]);
        cudaEventCreate(&events[i][3]);
    }

    /* set up persistent communications */
    if (comm->type == acgcomm_mpi) {
        MPI_Comm mpicomm = comm->mpicomm;
        int rank;
        MPI_Comm_rank(mpicomm, &rank);
        int tag = 99;
        for (int p = 0; p < halo->nsenders; p++) {
#if defined(ACG_DEBUG_HALO)
            fprintf(stderr, "%s: posting MPI_Irecv %d of %d for rank %d of size %d at offset %d from sender %d with tag %d\n", __func__, p+1, halo->nsenders, rank, halo->recvcounts[p], halo->rdispls[p], halo->senders[p], tag);
#endif
            void * recvbufp = (char *) d_recvbuf + recvtypesize*halo->rdispls[p];
            err = MPI_Recv_init(
                recvbufp, halo->recvcounts[p], acgdatatype_mpi(recvtype), halo->senders[p],
                tag, mpicomm, &((MPI_Request *) haloexchange->recvreqs)[p]);
            if (err) return ACG_ERR_MPI;
        }
        for (int p = 0; p < halo->nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
            fprintf(stderr, "%s: posting MPI_Isend %d of %d from rank %d of size %d at offset %d for recipient %d with tag %d\n", __func__, p+1, halo->nrecipients, rank, halo->sendcounts[p], halo->sdispls[p], halo->recipients[p], tag);
#endif
            void * sendbufp = (char *) d_sendbuf + sendtypesize*halo->sdispls[p];
            err = MPI_Send_init(
                sendbufp, halo->sendcounts[p], acgdatatype_mpi(sendtype), halo->recipients[p],
                tag, mpicomm, &((MPI_Request *) haloexchange->sendreqs)[p]);
            if (err) return ACG_ERR_MPI;
        }
    }

    haloexchange->d_sendbuf = d_sendbuf;
    haloexchange->d_recvbuf = d_recvbuf;
    haloexchange->d_sendbufidx = d_sendbufidx;
    haloexchange->d_recvbufidx = d_recvbufidx;
    haloexchange->d_recipients = d_recipients;
    haloexchange->d_sendcounts = d_sendcounts;
    haloexchange->d_sdispls = d_sdispls;
    haloexchange->d_senders = d_senders;
    haloexchange->d_recvcounts = d_recvcounts;
    haloexchange->d_rdispls = d_rdispls;
    haloexchange->d_putdispls = d_putdispls;
    haloexchange->d_putranks = d_putranks;
    haloexchange->d_getranks = d_getranks;
    haloexchange->cudastream = stream;
    haloexchange->d_received = d_received;
    haloexchange->d_readytoreceive = d_readytoreceive;
    haloexchange->putdispls = putdispls;
    haloexchange->putranks = putranks;
    haloexchange->getranks = getranks;
    haloexchange->use_nvshmem = use_nvshmem;
    haloexchange->maxevents = maxevents;
    haloexchange->nevents = 0;
    haloexchange->cudaevents = events;
    return ACG_SUCCESS;
}
#endif

/**
 * ‘acghaloexchange_free()’ free resources associated with a halo
 * exchange.
 */
void acghaloexchange_free(
    struct acghaloexchange * haloexchange)
{
    free(haloexchange->sendbuf);
    free(haloexchange->recvbuf);
    free(haloexchange->sendreqs);
    free(haloexchange->recvreqs);
#if defined(ACG_HAVE_CUDA)
    if (haloexchange->use_nvshmem) {
        acgcomm_nvshmem_free(haloexchange->d_recvbuf);
        acgcomm_nvshmem_free(haloexchange->d_sendbuf);
        if (haloexchange->d_received)
            acgcomm_nvshmem_free(haloexchange->d_received);
        if (haloexchange->d_readytoreceive)
            acgcomm_nvshmem_free(haloexchange->d_readytoreceive);
    } else {
        cudaFree(haloexchange->d_recvbuf);
        cudaFree(haloexchange->d_sendbuf);
    }
    if (haloexchange->d_recvbufidx) cudaFree(haloexchange->d_recvbufidx);
    if (haloexchange->d_sendbufidx) cudaFree(haloexchange->d_sendbufidx);
    if (haloexchange->d_sendcounts) cudaFree(haloexchange->d_sendcounts);
    if (haloexchange->d_sdispls) cudaFree(haloexchange->d_sdispls);
    if (haloexchange->d_senders) cudaFree(haloexchange->d_senders);
    if (haloexchange->d_recvcounts) cudaFree(haloexchange->d_recvcounts);
    if (haloexchange->d_rdispls) cudaFree(haloexchange->d_rdispls);
    if (haloexchange->d_putdispls) cudaFree(haloexchange->d_putdispls);
    free(haloexchange->putdispls);
    if (haloexchange->d_putranks) cudaFree(haloexchange->d_putranks);
    free(haloexchange->putranks);
    if (haloexchange->d_getranks) cudaFree(haloexchange->d_getranks);
    free(haloexchange->getranks);
    if (haloexchange->cudaevents) {
        for (int i = 0; i < haloexchange->maxevents; i++) {
            cudaEventDestroy(haloexchange->cudaevents[i][0]);
            cudaEventDestroy(haloexchange->cudaevents[i][1]);
            cudaEventDestroy(haloexchange->cudaevents[i][2]);
            cudaEventDestroy(haloexchange->cudaevents[i][3]);
        }
        free(haloexchange->cudaevents);
    }
#elif defined(ACG_HAVE_HIP)
    if (haloexchange->use_rocshmem) {
        acgcomm_rocshmem_free(haloexchange->d_recvbuf);
        acgcomm_rocshmem_free(haloexchange->d_sendbuf);
        acgcomm_rocshmem_free(haloexchange->d_received);
        acgcomm_rocshmem_free(haloexchange->d_readytoreceive);
    } else {
        hipFree(haloexchange->d_recvbuf);
        hipFree(haloexchange->d_sendbuf);
    }
    if (haloexchange->d_recvbufidx) hipFree(haloexchange->d_recvbufidx);
    if (haloexchange->d_sendbufidx) hipFree(haloexchange->d_sendbufidx);
    free(haloexchange->putdispls);
    if (haloexchange->hipevents) {
        for (int i = 0; i < haloexchange->maxevents; i++) {
            hipEventDestroy(haloexchange->hipevents[i][0]);
            hipEventDestroy(haloexchange->hipevents[i][1]);
            hipEventDestroy(haloexchange->hipevents[i][2]);
            hipEventDestroy(haloexchange->hipevents[i][3]);
        }
        free(haloexchange->hipevents);
    }
#endif
}

/**
 * ‘acghaloexchange_profile()’ obtain detailed performance profiling
 * information for halo exchanges.
 */
int acghaloexchange_profile(
    const struct acghaloexchange * haloexchange,
    int maxevents,
    int * nevents,
    double * texchange,
    double * tpack,
    double * tsendrecv,
    double * tunpack)
{
    *nevents = 0;
#if defined(ACG_HAVE_CUDA)
    int N = haloexchange->nevents < haloexchange->maxevents ? haloexchange->nevents : haloexchange->maxevents;
    for (int i = 0; i < N && i < maxevents; i++, (*nevents)++) {
        int j = ((haloexchange->nevents-i-1) % haloexchange->maxevents + haloexchange->maxevents) % haloexchange->maxevents;
        cudaEvent_t (* events)[4] = &haloexchange->cudaevents[j];
        cudaEventSynchronize((*events)[0]);
        cudaEventSynchronize((*events)[1]);
        cudaEventSynchronize((*events)[2]);
        cudaEventSynchronize((*events)[3]);
        float t;
        cudaEventElapsedTime(&t, (*events)[0], (*events)[3]); texchange[i] = 1.0e-3*t;
        cudaEventElapsedTime(&t, (*events)[0], (*events)[1]); tpack[i] = 1.0e-3*t;
        cudaEventElapsedTime(&t, (*events)[1], (*events)[2]); tsendrecv[i] = 1.0e-3*t;
        cudaEventElapsedTime(&t, (*events)[2], (*events)[3]); tunpack[i] = 1.0e-3*t;
    }
#elif defined(ACG_HAVE_HIP)
    int N = haloexchange->nevents < haloexchange->maxevents ? haloexchange->nevents : haloexchange->maxevents;
    for (int i = 0; i < N && i < maxevents; i++, (*nevents)++) {
        int j = ((haloexchange->nevents-i-1) % haloexchange->maxevents + haloexchange->maxevents) % haloexchange->maxevents;
        hipEvent_t (* events)[4] = &haloexchange->hipevents[j];
        hipEventSynchronize((*events)[0]);
        hipEventSynchronize((*events)[1]);
        hipEventSynchronize((*events)[2]);
        hipEventSynchronize((*events)[3]);
        float t;
        hipEventElapsedTime(&t, (*events)[0], (*events)[3]); texchange[i] = 1.0e-3*t;
        hipEventElapsedTime(&t, (*events)[0], (*events)[1]); tpack[i] = 1.0e-3*t;
        hipEventElapsedTime(&t, (*events)[1], (*events)[2]); tsendrecv[i] = 1.0e-3*t;
        hipEventElapsedTime(&t, (*events)[2], (*events)[3]); tunpack[i] = 1.0e-3*t;
    }
#endif
    return ACG_SUCCESS;
}

#if defined(ACG_HAVE_NCCL)
/**
 * ‘halo_alltoallv_nccl()’ performs an neighbour all-to-all halo exchange.
 *
 * This assumes that messages have already been packed into a sending
 * buffer and will be unpacked from a receiving buffer afterwards.
 *
 * The array ‘recipients’, which is of length ‘nrecipients’, specifies
 * the processes to send messages to (i.e., recipients). Moreover, the
 * number of elements to send to each recipient is given by the array
 * ‘sendcounts’. For every recipient, a send is posted using
 * ‘ncclSend()’.
 *
 * On each process, ‘sendbuf’ is an array containing data to send to
 * neighbouring processes. More specifically, data sent to the process
 * with rank ‘recipients[p]’ must be stored contiguously in ‘sendbuf’,
 * starting at the index ‘sdispls[p]’. Thus, the length of ‘sendbuf’
 * must be at least equal to the maximum of ‘sdispls[p]+sendcounts[p]’
 * for any recieving neighbouring process ‘p’.
 *
 * The array ‘senders’ is of length ‘nsenders’, and specifies the
 * processes from which to receive messages. Moreover, the number of
 * elements to receive from each process is given by the array
 * ‘recvcounts’. For every sender, a receive is posted using
 * ‘ncclRecv()’.
 *
 * On each process, ‘recvbuf’ is a buffer used to receive data from
 * neighbouring processes. More specifically, data received from the
 * process with rank ‘senders[p]’ will be stored contiguously in
 * ‘recvbuf’, starting at the index ‘rdispls[p]’. Thus, the length of
 * ‘recvbuf’ must be at least equal to the maximum of
 * ‘rdispls[p]+recvcounts[p]’ for any sending neighbour ‘p’.
 */
static int halo_alltoallv_nccl(
    const void * sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    ncclDataType_t sendtype,
    void * recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    ncclDataType_t recvtype,
    ncclComm_t comm,
    cudaStream_t stream,
    int * ncclerrcode,
    int64_t * nsendmsgs,
    int64_t * nsendbytes,
    int64_t * nrecvmsgs,
    int64_t * nrecvbytes)
{
    int sendtypesize, recvtypesize;
    if (sendtype == ncclDouble) sendtypesize = sizeof(double);
    else return ACG_ERR_NOT_SUPPORTED;
    if (recvtype == ncclDouble) recvtypesize = sizeof(double);
    else return ACG_ERR_NOT_SUPPORTED;

    /* 1. post non-blocking message receives */
    int err = ncclGroupStart();
    if (err) { if (ncclerrcode) *ncclerrcode = err; return ACG_ERR_NCCL; }
    for (int p = 0; p < nsenders; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting ncclRecv of size %d from sender %d\n", __func__, recvcounts[p], senders[p]);
#endif
        void * recvbufp = (char *) recvbuf + recvtypesize*rdispls[p];
        err = ncclRecv(recvbufp, recvcounts[p], recvtype, senders[p], comm, stream);
        if (err) { if (ncclerrcode) *ncclerrcode = err; return ACG_ERR_NCCL; }
        if (nrecvbytes) *nrecvbytes += recvcounts[p]*recvtypesize;
    }
    if (nrecvmsgs) *nrecvmsgs += nsenders;

    /* 2. post non-blocking message sends */
    for (int p = 0; p < nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting ncclSend of size %d for recipient %d\n", __func__, sendcounts[p], recipients[p]);
#endif
        void * sendbufp = (char *) sendbuf + sendtypesize*sdispls[p];
        err = ncclSend(sendbufp, sendcounts[p], sendtype, recipients[p], comm, stream);
        if (err) { if (ncclerrcode) *ncclerrcode = err; return ACG_ERR_NCCL; }
        if (nsendbytes) *nsendbytes += sendcounts[p]*sendtypesize;
    }
    if (nsendmsgs) *nsendmsgs += nrecipients;
    err = ncclGroupEnd();
    if (err) { if (ncclerrcode) *ncclerrcode = err; return ACG_ERR_NCCL; }
    return ACG_SUCCESS;
}
#endif

#if defined(ACG_HAVE_CUDA)
/**
 * ‘acghalo_exchange_cuda()’ performs a halo exchange for data
 * residing on a CUDA device.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange_cuda(
    struct acghalo * halo,
    struct acghaloexchange * haloexchange,
    int srcbufsize,
    const void * d_srcbuf,
    enum acgdatatype sendtype,
    int dstbufsize,
    void * d_dstbuf,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    int tag,
    int * mpierrcode,
    int warmup)
{
    if (sendtype != haloexchange->sendtype) return ACG_ERR_INVALID_VALUE;
    if (recvtype != haloexchange->recvtype) return ACG_ERR_INVALID_VALUE;

    int err;
    void * sendreqs = haloexchange->sendreqs;
    void * recvreqs = haloexchange->recvreqs;
    int * putdispls = haloexchange->putdispls;
    void * d_sendbuf = haloexchange->d_sendbuf;
    void * d_recvbuf = haloexchange->d_recvbuf;
    void * d_sendbufidx = haloexchange->d_sendbufidx;
    void * d_recvbufidx = haloexchange->d_recvbufidx;
    cudaStream_t stream = haloexchange->cudastream;
    uint64_t * d_received = haloexchange->d_received;
    uint64_t * d_readytoreceive = haloexchange->d_readytoreceive;
    int eventidx = haloexchange->nevents % haloexchange->maxevents;
    cudaEvent_t (* events)[4] = &haloexchange->cudaevents[eventidx];

    /* 2. pack data for sending */
    /* if (!warmup) cudaEventRecord((*events)[0], stream); */
    err = acghalo_pack_cuda(
        halo->sendsize, d_sendbuf, sendtype,
        srcbufsize, d_srcbuf, d_sendbufidx, stream,
        &halo->Bpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) cudaEventRecord((*events)[1], stream); */
    if (!warmup) halo->npack++;

    /* 3. exchange messages */
    if (comm->type == acgcomm_mpi) {
        cudaDeviceSynchronize();
        err = halo_alltoallv_mpi(
            d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls,
            acgdatatype_mpi(sendtype), sendreqs,
            halo->recvsize, d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls,
            acgdatatype_mpi(recvtype), recvreqs,
            tag, comm->mpicomm, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL,
            true);
        if (err) return err;
    } else if (comm->type == acgcomm_nccl) {
#if defined(ACG_HAVE_NCCL)
        err = halo_alltoallv_nccl(
            d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls,
            acgdatatype_nccl(sendtype),
            d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls,
            acgdatatype_nccl(recvtype),
            comm->ncclcomm, stream, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL);
        if (err) return err;
#else
        return ACG_ERR_NCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nvshmem) {
#if defined(ACG_HAVE_NVSHMEM)
        err = halo_alltoallv_nvshmem(
            halo->sendsize, d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls, sendtype, putdispls,
            d_received, d_readytoreceive,
            halo->recvsize, d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls, recvtype,
            comm->mpicomm, stream, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL);
        if (err) return err;
#else
        return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }

    /* 4. unpack received data */
    /* if (!warmup) cudaEventRecord((*events)[2], stream); */
    err = acghalo_unpack_cuda(
        halo->recvsize, d_recvbuf, recvtype,
        dstbufsize, d_dstbuf, d_recvbufidx, stream,
        &halo->Bunpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) cudaEventRecord((*events)[3], stream); */
    if (!warmup) halo->nunpack++;
    if (!warmup) halo->nexchanges++;
    /* if (!warmup) haloexchange->nevents++; */
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_exchange_cuda_begin()’ starts a halo exchange for data
 * residing on a CUDA device.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange_cuda_begin(
    struct acghalo * halo,
    struct acghaloexchange * haloexchange,
    int srcbufsize,
    const void * d_srcbuf,
    enum acgdatatype sendtype,
    int dstbufsize,
    void * d_dstbuf,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    int tag,
    int * mpierrcode,
    int warmup,
    cudaStream_t stream)
{
    if (sendtype != haloexchange->sendtype) return ACG_ERR_INVALID_VALUE;
    if (recvtype != haloexchange->recvtype) return ACG_ERR_INVALID_VALUE;

    int err;
    void * sendreqs = haloexchange->sendreqs;
    void * recvreqs = haloexchange->recvreqs;
    int * putdispls = haloexchange->putdispls;
    void * d_sendbuf = haloexchange->d_sendbuf;
    void * d_recvbuf = haloexchange->d_recvbuf;
    void * d_sendbufidx = haloexchange->d_sendbufidx;
    void * d_recvbufidx = haloexchange->d_recvbufidx;
    /* cudaStream_t stream = haloexchange->cudastream; */
    uint64_t * d_received = haloexchange->d_received;
    uint64_t * d_readytoreceive = haloexchange->d_readytoreceive;
    int eventidx = haloexchange->nevents % haloexchange->maxevents;
    cudaEvent_t (* events)[4] = &haloexchange->cudaevents[eventidx];

    /* 2. pack data for sending */
    /* if (!warmup) cudaEventRecord((*events)[0], stream); */
    err = acghalo_pack_cuda(
        halo->sendsize, d_sendbuf, sendtype,
        srcbufsize, d_srcbuf, d_sendbufidx, stream,
        &halo->Bpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) cudaEventRecord((*events)[1], stream); */
    if (!warmup) halo->npack++;

    /* 3. exchange messages */
    if (comm->type == acgcomm_mpi) {
        cudaStreamSynchronize(stream);
        err = MPI_Startall(halo->nsenders, recvreqs);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        err = MPI_Startall(halo->nrecipients, sendreqs);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    } else if (comm->type == acgcomm_nccl) {
#if defined(ACG_HAVE_NCCL)
        err = halo_alltoallv_nccl(
            d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls,
            acgdatatype_nccl(sendtype),
            d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls,
            acgdatatype_nccl(recvtype),
            comm->ncclcomm, stream, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL);
        if (err) return err;
#else
        return ACG_ERR_NCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nvshmem) {
#if defined(ACG_HAVE_NVSHMEM)
        err = halo_alltoallv_nvshmem(
            halo->sendsize, d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls, sendtype, putdispls,
            d_received, d_readytoreceive,
            halo->recvsize, d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls, recvtype,
            comm->mpicomm, stream, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL);
        if (err) return err;
#else
        return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_exchange_cuda_end()’ completes a halo exchange for data
 * residing on a CUDA device.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange_cuda_end(
    struct acghalo * halo,
    struct acghaloexchange * haloexchange,
    int srcbufsize,
    const void * d_srcbuf,
    enum acgdatatype sendtype,
    int dstbufsize,
    void * d_dstbuf,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    int tag,
    int * mpierrcode,
    int warmup,
    cudaStream_t stream)
{
    if (sendtype != haloexchange->sendtype) return ACG_ERR_INVALID_VALUE;
    if (recvtype != haloexchange->recvtype) return ACG_ERR_INVALID_VALUE;

    int err;
    void * sendreqs = haloexchange->sendreqs;
    void * recvreqs = haloexchange->recvreqs;
    int * putdispls = haloexchange->putdispls;
    void * d_sendbuf = haloexchange->d_sendbuf;
    void * d_recvbuf = haloexchange->d_recvbuf;
    void * d_sendbufidx = haloexchange->d_sendbufidx;
    void * d_recvbufidx = haloexchange->d_recvbufidx;
    /* cudaStream_t stream = haloexchange->cudastream; */
    uint64_t * d_received = haloexchange->d_received;
    uint64_t * d_readytoreceive = haloexchange->d_readytoreceive;
    int eventidx = haloexchange->nevents % haloexchange->maxevents;
    cudaEvent_t (* events)[4] = &haloexchange->cudaevents[eventidx];

    /* 3. wait for send/recv to complete */
    if (comm->type == acgcomm_mpi) {
        MPI_Waitall(halo->nrecipients, sendreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(halo->nsenders, recvreqs, MPI_STATUSES_IGNORE);
        if (!warmup) {
            int sendtypesize, recvtypesize;
            err = acgdatatype_size(sendtype, &sendtypesize); if (err) return err;
            err = acgdatatype_size(recvtype, &recvtypesize); if (err) return err;
            halo->nmpisend += halo->nrecipients;
            halo->Bmpisend += halo->sendsize*sendtypesize;
            halo->nmpiirecv += halo->nsenders;
            halo->Bmpiirecv += halo->recvsize*recvtypesize;
        }
    } else if (comm->type == acgcomm_nccl) {
#if defined(ACG_HAVE_NCCL)
        /* do nothing */
#else
        return ACG_ERR_NCCL_NOT_SUPPORTED;
#endif
    } else if (comm->type == acgcomm_nvshmem) {
#if defined(ACG_HAVE_NVSHMEM)
        /* do nothing */
#else
        return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
    } else { return ACG_ERR_INVALID_VALUE; }

    /* 4. unpack received data */
    /* if (!warmup) cudaEventRecord((*events)[2], stream); */
    err = acghalo_unpack_cuda(
        halo->recvsize, d_recvbuf, recvtype,
        dstbufsize, d_dstbuf, d_recvbufidx, stream,
        &halo->Bunpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) cudaEventRecord((*events)[3], stream); */
    if (!warmup) halo->nunpack++;
    if (!warmup) halo->nexchanges++;
    /* if (!warmup) haloexchange->nevents++; */
    return ACG_SUCCESS;
}
#endif

/*
 * halo communication with HIP-aware MPI
 */

#if defined(ACG_HAVE_HIP)
/**
 * ‘acghaloexchange_init_hip()’ allocate additional storage needed
 * to perform a halo exchange for data residing on a HIP device.
 */
int acghaloexchange_init_hip(
    struct acghaloexchange * haloexchange,
    const struct acghalo * halo,
    enum acgdatatype sendtype,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    hipStream_t stream)
{
    int err = acghaloexchange_init(haloexchange, halo, sendtype, recvtype, comm);
    if (err) return err;

    /* allocate storage for device-side send/receive buffers */
    int sendtypesize, recvtypesize;
    err = acgdatatype_size(sendtype, &sendtypesize);
    if (err) return err;
    err = acgdatatype_size(recvtype, &recvtypesize);
    if (err) return err;
    void * d_sendbuf = NULL, * d_recvbuf = NULL;
    int use_rocshmem = comm->type == acgcomm_rocshmem;
    if (use_rocshmem) {
        int commsize, rank;
        MPI_Comm_size(comm->mpicomm, &commsize);
        MPI_Comm_rank(comm->mpicomm, &rank);
        int maxsendsize = halo->sendsize;
        MPI_Allreduce(MPI_IN_PLACE, &maxsendsize, 1, MPI_INT, MPI_MAX, comm->mpicomm);
        int maxrecvsize = halo->recvsize;
        MPI_Allreduce(MPI_IN_PLACE, &maxrecvsize, 1, MPI_INT, MPI_MAX, comm->mpicomm);
        int errcode;
        err = acgcomm_rocshmem_malloc(&d_sendbuf, maxsendsize*sendtypesize, &errcode);
        if (err) return err;
        err = acgcomm_rocshmem_malloc(&d_recvbuf, maxrecvsize*recvtypesize, &errcode);
        if (err) { acgcomm_rocshmem_free(d_sendbuf); return err; }
    } else {
        err = hipMalloc((void **) &d_sendbuf, halo->sendsize*sendtypesize);
        if (err) return ACG_ERR_HIP;
        err = hipMalloc((void **) &d_recvbuf, halo->recvsize*recvtypesize);
        if (err) { hipFree(d_sendbuf); return ACG_ERR_HIP; }
    }

    /* if rocSHMEM will be used, let each sender know the offset in the
    * receive buffer to use for its put operations */
    uint64_t * d_received = NULL, * d_readytoreceive = NULL;
    int * putdispls = NULL;
    if (use_rocshmem) {
        putdispls = malloc(halo->nsenders*sizeof(*putdispls));
        if (!putdispls) return ACG_ERR_ERRNO;
        for (int i = 0; i < halo->nsenders; i++) putdispls[i] = 0;
        int tag = 1;
        for (int p = 0; p < halo->nrecipients; p++) {
            err = MPI_Isend(
                &halo->rdispls[p], 1, MPI_INT, halo->recipients[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->sendreqs)[p]);
        }
        for (int p = 0; p < halo->nsenders; p++) {
            err = MPI_Irecv(
                &putdispls[p], 1, MPI_INT, halo->senders[p],
                tag, comm->mpicomm, &((MPI_Request *) haloexchange->recvreqs)[p]);
        }
        MPI_Waitall(halo->nrecipients, haloexchange->sendreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(halo->nsenders, haloexchange->recvreqs, MPI_STATUSES_IGNORE);

        /* allocate device-side storage for signal and reset stream */
        err = acgcomm_rocshmem_calloc((void **) &d_received, 1, sizeof(*d_received), NULL);
        if (err) return err;
        err = acgcomm_rocshmem_calloc((void **) &d_readytoreceive, 1, sizeof(*d_readytoreceive), NULL);
        if (err) return err;
    }

    /* copy buffers needed for packing/unpacking to device */
    void * d_sendbufidx, * d_recvbufidx;
    err = hipMalloc((void **) &d_sendbufidx, halo->sendsize*sizeof(*halo->sendbufidx));
    if (err) {
        if (use_rocshmem) { acgcomm_rocshmem_free(d_recvbuf); acgcomm_rocshmem_free(d_sendbuf); }
        else { hipFree(d_recvbuf); hipFree(d_sendbuf); }
        return ACG_ERR_HIP;
    }
    err = hipMemcpy(d_sendbufidx, halo->sendbufidx, halo->sendsize*sizeof(*halo->sendbufidx), hipMemcpyHostToDevice);
    if (err) {
        hipFree(d_sendbufidx);
        if (use_rocshmem) { acgcomm_rocshmem_free(d_recvbuf); acgcomm_rocshmem_free(d_sendbuf); }
        else { hipFree(d_recvbuf); hipFree(d_sendbuf); }
        return ACG_ERR_HIP;
    }
    err = hipMalloc((void **) &d_recvbufidx, halo->recvsize*sizeof(*halo->recvbufidx));
    if (err) {
        hipFree(d_sendbufidx);
        if (use_rocshmem) { acgcomm_rocshmem_free(d_recvbuf); acgcomm_rocshmem_free(d_sendbuf); }
        else { hipFree(d_recvbuf); hipFree(d_sendbuf); }
        return ACG_ERR_HIP;
    }
    err = hipMemcpy(d_recvbufidx, halo->recvbufidx, halo->recvsize*sizeof(*halo->recvbufidx), hipMemcpyHostToDevice);
    if (err) {
        hipFree(d_recvbufidx); hipFree(d_sendbufidx);
        if (use_rocshmem) { acgcomm_rocshmem_free(d_recvbuf); acgcomm_rocshmem_free(d_sendbuf); }
        else { hipFree(d_recvbuf); hipFree(d_sendbuf); }
        return ACG_ERR_HIP;
    }

    /* allocate storage for performance monitoring events */
    int maxevents = ACG_HALO_MAX_PERF_EVENTS;
    hipEvent_t (* events)[4] = malloc(maxevents*sizeof(*events));
    if (!events) {
        hipFree(d_recvbufidx); hipFree(d_sendbufidx);
        if (use_rocshmem) { acgcomm_rocshmem_free(d_recvbuf); acgcomm_rocshmem_free(d_sendbuf); }
        else { hipFree(d_recvbuf); hipFree(d_sendbuf); }
        return ACG_ERR_ERRNO;
    }
    for (int i = 0; i < maxevents; i++) {
        hipEventCreate(&events[i][0]);
        hipEventCreate(&events[i][1]);
        hipEventCreate(&events[i][2]);
        hipEventCreate(&events[i][3]);
    }

    /* set up persistent communications */
    if (comm->type == acgcomm_mpi) {
        MPI_Comm mpicomm = comm->mpicomm;
        int rank;
        MPI_Comm_rank(mpicomm, &rank);
        int tag = 99;
        for (int p = 0; p < halo->nsenders; p++) {
#if defined(ACG_DEBUG_HALO)
            fprintf(stderr, "%s: posting MPI_Irecv %d of %d for rank %d of size %d at offset %d from sender %d with tag %d\n", __func__, p+1, halo->nsenders, rank, halo->recvcounts[p], halo->rdispls[p], halo->senders[p], tag);
#endif
            void * recvbufp = (char *) d_recvbuf + recvtypesize*halo->rdispls[p];
            err = MPI_Recv_init(
                recvbufp, halo->recvcounts[p], acgdatatype_mpi(recvtype), halo->senders[p],
                tag, mpicomm, &((MPI_Request *) haloexchange->recvreqs)[p]);
            if (err) return ACG_ERR_MPI;
        }
        for (int p = 0; p < halo->nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
            fprintf(stderr, "%s: posting MPI_Isend %d of %d from rank %d of size %d at offset %d for recipient %d with tag %d\n", __func__, p+1, halo->nrecipients, rank, halo->sendcounts[p], halo->sdispls[p], halo->recipients[p], tag);
#endif
            void * sendbufp = (char *) d_sendbuf + sendtypesize*halo->sdispls[p];
            err = MPI_Send_init(
                sendbufp, halo->sendcounts[p], acgdatatype_mpi(sendtype), halo->recipients[p],
                tag, mpicomm, &((MPI_Request *) haloexchange->sendreqs)[p]);
            if (err) return ACG_ERR_MPI;
        }
    }

    haloexchange->d_sendbuf = d_sendbuf;
    haloexchange->d_recvbuf = d_recvbuf;
    haloexchange->d_sendbufidx = d_sendbufidx;
    haloexchange->d_recvbufidx = d_recvbufidx;
    haloexchange->hipstream = stream;
    haloexchange->d_received = d_received;
    haloexchange->d_readytoreceive = d_readytoreceive;
    haloexchange->putdispls = putdispls;
    haloexchange->putranks = NULL;
    haloexchange->getranks = NULL;
    haloexchange->use_rocshmem = use_rocshmem;
    haloexchange->maxevents = maxevents;
    haloexchange->nevents = 0;
    haloexchange->hipevents = events;
    return ACG_SUCCESS;
}
#endif

#if defined(ACG_HAVE_RCCL)
/**
 * ‘halo_alltoallv_rccl()’ performs an neighbour all-to-all halo exchange.
 *
 * This assumes that messages have already been packed into a sending
 * buffer and will be unpacked from a receiving buffer afterwards.
 *
 * The array ‘recipients’, which is of length ‘nrecipients’, specifies
 * the processes to send messages to (i.e., recipients). Moreover, the
 * number of elements to send to each recipient is given by the array
 * ‘sendcounts’. For every recipient, a send is posted using
 * ‘ncclSend()’.
 *
 * On each process, ‘sendbuf’ is an array containing data to send to
 * neighbouring processes. More specifically, data sent to the process
 * with rank ‘recipients[p]’ must be stored contiguously in ‘sendbuf’,
 * starting at the index ‘sdispls[p]’. Thus, the length of ‘sendbuf’
 * must be at least equal to the maximum of ‘sdispls[p]+sendcounts[p]’
 * for any recieving neighbouring process ‘p’.
 *
 * The array ‘senders’ is of length ‘nsenders’, and specifies the
 * processes from which to receive messages. Moreover, the number of
 * elements to receive from each process is given by the array
 * ‘recvcounts’. For every sender, a receive is posted using
 * ‘ncclRecv()’.
 *
 * On each process, ‘recvbuf’ is a buffer used to receive data from
 * neighbouring processes. More specifically, data received from the
 * process with rank ‘senders[p]’ will be stored contiguously in
 * ‘recvbuf’, starting at the index ‘rdispls[p]’. Thus, the length of
 * ‘recvbuf’ must be at least equal to the maximum of
 * ‘rdispls[p]+recvcounts[p]’ for any sending neighbour ‘p’.
 */
static int halo_alltoallv_rccl(
    const void * sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    ncclDataType_t sendtype,
    void * recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    ncclDataType_t recvtype,
    ncclComm_t comm,
    hipStream_t stream,
    int * rcclerrcode,
    int64_t * nsendmsgs,
    int64_t * nsendbytes,
    int64_t * nrecvmsgs,
    int64_t * nrecvbytes)
{
    int sendtypesize, recvtypesize;
    if (sendtype == ncclDouble) sendtypesize = sizeof(double);
    else return ACG_ERR_NOT_SUPPORTED;
    if (recvtype == ncclDouble) recvtypesize = sizeof(double);
    else return ACG_ERR_NOT_SUPPORTED;

    /* 1. post non-blocking message receives */
    int err = ncclGroupStart();
    if (err) { if (rcclerrcode) *rcclerrcode = err; return ACG_ERR_RCCL; }
    for (int p = 0; p < nsenders; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting ncclRecv of size %d from sender %d\n", __func__, recvcounts[p], senders[p]);
#endif
        void * recvbufp = (char *) recvbuf + recvtypesize*rdispls[p];
        err = ncclRecv(recvbufp, recvcounts[p], recvtype, senders[p], comm, stream);
        if (err) { if (rcclerrcode) *rcclerrcode = err; return ACG_ERR_RCCL; }
        if (nrecvbytes) *nrecvbytes += recvcounts[p]*recvtypesize;
    }
    if (nrecvmsgs) *nrecvmsgs += nsenders;

    /* 2. post non-blocking message sends */
    for (int p = 0; p < nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting ncclSend of size %d for recipient %d\n", __func__, sendcounts[p], recipients[p]);
#endif
        void * sendbufp = (char *) sendbuf + sendtypesize*sdispls[p];
        err = ncclSend(sendbufp, sendcounts[p], sendtype, recipients[p], comm, stream);
        if (err) { if (rcclerrcode) *rcclerrcode = err; return ACG_ERR_RCCL; }
        if (nsendbytes) *nsendbytes += sendcounts[p]*sendtypesize;
    }
    if (nsendmsgs) *nsendmsgs += nrecipients;
    err = ncclGroupEnd();
    if (err) { if (rcclerrcode) *rcclerrcode = err; return ACG_ERR_RCCL; }
    return ACG_SUCCESS;
}
#endif

#if defined(ACG_HAVE_HIP)
/**
 * ‘acghalo_exchange_hip()’ performs a halo exchange for data
 * residing on a HIP device.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange_hip(
    struct acghalo * halo,
    struct acghaloexchange * haloexchange,
    int srcbufsize,
    const void * d_srcbuf,
    enum acgdatatype sendtype,
    int dstbufsize,
    void * d_dstbuf,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    int tag,
    int * mpierrcode,
    int warmup)
{
    if (sendtype != haloexchange->sendtype) return ACG_ERR_INVALID_VALUE;
    if (recvtype != haloexchange->recvtype) return ACG_ERR_INVALID_VALUE;

    int err;
    void * sendreqs = haloexchange->sendreqs;
    void * recvreqs = haloexchange->recvreqs;
    int * putdispls = haloexchange->putdispls;
    void * d_sendbuf = haloexchange->d_sendbuf;
    void * d_recvbuf = haloexchange->d_recvbuf;
    void * d_sendbufidx = haloexchange->d_sendbufidx;
    void * d_recvbufidx = haloexchange->d_recvbufidx;
    hipStream_t stream = haloexchange->hipstream;
    /* uint64_t * d_sigaddr = haloexchange->d_sigaddr; */
    /* int eventidx = haloexchange->nevents % haloexchange->maxevents; */
    /* hipEvent_t (* events)[4] = &haloexchange->hipevents[eventidx]; */

    /* 2. pack data for sending */
    /* if (!warmup) hipEventRecord((*events)[0], stream); */
    err = acghalo_pack_hip(
        halo->sendsize, d_sendbuf, sendtype,
        srcbufsize, d_srcbuf, d_sendbufidx, stream,
        &halo->Bpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) hipEventRecord((*events)[1], stream); */
    if (!warmup) halo->npack++;

    /* 3. exchange messages */
    if (comm->type == acgcomm_mpi) {
        hipDeviceSynchronize();
        err = halo_alltoallv_mpi(
            d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls,
            acgdatatype_mpi(sendtype), sendreqs,
            halo->recvsize, d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls,
            acgdatatype_mpi(recvtype), recvreqs,
            tag, comm->mpicomm, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL,
            true);
        if (err) return err;
    } else if (comm->type == acgcomm_rccl) {
#if defined(ACG_HAVE_RCCL)
        err = halo_alltoallv_rccl(
            d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls,
            acgdatatype_nccl(sendtype),
            d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls,
            acgdatatype_nccl(recvtype),
            comm->ncclcomm, stream, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL);
        if (err) return err;
#else
        return ACG_ERR_RCCL_NOT_SUPPORTED;
#endif
/*     } else if (comm->type == acgcomm_rocshmem) { */
/* #if defined(ACG_HAVE_ROCSHMEM) */
/*         err = halo_alltoallv_rocshmem( */
/*             halo->sendsize, d_sendbuf, halo->nrecipients, halo->recipients, */
/*             halo->sendcounts, halo->sdispls, sendtype, putdispls, d_sigaddr, */
/*             halo->recvsize, d_recvbuf, halo->nsenders, halo->senders, */
/*             halo->recvcounts, halo->rdispls, recvtype, */
/*             comm->mpicomm, stream, mpierrcode, */
/*             !warmup ? &halo->nmpisend : NULL, */
/*             !warmup ? &halo->Bmpisend : NULL, */
/*             !warmup ? &halo->nmpiirecv : NULL, */
/*             !warmup ? &halo->Bmpiirecv : NULL); */
/*         if (err) return err; */
/* #else */
/*         return ACG_ERR_ROCSHMEM_NOT_SUPPORTED; */
/* #endif */
    } else { return ACG_ERR_INVALID_VALUE; }

    /* 4. unpack received data */
    /* if (!warmup) hipEventRecord((*events)[2], stream); */
    err = acghalo_unpack_hip(
        halo->recvsize, d_recvbuf, recvtype,
        dstbufsize, d_dstbuf, d_recvbufidx, stream,
        &halo->Bunpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) hipEventRecord((*events)[3], stream); */
    if (!warmup) halo->nunpack++;
    if (!warmup) halo->nexchanges++;
    /* if (!warmup) haloexchange->nevents++; */
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_exchange_hip_begin()’ starts a halo exchange for data
 * residing on a HIP device.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange_hip_begin(
    struct acghalo * halo,
    struct acghaloexchange * haloexchange,
    int srcbufsize,
    const void * d_srcbuf,
    enum acgdatatype sendtype,
    int dstbufsize,
    void * d_dstbuf,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    int tag,
    int * mpierrcode,
    int warmup,
    hipStream_t stream)
{
    if (sendtype != haloexchange->sendtype) return ACG_ERR_INVALID_VALUE;
    if (recvtype != haloexchange->recvtype) return ACG_ERR_INVALID_VALUE;

    int err;
    void * sendreqs = haloexchange->sendreqs;
    void * recvreqs = haloexchange->recvreqs;
    int * putdispls = haloexchange->putdispls;
    void * d_sendbuf = haloexchange->d_sendbuf;
    void * d_recvbuf = haloexchange->d_recvbuf;
    void * d_sendbufidx = haloexchange->d_sendbufidx;
    void * d_recvbufidx = haloexchange->d_recvbufidx;
    /* hipStream_t stream = haloexchange->hipstream; */
    /* uint64_t * d_sigaddr = haloexchange->d_sigaddr; */
    /* int eventidx = haloexchange->nevents % haloexchange->maxevents; */
    /* hipEvent_t (* events)[4] = &haloexchange->hipevents[eventidx]; */

    /* 2. pack data for sending */
    /* if (!warmup) hipEventRecord((*events)[0], stream); */
    err = acghalo_pack_hip(
        halo->sendsize, d_sendbuf, sendtype,
        srcbufsize, d_srcbuf, d_sendbufidx, stream,
        &halo->Bpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) hipEventRecord((*events)[1], stream); */
    if (!warmup) halo->npack++;

    /* 3. exchange messages */
    if (comm->type == acgcomm_mpi) {
        hipStreamSynchronize(stream);
        err = MPI_Startall(halo->nsenders, recvreqs);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
        err = MPI_Startall(halo->nrecipients, sendreqs);
        if (err) { if (mpierrcode) *mpierrcode = err; return ACG_ERR_MPI; }
    } else if (comm->type == acgcomm_rccl) {
#if defined(ACG_HAVE_RCCL)
        err = halo_alltoallv_rccl(
            d_sendbuf, halo->nrecipients, halo->recipients,
            halo->sendcounts, halo->sdispls,
            acgdatatype_nccl(sendtype),
            d_recvbuf, halo->nsenders, halo->senders,
            halo->recvcounts, halo->rdispls,
            acgdatatype_nccl(recvtype),
            comm->ncclcomm, stream, mpierrcode,
            !warmup ? &halo->nmpisend : NULL,
            !warmup ? &halo->Bmpisend : NULL,
            !warmup ? &halo->nmpiirecv : NULL,
            !warmup ? &halo->Bmpiirecv : NULL);
        if (err) return err;
#else
        return ACG_ERR_RCCL_NOT_SUPPORTED;
#endif
/*     } else if (comm->type == acgcomm_rocshmem) { */
/* #if defined(ACG_HAVE_ROCSHMEM) */
/*         err = halo_alltoallv_rocshmem( */
/*             halo->sendsize, d_sendbuf, halo->nrecipients, halo->recipients, */
/*             halo->sendcounts, halo->sdispls, sendtype, putdispls, d_sigaddr, */
/*             halo->recvsize, d_recvbuf, halo->nsenders, halo->senders, */
/*             halo->recvcounts, halo->rdispls, recvtype, */
/*             comm->mpicomm, stream, mpierrcode, */
/*             !warmup ? &halo->nmpisend : NULL, */
/*             !warmup ? &halo->Bmpisend : NULL, */
/*             !warmup ? &halo->nmpiirecv : NULL, */
/*             !warmup ? &halo->Bmpiirecv : NULL); */
/*         if (err) return err; */
/* #else */
/*         return ACG_ERR_ROCSHMEM_NOT_SUPPORTED; */
/* #endif */
    } else { return ACG_ERR_INVALID_VALUE; }
    return ACG_SUCCESS;
}

/**
 * ‘acghalo_exchange_hip_end()’ completes a halo exchange for data
 * residing on a HIP device.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
int acghalo_exchange_hip_end(
    struct acghalo * halo,
    struct acghaloexchange * haloexchange,
    int srcbufsize,
    const void * d_srcbuf,
    enum acgdatatype sendtype,
    int dstbufsize,
    void * d_dstbuf,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    int tag,
    int * mpierrcode,
    int warmup,
    hipStream_t stream)
{
    if (sendtype != haloexchange->sendtype) return ACG_ERR_INVALID_VALUE;
    if (recvtype != haloexchange->recvtype) return ACG_ERR_INVALID_VALUE;

    int err;
    void * sendreqs = haloexchange->sendreqs;
    void * recvreqs = haloexchange->recvreqs;
    int * putdispls = haloexchange->putdispls;
    void * d_sendbuf = haloexchange->d_sendbuf;
    void * d_recvbuf = haloexchange->d_recvbuf;
    void * d_sendbufidx = haloexchange->d_sendbufidx;
    void * d_recvbufidx = haloexchange->d_recvbufidx;
    /* hipStream_t stream = haloexchange->hipstream; */
    /* uint64_t * d_sigaddr = haloexchange->d_sigaddr; */
    /* int eventidx = haloexchange->nevents % haloexchange->maxevents; */
    /* hipEvent_t (* events)[4] = &haloexchange->hipevents[eventidx]; */

    /* 3. wait for send/recv to complete */
    if (comm->type == acgcomm_mpi) {
        MPI_Waitall(halo->nrecipients, sendreqs, MPI_STATUSES_IGNORE);
        MPI_Waitall(halo->nsenders, recvreqs, MPI_STATUSES_IGNORE);
        if (!warmup) {
            int sendtypesize, recvtypesize;
            err = acgdatatype_size(sendtype, &sendtypesize); if (err) return err;
            err = acgdatatype_size(recvtype, &recvtypesize); if (err) return err;
            halo->nmpisend += halo->nrecipients;
            halo->Bmpisend += halo->sendsize*sendtypesize;
            halo->nmpiirecv += halo->nsenders;
            halo->Bmpiirecv += halo->recvsize*recvtypesize;
        }
    } else if (comm->type == acgcomm_rccl) {
#if defined(ACG_HAVE_RCCL)
        /* do nothing */
#else
        return ACG_ERR_RCCL_NOT_SUPPORTED;
#endif
/*     } else if (comm->type == acgcomm_rocshmem) { */
/* #if defined(ACG_HAVE_ROCSHMEM) */
/*         /\* do nothing *\/ */
/* #else */
/*         return ACG_ERR_ROCSHMEM_NOT_SUPPORTED; */
/* #endif */
    } else { return ACG_ERR_INVALID_VALUE; }

    /* 4. unpack received data */
    /* if (!warmup) hipEventRecord((*events)[2], stream); */
    err = acghalo_unpack_hip(
        halo->recvsize, d_recvbuf, recvtype,
        dstbufsize, d_dstbuf, d_recvbufidx, stream,
        &halo->Bunpack, mpierrcode);
    if (err) return err;
    /* if (!warmup) hipEventRecord((*events)[3], stream); */
    if (!warmup) halo->nunpack++;
    if (!warmup) halo->nexchanges++;
    /* if (!warmup) haloexchange->nevents++; */
    return ACG_SUCCESS;
}
#endif
