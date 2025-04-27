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
 * CUDA kernels for halo exchange
 */

#include "acg/config.h"
#include "acg/halo.h"
#include "acg/error.h"

#if defined(ACG_HAVE_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

__global__ void acghalo_pack_cuda_double(
    int sendbufsize,
    double * sendbuf,
    int srcbufsize,
    const double * srcbuf,
    const int * srcbufidx)
{
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < sendbufsize;
         i += blockDim.x * gridDim.x)
    {
        sendbuf[i] = srcbuf[srcbufidx[i]];
    }
}

/**
 * ‘acghalo_pack_cuda()’ packs messages for sending in a halo exchange.
 *
 * Data is copied from the array ‘srcbuf’, which is of length
 * ‘srcbufsize’ and contains elements of the type specified by
 * ‘datatype’, to the ‘sendbuf’ array, which must be of length
 * ‘sendbufsize’. The number of elements to be copied is given by
 * ‘sendbufsize’, and the ‘i’th element in ‘sendbuf’ is copied from
 * the position ‘srcbufidx[i]’ in ‘srcbuf’.
 *
 * The arrays ‘sendbuf’ and ‘srcbuf’ may not overlap.
 */
int acghalo_pack_cuda(
    int sendbufsize,
    void * d_sendbuf,
    enum acgdatatype datatype,
    int srcbufsize,
    const void * d_srcbuf,
    const int * d_srcbufidx,
    cudaStream_t stream,
    int64_t * nbytes,
    int * errcode)
{
    if (datatype == ACG_DOUBLE) {
        static int mingridsize = 0, blocksize = 0;
        if (mingridsize == 0 && blocksize == 0) {
            cudaOccupancyMaxPotentialBlockSize(
                &mingridsize, &blocksize, acghalo_pack_cuda_double, 0, 0);
        }
        acghalo_pack_cuda_double<<<mingridsize,blocksize,0,stream>>>(
            sendbufsize, (double *) d_sendbuf,
            srcbufsize, (const double *) d_srcbuf, d_srcbufidx);
        if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
        if (nbytes) *nbytes += sendbufsize*(2*sizeof(double)+sizeof(*d_srcbufidx));
    } else { return ACG_ERR_NOT_SUPPORTED; }
    return ACG_SUCCESS;
}

__global__ void acghalo_unpack_cuda_double(
    int recvbufsize,
    const double * recvbuf,
    int dstbufsize,
    double * dstbuf,
    const int * dstbufidx)
{
    for (int i = blockIdx.x*blockDim.x+threadIdx.x;
         i < recvbufsize;
         i += blockDim.x * gridDim.x)
    {
        dstbuf[dstbufidx[i]] = recvbuf[i];
    }
}

/**
 * ‘acghalo_unpack_cuda()’ unpacks messages received in a halo exchange.
 *
 * Data is copied to the array ‘dstbuf’, which is of length
 * ‘dstbufsize’ and contains elements of the type specified by
 * ‘datatype’, from the ‘recvbuf’ array, which must be of length
 * ‘recvbufsize’. The number of elements to be copied is given by
 * ‘recvbufsize’, and the ‘i’th element in ‘recvbuf’ is copied to the
 * position ‘dstbufidx[i]’ in ‘dstbuf’.
 *
 * The arrays ‘dstbuf’ and ‘recvbuf’ may not overlap.
 */
int acghalo_unpack_cuda(
    int recvbufsize,
    const void * d_recvbuf,
    enum acgdatatype datatype,
    int dstbufsize,
    void * d_dstbuf,
    const int * d_dstbufidx,
    cudaStream_t stream,
    int64_t * nbytes,
    int * errcode)
{
    if (datatype == ACG_DOUBLE) {
        static int mingridsize = 0, blocksize = 0;
        if (mingridsize == 0 && blocksize == 0) {
            cudaOccupancyMaxPotentialBlockSize(
                &mingridsize, &blocksize, acghalo_unpack_cuda_double, 0, 0);
        }
        acghalo_unpack_cuda_double<<<mingridsize,blocksize,0,stream>>>(
            recvbufsize, (const double *) d_recvbuf,
            dstbufsize, (double *) d_dstbuf, d_dstbufidx);
        if (cudaPeekAtLastError()) return ACG_ERR_CUDA;
        if (nbytes) *nbytes += recvbufsize*(2*sizeof(double)+sizeof(*d_dstbufidx));
    } else { return ACG_ERR_NOT_SUPPORTED; }
    return ACG_SUCCESS;
}

/*
 * halo communication with NVSHMEM
 */

/**
 * ‘halo_alltoallv_nvshmem()’ performs an neighbour all-to-all halo exchange.
 *
 * This assumes that messages have already been packed into a sending
 * buffer and will be unpacked from a receiving buffer afterwards.
 *
 * The array ‘recipients’, which is of length ‘nrecipients’, specifies
 * the processes to send messages to (i.e., recipients). Moreover, the
 * number of elements to send to each recipient is given by the array
 * ‘sendcounts’.
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
 * ‘recvcounts’.
 *
 * On each process, ‘recvbuf’ is a buffer used to receive data from
 * neighbouring processes. More specifically, data received from the
 * process with rank ‘senders[p]’ will be stored contiguously in
 * ‘recvbuf’, starting at the index ‘rdispls[p]’. Thus, the length of
 * ‘recvbuf’ must be at least equal to the maximum of
 * ‘rdispls[p]+recvcounts[p]’ for any sending neighbour ‘p’.
 */
int halo_alltoallv_nvshmem(
    int sendsize,
    const void * d_sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    acgdatatype sendtype,
    const int * putdispls,
    uint64_t * d_received,
    uint64_t * d_readytoreceive,
    int recvsize,
    void * d_recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    acgdatatype recvtype,
    MPI_Comm comm,
    cudaStream_t stream,
    int * errcode,
    int64_t * nsendmsgs,
    int64_t * nsendbytes,
    int64_t * nrecvmsgs,
    int64_t * nrecvbytes)
{
#if defined(ACG_HAVE_NVSHMEM)
    int commsize, rank;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);
    int err;
    int sendtypesize, recvtypesize;
    if (sendtype == ACG_DOUBLE) sendtypesize = sizeof(double);
    else return ACG_ERR_NOT_SUPPORTED;
    if (recvtype == ACG_DOUBLE) recvtypesize = sizeof(double);
    else return ACG_ERR_NOT_SUPPORTED;

    /* post PUT operations */
    nvshmemx_sync_all_on_stream(stream);
    for (int p = 0; p < nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting PUT %d of %d from rank %d of size %d at offset %d for recipient %d at offset %d\n", __func__, p+1, nrecipients, nvshmem_my_pe() /* rank */, sendcounts[p], sdispls[p], recipients[p], putdispls[p]);
#endif
        const double * d_sendbufp = (const double *) d_sendbuf + sdispls[p];
        double * d_recvbufp = (double *) d_recvbuf + putdispls[p];
        /* nvshmemx_double_put_on_stream(d_recvbufp, d_sendbufp, sendcounts[p], recipients[p], stream); */
        nvshmemx_double_put_signal_on_stream(
            d_recvbufp, d_sendbufp, sendcounts[p],
            d_received, 1, NVSHMEM_SIGNAL_ADD,
            recipients[p], stream);
        if (nsendbytes) *nsendbytes += sendcounts[p]*sendtypesize;
    }
    /* nvshmemx_barrier_all_on_stream(stream); */
    nvshmemx_signal_wait_until_on_stream(
        d_received, NVSHMEM_CMP_GE, nsenders, stream);
    cudaMemsetAsync(d_received, 0, sizeof(*d_received), stream);
    if (nsendmsgs) *nsendmsgs += nrecipients;
    return ACG_SUCCESS;
#else
    return ACG_ERR_NVSHMEM_NOT_SUPPORTED;
#endif
}
