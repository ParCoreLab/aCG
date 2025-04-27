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
 * HIP kernels for halo exchange
 */

#include "acg/config.h"
#include "acg/halo.h"
#include "acg/error.h"

#include <hip/hip_runtime.h>

#if defined(ACG_HAVE_ROCSHMEM)
#include <rocshmem.h>
#include <rocshmemx.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

__launch_bounds__(256)
__global__ void acghalo_pack_hip_double(
    int sendbufsize,
    double * __restrict__ sendbuf,
    int srcbufsize,
    const double * __restrict__ srcbuf,
    const int * __restrict__ srcbufidx)
{
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < sendbufsize;
    //      i += blockDim.x * gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < sendbufsize)
    {
        sendbuf[i] = srcbuf[srcbufidx[i]];
    }
}

/**
 * ‘acghalo_pack_hip()’ packs messages for sending in a halo exchange.
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
int acghalo_pack_hip(
    int sendbufsize,
    void * d_sendbuf,
    enum acgdatatype datatype,
    int srcbufsize,
    const void * d_srcbuf,
    const int * d_srcbufidx,
    hipStream_t stream,
    int64_t * nbytes,
    int * errcode)
{
    if (datatype == ACG_DOUBLE) {
        // static int mingridsize = 0, blocksize = 0;
        // if (mingridsize == 0 && blocksize == 0) {
        //     hipOccupancyMaxPotentialBlockSize(
        //         &mingridsize, &blocksize, acghalo_pack_hip_double, 0, 0);
        // }
        int mingridsize = (sendbufsize + 255) / 256, blocksize = 256;
        acghalo_pack_hip_double<<<mingridsize,blocksize,0,stream>>>(
            sendbufsize, (double *) d_sendbuf,
            srcbufsize, (const double *) d_srcbuf, d_srcbufidx);
        if (hipPeekAtLastError()) return ACG_ERR_HIP;
        if (nbytes) *nbytes += sendbufsize*(2*sizeof(double)+sizeof(*d_srcbufidx));
    } else { return ACG_ERR_NOT_SUPPORTED; }
    return ACG_SUCCESS;
}

__launch_bounds__(256)
__global__ void acghalo_unpack_hip_double(
    int recvbufsize,
    const double * __restrict__ recvbuf,
    int dstbufsize,
    double * __restrict__ dstbuf,
    const int * __restrict__ dstbufidx)
{
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < recvbufsize;
    //      i += blockDim.x * gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < recvbufsize)
    {
        dstbuf[dstbufidx[i]] = recvbuf[i];
    }
}

/**
 * ‘acghalo_unpack_hip()’ unpacks messages received in a halo exchange.
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
int acghalo_unpack_hip(
    int recvbufsize,
    const void * d_recvbuf,
    enum acgdatatype datatype,
    int dstbufsize,
    void * d_dstbuf,
    const int * d_dstbufidx,
    hipStream_t stream,
    int64_t * nbytes,
    int * errcode)
{
    if (datatype == ACG_DOUBLE) {
        // static int mingridsize = 0, blocksize = 0;
        // if (mingridsize == 0 && blocksize == 0) {
        //     hipOccupancyMaxPotentialBlockSize(
        //         &mingridsize, &blocksize, acghalo_unpack_hip_double, 0, 0);
        // }
        int mingridsize = (recvbufsize + 255) / 256, blocksize = 256;
        acghalo_unpack_hip_double<<<mingridsize,blocksize,0,stream>>>(
            recvbufsize, (const double *) d_recvbuf,
            dstbufsize, (double *) d_dstbuf, d_dstbufidx);
        if (hipPeekAtLastError()) return ACG_ERR_HIP;
        if (nbytes) *nbytes += recvbufsize*(2*sizeof(double)+sizeof(*d_dstbufidx));
    } else { return ACG_ERR_NOT_SUPPORTED; }
    return ACG_SUCCESS;
}

/*
 * halo communication with ROCSHMEM
 */

#if defined(ACG_HAVE_MPI)
/**
 * ‘halo_alltoallv_rocshmem()’ performs an neighbour all-to-all halo exchange.
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
int halo_alltoallv_rocshmem(
    int sendsize,
    const void * d_sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    acgdatatype sendtype,
    const int * putdispls,
    uint64_t * d_sigaddr,
    int recvsize,
    void * d_recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    acgdatatype recvtype,
    MPI_Comm comm,
    hipStream_t stream,
    int * errcode,
    int64_t * nsendmsgs,
    int64_t * nsendbytes,
    int64_t * nrecvmsgs,
    int64_t * nrecvbytes)
{
#if defined(ACG_HAVE_ROCSHMEM)
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
    rocshmemx_sync_all_on_stream(stream);
    for (int p = 0; p < nrecipients; p++) {
#if defined(ACG_DEBUG_HALO)
        fprintf(stderr, "%s: posting PUT %d of %d from rank %d of size %d at offset %d for recipient %d at offset %d\n", __func__, p+1, nrecipients, rocshmem_my_pe() /* rank */, sendcounts[p], sdispls[p], recipients[p], putdispls[p]);
#endif
        const double * d_sendbufp = (const double *) d_sendbuf + sdispls[p];
        double * d_recvbufp = (double *) d_recvbuf + putdispls[p];
        /* rocshmemx_double_put_on_stream(d_recvbufp, d_sendbufp, sendcounts[p], recipients[p], stream); */
        rocshmemx_double_put_signal_on_stream(
            d_recvbufp, d_sendbufp, sendcounts[p],
            d_sigaddr, 1, ROCSHMEM_SIGNAL_ADD,
            recipients[p], stream);
        if (nsendbytes) *nsendbytes += sendcounts[p]*sendtypesize;
    }
    /* rocshmemx_barrier_all_on_stream(stream); */
    rocshmemx_signal_wait_until_on_stream(
        d_sigaddr, ROCSHMEM_CMP_GE, nsenders, stream);
    hipMemsetAsync(d_sigaddr, 0, sizeof(*d_sigaddr), stream);
    if (nsendmsgs) *nsendmsgs += nrecipients;
    return ACG_SUCCESS;
#else
    return ACG_ERR_ROCSHMEM_NOT_SUPPORTED;
#endif
}
#endif

#ifdef __cplusplus
}
#endif
