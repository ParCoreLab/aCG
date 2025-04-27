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

#ifndef ACG_HALO_H
#define ACG_HALO_H

#include "acg/config.h"
#include "acg/comm.h"

#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif
#ifdef ACG_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef ACG_HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif

#include <stdio.h>

#ifndef ACG_HALO_MAX_EXCHANGE_STATS
#define ACG_HALO_MAX_EXCHANGE_STATS 1024
#endif
#ifndef ACG_HALO_MAX_PERF_EVENTS
#define ACG_HALO_MAX_PERF_EVENTS 1024
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ‘acghalo’ is a data structure used for halo exchange communication
 * patterns for data distributed across multiple MPI processes.
 *
 * This kind of communication pattern can be described by an
 * unstructured, irregular communication graph, where neighbouring
 * processes of the graph exchange messages pairwise. Each process
 * typically exchanges data with only a few others. The communication
 * pattern and message sizes are irregular and depend on the
 * underlying data.
 */
struct acghalo
{
    /*
     * sender
     */

    /**
     * ‘nrecipients’ is the number of processes for which the current
     * process will send messages to.
     */
    int nrecipients;

    /**
     * ‘recipients’ is an array of length ‘nrecipients’ containing the
     * ranks of the processes to send messages to.
     */
    int * recipients;

    /**
     * ‘sendcounts’ is an array of length ‘nrecipients’ containing the
     * number of array elements to send to each process. In other
     * words, ‘sendcounts[i]’ elements are sent to the process with
     * rank ‘recipients[i]’, for ‘i=0,1,...,nrecipients-1’.
     */
    int * sendcounts;

    /**
     * ‘sdispls’ is an array of length ‘nrecipients’ containing the
     * offsets to the first element to send to each process. More
     * specifically, ‘sdispls[i]’ is the offset to the first element
     * to send to the process with rank ‘recipients[i]’, for
     * ‘i=0,1,...,nrecipients-1’.
     */
    int * sdispls;

    /**
     * ‘sendsize’ is the total number of elements for the current
     * process to send to its recipients.
     */
    int sendsize;

    /**
     * ‘sendbufidx’ is an array of length ‘sendsize’ containing the
     * offsets to elements in the source array for which data is sent
     * during the halo exchange.
     *
     * Before sending ‘sendsize’ elements from a buffer ‘sendbuf’, it
     * may be necessary to pack the data to be sent. In this case, the
     * ‘i’th element to send, ‘sendbuf[i]’, should be copied from
     * ‘srcbuf[sendbufidx[i]]’ for a source array ‘srcbuf’.
     */
    int * sendbufidx;

    /*
     * receiver
     */

    /**
     * ‘nsenders’ is the number of processes for which the current
     * process will receive messages from.
     */
    int nsenders;

    /**
     * ‘senders’ is an array of length ‘nsenders’ containing the
     * ranks of the processes to receive messages from.
     */
    int * senders;

    /**
     * ‘recvcounts’ is an array of length ‘nsenders’ containing the
     * number of array elements to receive from each process. In other
     * words, ‘recvcounts[i]’ elements are received from the process
     * with rank ‘senders[i]’, for ‘i=0,1,...,nsenders-1’.
     */
    int * recvcounts;

    /**
     * ‘rdispls’ is an array of length ‘nsenders’ containing the
     * offsets to the first element to receive from each sending
     * process. More specifically, ‘rdispls[i]’ is the offset to the
     * first element received from the process with rank ‘senders[i]’,
     * for ‘i=0,1,...,nsenders-1’.
     */
    int * rdispls;

    /**
     * ‘recvsize’ is the total number of elements for the current
     * process to receive from its senders.
     */
    int recvsize;

    /**
     * ‘recvbufidx’ is an array of length ‘recvsize’ containing the
     * offsets to elements in the destination array for which data is
     * received during the halo exchange.
     *
     * After receiving ‘recvsize’ elements into a buffer ‘recvbuf’, it
     * may be necessary to unpack the received data. In this case, the
     * ‘i’th element received, ‘recvbuf[i]’, should be copied to
     * ‘dstbuf[recvbufidx[i]]’ for a destination array ‘dstbuf’.
     */
    int * recvbufidx;

    /* performance breakdown of time spent in different parts */
    int nexchanges;
    double texchange;
    double tpack, tunpack, tsendrecv, tmpiirecv, tmpisend, tmpiwaitall;
    int64_t npack, nunpack, nmpiirecv, nmpisend;
    int64_t Bpack, Bunpack, Bmpiirecv, Bmpisend;

    /* performance breakdown per call to acghalo_exchange_cuda */
    int maxexchangestats;
    double (* thaloexchangestats)[4];
};

/**
 * ‘acghalo_init()’ sets up a halo exchange pattern based on a
 * partitioned, unstructured computational mesh.
 */
ACG_API int acghalo_init(
    struct acghalo * halo,
    int nsendnodes,
    const acgidx_t * sendnodetags,
    const int * sendnodenneighbours,
    const int * sendnodeneighbours,
    acgidx_t nrecvnodes,
    const acgidx_t * recvnodetags,
    const int * recvnodeparts);

/**
 * ‘acghalo_free()’ frees resources associated with a halo exchange.
 */
ACG_API void acghalo_free(
    struct acghalo * halo);

/**
 * ‘acghalo_copy()’ creates a copy of a halo exchange data structure.
 */
ACG_API int acghalo_copy(
    struct acghalo * dst,
    const struct acghalo * src);

/*
 * output (e.g., for debugging)
 */

ACG_API int acghalo_fwrite(
    FILE * f,
    const struct acghalo * halo);

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
ACG_API int acghalo_pack(
    int sendbufsize,
    void * sendbuf,
    MPI_Datatype datatype,
    int srcbufsize,
    const void * srcbuf,
    const int * srcbufidx,
    int64_t * nbytes,
    int * mpierrcode);

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
ACG_API int acghalo_unpack(
    int recvbufsize,
    const void * recvbuf,
    MPI_Datatype datatype,
    int dstbufsize,
    void * dstbuf,
    const int * dstbufidx,
    int64_t * nbytes,
    int * mpierrcode);
#endif

/*
 * halo communication routines
 */

#ifdef ACG_HAVE_MPI
/**
 * ‘acghalo_exchange()’ performs a halo exchange.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange(
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
    int * mpierrcode);
#endif

/*
 * halo communication with CUDA-aware MPI
 */

struct acghaloexchange
{
    enum acgdatatype sendtype;
    enum acgdatatype recvtype;
    void * sendbuf;
    void * recvbuf;
    void * sendreqs;
    void * recvreqs;
#if defined(ACG_HAVE_CUDA) || defined(ACG_HAVE_HIP)
    void * d_sendbuf;
    void * d_recvbuf;
    void * d_sendbufidx;
    void * d_recvbufidx;
    int * d_recipients;
    int * d_sendcounts;
    int * d_sdispls;
    int * d_senders;
    int * d_recvcounts;
    int * d_rdispls;
    int * d_putdispls;
    int * d_putranks;
    int * d_getranks;
#if defined(ACG_HAVE_CUDA)
    cudaStream_t cudastream;
#endif
#if defined(ACG_HAVE_HIP)
    hipStream_t hipstream;
#endif
    uint64_t * d_received, * d_readytoreceive;
    int * putdispls;
    int * putranks;
    int * getranks;
    int use_nvshmem;
    int use_rocshmem;

    /* performance breakdown per call to acghalo_exchange_cuda */
    int maxevents, nevents;
#if defined(ACG_HAVE_CUDA)
    cudaEvent_t (* cudaevents)[4];
#endif
#if defined(ACG_HAVE_HIP)
    hipEvent_t (* hipevents)[4];
#endif
#endif
};

/**
 * ‘acghaloexchange_init()’ allocate additional storage needed to
 * perform a halo exchange.
 */
ACG_API int acghaloexchange_init(
    struct acghaloexchange * haloexchange,
    const struct acghalo * halo,
    enum acgdatatype sendtype,
    enum acgdatatype recvtype,
    const struct acgcomm * comm);

#if defined(ACG_HAVE_CUDA)
/**
 * ‘acghaloexchange_init_cuda()’ allocate additional storage needed
 * to perform a halo exchange for data residing on a CUDA device.
 */
ACG_API int acghaloexchange_init_cuda(
    struct acghaloexchange * haloexchange,
    const struct acghalo * halo,
    enum acgdatatype sendtype,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    cudaStream_t stream);
#endif

#if defined(ACG_HAVE_HIP)
/**
 * ‘acghaloexchange_init_hip()’ allocate additional storage needed
 * to perform a halo exchange for data residing on a HIP device.
 */
ACG_API int acghaloexchange_init_hip(
    struct acghaloexchange * haloexchange,
    const struct acghalo * halo,
    enum acgdatatype sendtype,
    enum acgdatatype recvtype,
    const struct acgcomm * comm,
    hipStream_t stream);
#endif

/**
 * ‘acghaloexchange_free()’ free resources associated with a halo
 * exchange.
 */
ACG_API void acghaloexchange_free(
    struct acghaloexchange * haloexchange);

/**
 * ‘acghaloexchange_profile()’ obtain detailed performance profiling
 * information for halo exchanges.
 */
ACG_API int acghaloexchange_profile(
    const struct acghaloexchange * haloexchange,
    int maxevents,
    int * nevents,
    double * texchange,
    double * tpack,
    double * tsendrecv,
    double * tunpack);

#if defined(ACG_HAVE_MPI) && defined(ACG_HAVE_CUDA)
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
ACG_API int acghalo_pack_cuda(
    int sendbufsize,
    void * d_sendbuf,
    enum acgdatatype datatype,
    int srcbufsize,
    const void * d_srcbuf,
    const int * d_srcbufidx,
    cudaStream_t stream,
    int64_t * nbytes,
    int * errcode);

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
ACG_API int acghalo_unpack_cuda(
    int recvbufsize,
    const void * d_recvbuf,
    enum acgdatatype datatype,
    int dstbufsize,
    void * d_dstbuf,
    const int * d_dstbufidx,
    cudaStream_t stream,
    int64_t * nbytes,
    int * errcode);

/**
 * ‘acghalo_exchange_cuda()’ performs a halo exchange for data
 * residing on CUDA devices.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange_cuda(
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
    int warmup);

/**
 * ‘acghalo_exchange_cuda_begin()’ starts a halo exchange for data
 * residing on CUDA devices.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange_cuda_begin(
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
    cudaStream_t stream);

/**
 * ‘acghalo_exchange_cuda_end()’ starts a halo exchange for data
 * residing on CUDA devices.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange_cuda_end(
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
    cudaStream_t stream);
#endif

/*
 * halo communication with NVSHMEM
 */

#ifdef ACG_HAVE_CUDA
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
ACG_API int halo_alltoallv_nvshmem(
    int sendsize,
    const void * sendbuf,
    int nrecipients,
    const int * recipients,
    const int * sendcounts,
    const int * sdispls,
    enum acgdatatype sendtype,
    const int * putdispls,
    uint64_t * received,
    uint64_t * readytoreceive,
    int recvsize,
    void * recvbuf,
    int nsenders,
    const int * senders,
    const int * recvcounts,
    const int * rdispls,
    enum acgdatatype recvtype,
    MPI_Comm comm,
    cudaStream_t stream,
    int * errcode,
    int64_t * nsendmsgs,
    int64_t * nsendbytes,
    int64_t * nrecvmsgs,
    int64_t * nrecvbytes);
#endif

#if defined(ACG_HAVE_MPI) && defined(ACG_HAVE_HIP)
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
ACG_API int acghalo_pack_hip(
    int sendbufsize,
    void * d_sendbuf,
    enum acgdatatype datatype,
    int srcbufsize,
    const void * d_srcbuf,
    const int * d_srcbufidx,
    hipStream_t stream,
    int64_t * nbytes,
    int * errcode);

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
ACG_API int acghalo_unpack_hip(
    int recvbufsize,
    const void * d_recvbuf,
    enum acgdatatype datatype,
    int dstbufsize,
    void * d_dstbuf,
    const int * d_dstbufidx,
    hipStream_t stream,
    int64_t * nbytes,
    int * errcode);

/**
 * ‘acghalo_exchange_hip()’ performs a halo exchange for data
 * residing on HIP devices.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange_hip(
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
    int warmup);

/**
 * ‘acghalo_exchange_hip_begin()’ starts a halo exchange for data
 * residing on HIP devices.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange_hip_begin(
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
    hipStream_t stream);

/**
 * ‘acghalo_exchange_hip_end()’ starts a halo exchange for data
 * residing on HIP devices.
 *
 * This function returns ‘ACG_ERR_MPI’ if it fails due to an MPI
 * error. Moreover, if ‘mpierrcode’ is not ‘NULL’, then it may be used
 * to store any error codes that are returned by underlying MPI calls.
 */
ACG_API int acghalo_exchange_hip_end(
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
    hipStream_t stream);
#endif

#ifdef __cplusplus
}
#endif

#endif
