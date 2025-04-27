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

#include "hip/hip_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(ACG_HAVE_MPI) && defined(ACG_HAVE_HIP)
__global__ void acghalo_pack_hip_double(
    int sendbufsize,
    double * sendbuf,
    int srcbufsize,
    const double * srcbuf,
    const int * srcbufidx)
{
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < sendbufsize;
    //      i += blockDim.x * gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
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
    MPI_Datatype datatype,
    int srcbufsize,
    const void * d_srcbuf,
    const int * d_srcbufidx,
    int64_t * nbytes,
    int * mpierrcode)
{
    if (datatype == MPI_DOUBLE) {
        // static int mingridsize = 0, blocksize = 0;
        // if (mingridsize == 0 && blocksize == 0) {
        //     hipOccupancyMaxPotentialBlockSize(
        //         &mingridsize, &blocksize, acghalo_pack_hip_double, 0, 0);
        // }
        int mingridsize = (sendbufsize + 255) / 256, blocksize = 256;
        acghalo_pack_hip_double<<<mingridsize,blocksize>>>(
            sendbufsize, (double *) d_sendbuf,
            srcbufsize, (const double *) d_srcbuf, d_srcbufidx);
        if (hipPeekAtLastError()) return ACG_ERR_HIP;
        if (nbytes) *nbytes += sendbufsize*(2*sizeof(double)+sizeof(*d_srcbufidx));
    } else { return ACG_ERR_NOT_SUPPORTED; }
    return ACG_SUCCESS;
}

__global__ void acghalo_unpack_hip_double(
    int recvbufsize,
    const double * recvbuf,
    int dstbufsize,
    double * dstbuf,
    const int * dstbufidx)
{
    // for (int i = blockIdx.x*blockDim.x+threadIdx.x;
    //      i < recvbufsize;
    //      i += blockDim.x * gridDim.x)
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n)
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
    MPI_Datatype datatype,
    int dstbufsize,
    void * d_dstbuf,
    const int * d_dstbufidx,
    int64_t * nbytes,
    int * mpierrcode)
{
    if (datatype == MPI_DOUBLE) {
        // static int mingridsize = 0, blocksize = 0;
        // if (mingridsize == 0 && blocksize == 0) {
        //     hipOccupancyMaxPotentialBlockSize(
        //         &mingridsize, &blocksize, acghalo_unpack_hip_double, 0, 0);
        // }
        int mingridsize = (recvbufsize + 255) / 256, blocksize = 256;
        acghalo_unpack_hip_double<<<mingridsize,blocksize>>>(
            recvbufsize, (const double *) d_recvbuf,
            dstbufsize, (double *) d_dstbuf, d_dstbufidx);
        if (hipPeekAtLastError()) return ACG_ERR_HIP;
        if (nbytes) *nbytes += recvbufsize*(2*sizeof(double)+sizeof(*d_dstbufidx));
    } else { return ACG_ERR_NOT_SUPPORTED; }
    return ACG_SUCCESS;
}
#endif

#ifdef __cplusplus
}
#endif
