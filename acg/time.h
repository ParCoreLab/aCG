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
 * timing functions
 */

#ifndef ACG_TIME_H
#define ACG_TIME_H

#include "acg/config.h"

#include <time.h>
#include <unistd.h>

#if defined(ACG_HAVE_MPI)
typedef double acgtime_t;
#elif defined(HAVE_CLOCK_GETTIME)
typedef struct timespec acgtime_t;
#elif defined(__MACH__)
#include <mach/clock.h>
#include <mach/mach.h>
typedef mach_timespec_t acgtime_t;
#else
typedef struct {} acgtime_t;
#endif

/**
 * ‘gettime()’ records the current time (via clock_gettime()),
 * preferably using a monotonic clock, if supported.
 */
static inline void gettime(acgtime_t * t)
{
#if defined(ACG_HAVE_MPI)
    *t = MPI_Wtime();
#elif defined(HAVE_CLOCK_GETTIME)
#if _POSIX_TIMERS >= 200809L && _POSIX_MONOTONIC_CLOCK > 0
    clock_gettime(CLOCK_MONOTONIC, t);
#else
    clock_gettime(CLOCK_REALTIME, t);
#endif
#elif defined(__MACH__)
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    t->tv_sec = mts.tv_sec;
    t->tv_nsec = mts.tv_nsec;
#endif
}

/**
 * ‘elapsed()’ is the duration, in seconds, elapsed between
 * two given time points.
 */
static inline double elapsed(
    acgtime_t t0,
    acgtime_t t1)
{
#if defined(ACG_HAVE_MPI)
    return t1-t0;
#elif defined(HAVE_CLOCK_GETTIME) || defined(__MACH__)
    return (t1.tv_sec - t0.tv_sec) +
        (t1.tv_nsec - t0.tv_nsec) * 1e-9;
#else
    return 0.0;
#endif
}

#endif
