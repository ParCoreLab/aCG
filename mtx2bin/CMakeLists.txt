# This file is part of acg.
#
# Copyright 2025 Koç University and Simula Research Laboratory
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors:
#  James D. Trotter <james@simula.no>
#  Sinan Ekmekçibaşı <sekmekcibasi23@ku.edu.tr>
#
# Last modified: 2025-04-26

cmake_minimum_required(VERSION 3.12)

project(mtx2bin LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

set(CMAKE_CXX_STANDARD 17)

set(PROJECT_SOURCES
	../acg/config.h
	../acg/comm.c
	../acg/comm.h
#	../acg/comm-rocshmem.hip
	../acg/error.c
	../acg/error.h
	../acg/fmtspec.c
	../acg/fmtspec.h
	../acg/graph.c
	../acg/graph.h
	../acg/halo.c
#	../acg/halo-kernels-hip.hip
	../acg/halo.h
	../acg/metis.c
	../acg/metis.h
	../acg/mtxfile.c
	../acg/mtxfile.h
	../acg/prefixsum.c
	../acg/prefixsum.h
	../acg/sort.c
	../acg/sort.h
	../acg/symcsrmatrix.c
	../acg/symcsrmatrix.h
	../acg/vector.c
	../acg/vector.h
        mtx2bin.c
        )

include(CheckCSourceCompiles)
check_c_source_compiles(
"#define _POSIX_C_SOURCE 200112L
#include <time.h>
int main () {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return 0;
}" HAVE_CLOCK_GETTIME)
if(HAVE_CLOCK_GETTIME)
    add_definitions(-DHAVE_CLOCK_GETTIME)
endif()

find_package(ZLIB)
if (ZLIB_FOUND)
    add_definitions(-DACG_HAVE_LIBZ)
endif ()

if (IDXSIZE EQUAL 64)
  message(STATUS "index type for matrices and vectors (acgidx_t) - int64_t")
  add_definitions(-DACG_IDX_SIZE=64)
else ()
  message(STATUS "index type for matrices and vectors (acgidx_t) - int")
endif ()

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} m)
target_include_directories(${PROJECT_NAME} PRIVATE ..)
if (ZLIB_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${ZLIB_INCLUDE_DIRS})
  target_link_directories(${PROJECT_NAME} PRIVATE ${ZLIB_LIBRARY_DIR})
  target_link_libraries(${PROJECT_NAME} z)
endif ()
