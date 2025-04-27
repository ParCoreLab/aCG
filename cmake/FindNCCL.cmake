#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tries to find NCCL headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(NCCL)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  NCCL_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the NCCL installation.
#              The environment variable NCCL_ROOT overrides this variable.
#  NCCL_HOME - When set, this path is inspected instead of standard library
#              locations as the root of the NCCL installation.
#              The environment variable NCCL_HOME overrides this variable.
#
# This module defines
#  NCCL_FOUND, whether nccl has been found
#  NCCL_INCLUDE_DIR, directory containing header
#  NCCL_LIBRARY, path to nccl library
#
# This module assumes that the user has already called find_package(CUDA)

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS  ${NCCL_HOME}/include $ENV{NCCL_HOME}/include ${NCCL_ROOT}/include $ENV{NCCL_ROOT}/include)

if(USE_DLOPEN_NCCL)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR)

  mark_as_advanced(NCCL_INCLUDE_DIR)
else()
  find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NCCL_HOME}/lib $ENV{NCCL_HOME}/lib ${NCCL_ROOT}/lib $ENV{NCCL_ROOT}/lib)

  message(STATUS "Using nccl library: ${NCCL_LIBRARY}")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(NCCL DEFAULT_MSG
    NCCL_INCLUDE_DIR NCCL_LIBRARY)

  mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY)
endif()
