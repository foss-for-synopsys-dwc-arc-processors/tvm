# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_METAWARE STREQUAL "ON") 
  
  tvm_file_glob(GLOB METAWARE_RELAY_CONTRIB_SRC src/relay/backend/contrib/metaware/*.cc)

  list(APPEND COMPILER_SRCS ${METAWARE_RELAY_CONTRIB_SRC})

  tvm_file_glob(GLOB METAWARE_CONTRIB_SRC src/runtime/contrib/metaware/*.cc)

  list(APPEND RUNTIME_SRCS ${METAWARE_CONTRIB_SRC})

  # TODO: Update when MWTVM packaging finalizes
  #
  if(DEFINED ENV{MWTVM_INSTALL})
    set(MWTVM_INSTALL $ENV{MWTVM_INSTALL})
    include_directories(${MWTVM_INSTALL}/include)

    find_library(EXTERN_MWTVM_LIB NAMES mwtvm HINTS ${MWTVM_INSTALL}/lib)
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_MWTVM_LIB})

  else()
    message(FATAL_ERROR "METAWARE builds currently need MWTVM_INSTALL environment variable")
  endif()

  message(STATUS "Build with METAWARE support ")

elseif(USE_METAWARE STREQUAL "OFF")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_METAWARE=" ${USE_METAWARE})
endif()
