cmake_minimum_required(VERSION 3.24)

project(tiny_nccl VERSION 0.1.0 LANGUAGES CXX)

option(TNCCL_BUILD_CUDA "Build the CUDA library" ON)
option(TNCCL_BUILD_EXAMPLES "Build examples" ON)
option(TNCCL_BUILD_TESTS "Build CPU model tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(TNCCL_BUILD_CUDA)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "the tiny-nccl CUDA library supports Linux only")
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(tiny_nccl STATIC
      src/api.cc
      src/bootstrap.cc
      src/comm.cc
      src/ring_allreduce.cu)
    add_library(tiny-nccl::tiny-nccl ALIAS tiny_nccl)

    target_include_directories(tiny_nccl
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(tiny_nccl PUBLIC CUDA::cudart)
    target_compile_features(tiny_nccl PUBLIC cxx_std_17 cuda_std_17)
    target_compile_options(tiny_nccl PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    set_target_properties(tiny_nccl PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      POSITION_INDEPENDENT_CODE ON)

    if(TNCCL_BUILD_EXAMPLES)
      add_executable(tnccl_allreduce examples/allreduce.cc)
      target_link_libraries(tnccl_allreduce PRIVATE tiny-nccl::tiny-nccl)
      target_compile_options(tnccl_allreduce PRIVATE -Wall -Wextra -Wpedantic)
    endif()
  else()
    message(FATAL_ERROR
      "TNCCL_BUILD_CUDA=ON but no CUDA compiler was found; use OFF for CPU model tests")
  endif()
endif()

if(TNCCL_BUILD_TESTS)
  enable_testing()
  add_executable(tnccl_ring_model_test tests/ring_model_test.cc)
  target_compile_options(tnccl_ring_model_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME ring_model COMMAND tnccl_ring_model_test)
endif()
