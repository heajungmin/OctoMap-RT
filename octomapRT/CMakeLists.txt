# Copyright 2024 The Ewha Womans University.
# All Rights Reserved.
#
# Permission to use, copy, modify OR distribute this software and its
# documentation for educational, research and non-profit purposes, without
# fee, and without a written agreement is hereby granted, provided that the
# above copyright notice and the following three paragraphs appear in all
# copies.
#
# IN NO EVENT SHALL THE EWHA WOMANS UNIVERSITY BE
# LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
# CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
# USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE EWHA WOMANS UNIVERSITY
# HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
#
# THE EWHA WOMANS UNIVERSITY SPECIFICALLY DISCLAIM ANY
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
# PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE EWHA WOMANS UNIVERSITY
# HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
#
#
# The authors may be contacted via:
#
# Mail:        Heajung Min, Kyung Min Han, and Young J. Kim
#              Computer Graphics Lab                       
#              Department of Computer Science and Engineering
#              Ewha Womans University
#              11-1 Daehyun-Dong Seodaemun-gu, Seoul, Korea 120-750
#
# Phone:       +82-2-3277-6798
#
# EMail:       hjmin@ewha.ac.kr
#              hankm@ewha.ac.kr
#              kimy@ewha.ac.kr

cmake_minimum_required(VERSION 3.7)
project(octomapRT)

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})
set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ~/Downloads/octomap/octomap/include
)

link_directories(~/Downloads/octomap/lib)

set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\"${OptiX_INCLUDE}\", \"${CUDA_INCLUDE_DIRS}\", ")

macro(OPTIX_octomapRT_compile_to_optix_input generated_files)
  if(NOT CUDA_NVRTC_ENABLED)
    if(SAMPLES_INPUT_GENERATE_OPTIXIR)
      message("octomapRT OPTIXIR")
      CUDA_WRAP_SRCS(octomapRT OPTIXIR generated_files2 ${ARGN})
      list(APPEND ${generated_files} ${generated_files2})
    endif()
    if(SAMPLES_INPUT_GENERATE_PTX)
      message("octomapRT PTX")
      CUDA_WRAP_SRCS(octomapRT PTX generated_files3 ${ARGN})
      list(APPEND ${generated_files} ${generated_files3})
    endif()
  endif()
endmacro()

set(sources
  octomapRT.cu 
  octomapRT.cpp 
  octomapRT.h
)

if(NOT CUDA_NVRTC_ENABLED)
  OPTIX_octomapRT_compile_to_optix_input(ptx_files ${sources})
else()
  set(ptx_files)
endif()

add_library(octomapRT SHARED ${sources} ${ptx_files})

target_link_libraries(octomapRT LINK_PRIVATE
  ${GLFW_LIB_NAME}
  glad
  imgui
  ${CUDA_LIBRARIES}
  octomap
  octomath
  sutil_7_sdk
)

set_property(TARGET octomapRT PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

