# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(YOLO_HBB_OBB)

add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/7.0.RC1/runtime/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

set(THIRDPART $ENV{THIRDPART_PATH})
if (NOT DEFINED ENV{THIRDPART_PATH})
    set(THIRDPART "/usr/local/Ascend/thirdpart/aarch64")
    message(STATUS "set default THIRDPART: ${THIRDPART}")
else()
    message(STATUS "set THIRDPART: ${THIRDPART}")
endif()

include_directories(
   ${INC_PATH}/runtime/include/
   /usr/local/include/opencv4/
   ${THIRDPART}/include/acllite/
   /usr/local/Ascend/ascend-toolkit/latest/thirdpart/include/acllite
)

link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
    /usr/local/lib
)

set(CMAKE_BUILD_TYPE "Debug")

# -------- YOLO HBB 可执行文件 --------
add_executable(main_hbb YOLO_hbb.cpp)
if(target STREQUAL "Simulator_Function")
    target_link_libraries(main_hbb funcsim)
else()
    target_link_libraries(main_hbb ascendcl acl_dvpp stdc++ acllite opencv_core opencv_imgproc opencv_imgcodecs dl rt)
endif()
install(TARGETS main_hbb DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# -------- YOLO OBB 可执行文件 --------
add_executable(main_obb YOLO_obb.cpp)
if(target STREQUAL "Simulator_Function")
    target_link_libraries(main_obb funcsim)
else()
    target_link_libraries(main_obb ascendcl acl_dvpp stdc++ acllite opencv_core opencv_imgproc opencv_imgcodecs dl rt)
endif()
install(TARGETS main_obb DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
