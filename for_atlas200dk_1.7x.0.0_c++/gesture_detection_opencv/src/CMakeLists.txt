# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(GestureDetect)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)
# set(INC_PATH $ENV{DDK_PATH})
# set(LIB_PATH $ENV{NPU_HOST_LIB})
set(INC_PATH "/home/mind/Ascend/ascend-toolkit/20.0.RC1/acllib_centos7.6.aarch64")
set(LIB_PATH "/home/mind/Ascend/ascend-toolkit/20.0.RC1/acllib_centos7.6.aarch64/acllib/lib64/stub")

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()



if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
# message(STATUS "$ENV{HOME}")
include_directories(
    $ENV{HOME}/ascend_ddk/arm/include/
    $ENV{HOME}/ascend_ddk/arm/include/opencv4
    $ENV{HOME}/presenteragent/include/
    ${INC_PATH}/acllib/include/
    ../inc/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    ${LIB_PATH}
    $ENV{HOME}/ascend_ddk/arm/lib/
    ${INC_PATH}/atc/lib64
    $ENV{HOME}/presenteragent/include/agent/
)

add_executable(main
        utils.cpp
        model_process.cpp
        pose_process.cpp
        gesture_process.cpp
        gesture_detect.cpp
        main.cpp)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl stdc++ ascendcl acl_dvpp stdc++ opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio protobuf presenteragent dl rt)

endif()
