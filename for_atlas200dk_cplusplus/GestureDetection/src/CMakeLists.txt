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


# DDK_PATH and NPU_HOST_LIB are defined in ADK in settings

add_definitions(-DENABLE_DVPP_INTERFACE)
set(INC_PATH $ENV{DDK_PATH}) # ascend-toolkit 20.1
set(LIB_PATH $ENV{NPU_HOST_LIB})

set(ROOT_PATH "/home/mind/Ascend")
#message("${INC_PATH}/acllib/include/ ")
# Header path
# message(STATUS "$ENV{HOME}")
# message("${ROOT_PATH}/ascend_ddk/arm/include/opencv4")
include_directories(
    ${ROOT_PATH}/ascend_ddk/arm/include/opencv4/ # opencv
    ${ROOT_PATH}/ascend_ddk/arm/include/
    ${ROOT_PATH}/presenteragent/include/ # presenter agent lib path
    ${ROOT_PATH}/presenteragent/include/ascenddk/ # presenter agent lib path
    ${INC_PATH}/acllib/include/ # acllib
    ${ROOT_PATH}/  # driver etc.
    ../inc/
)

if(target STREQUAL "Simulator_Function")
    add_compile_options(-DFUNC_SIM)
endif()

# add host lib path
link_directories(
    ${LIB_PATH}
    ${ROOT_PATH}/driver/
    ${ROOT_PATH}/ascend_ddk/arm/lib/ # opencv here
    ${ROOT_PATH}}/presenteragent/include/agent/ # presenteragent here
)

add_executable(main
        utils.cpp
        camera.cpp
        model_process.cpp
        dvpp_process.cpp
        dvpp_resize.cpp
        dvpp_jpege.cpp
        pose_process.cpp
        gesture_process.cpp
        gesture_detect.cpp
        main.cpp)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl stdc++ ascendcl acl_dvpp stdc++ c_sec slog media_mini pthread opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio protobuf presenteragent dl rt)

endif()
