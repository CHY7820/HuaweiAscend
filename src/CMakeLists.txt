# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(sample_ascendcamera)

# Compile options
add_compile_options(-std=c++11)

# Specify target generation path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  "../../outputs")
set(CMAKE_INSTALL_PREFIX "../../../run")
set(CMAKE_OUTPUT_DIR "out")


set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

#execute_process(COMMAND bash ../../../script/deploy.sh RESULT_VARIABLE ret)
#IF(${ret})
#   message(FATAL_ERROR "some error occured!")
#ENDIF(${ret})

# Header path
include_directories(
.
$ENV{DDK_PATH}/include/inc/
$ENV{DDK_PATH}/include/third_party/protobuf/include
$ENV{DDK_PATH}/include/third_party/cereal/include
$ENV{DDK_PATH}/include/libc_sec/include
$ENV{HOME}/ascend_ddk/include
)

link_directories($ENV{NPU_HOST_LIB} $ENV{HOME}/ascend_ddk/host/lib /home/HwHiAiUser/HIAI_PROJECTS/ascend_lib $ENV{HOME}/ascend_ddk/device/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(main Main.cpp ascend_camera_parameter.cpp camera.cpp main_process.cpp output_info_process.cpp parameter_utils.cpp)
target_link_libraries(main c_sec matrixdaemon mmpa pthread dl rt hiai_common media_mini presenteragent Dvpp_api Dvpp_jpeg_decoder Dvpp_jpeg_encoder Dvpp_vpc ascend_ezdvpp)

install(TARGETS main DESTINATION ${CMAKE_OUTPUT_DIR})
