/**
* @file main.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>

#include "gesture_detect.h"
#include "utils.h"

using namespace std;
bool g_isDevice = false;


namespace {
const char* kOpenPoseModelPath = "../model/openpose_deploy.om";
const char* kGestureModelPath = "../model/stgcn_deploy_24.om";
string kImageDir = "../data/";
}

// OpenPose model: (1,3,120,160) -- NCHW from caffe
// STGCN model: (1,3,100,18) -- NHWC from tensorflow

int main()
{
    GestureDetect detect(kOpenPoseModelPath,kGestureModelPath);
    Result ret = detect.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("Detect init resource failed");
        return FAILED;
    }

    ret = detect.Process(); // main process here
    if (ret != SUCCESS) {
        ERROR_LOG("Detect process failed");
        return FAILED;
    }

    INFO_LOG("Detect process success");
    detect.DeInit();

    return SUCCESS;
}