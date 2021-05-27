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
#include "camera.h"

using namespace std;
bool g_isDevice = false;


namespace {
const char* kOpenPoseModelPath = "../model/openpose.om"; // OpenPose offline model path
const char* kGestureModelPath = "../model/stgcn-deploy.om"; // STGCN offline model path

string kDataDir = "../data"; // file containing motion text data to be processed in Test function for debugging
}

// OpenPose model: (1,3,120,160) -- NCHW from caffe
// STGCN model: (1,100,18,3) -- NHWC from tensorflow

int main(int argc, char *argv[])
{
    int mode = 0; // 0--strawberry camera, 1--USB camera, 2--debug
    int channelId = 0; // camera id
    bool use_dvpp = true; // if use dvpp to process motion frame, set use_dvpp true;

    GestureDetect detect(kOpenPoseModelPath,kGestureModelPath);
    Result ret = detect.Init(use_dvpp);
    if (ret != SUCCESS) {
        ERROR_LOG("Detect init resource failed");
        return FAILED;
    }
    switch(mode)
    {
        case 0:
            detect.Process(channelId); // main process with strawberry camera
            break;
        case 1:
            ret = detect.Process(); // main process with USB camera, low fps
            break;
        case 2:
            detect.Test(kDataDir); // for debugging, process all txt files in kDataDir
            break;
    }


    INFO_LOG("Detect process success");
    detect.DeInit();

    return SUCCESS;
}