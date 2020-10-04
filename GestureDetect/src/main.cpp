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
const char* kOpenPoseModelPath = "../model/pose_deploy_final.om";
//const char* kGestureModelPath = "../model/msg3d.om";
const char* kGestureModelPath = "../model/colorization.om";
uint32_t ImgWidth = 128;
uint32_t ImgHeight = 128;
}

int main()
{
    GestureDetect detect(kOpenPoseModelPath,kGestureModelPath,ImgWidth,ImgHeight);
    Result ret = detect.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("Detect init resource failed");
        return FAILED;
    }

//    //Get all the image file path in the image directory
//    string inputImageDir = string(argv[1]);
//    vector<string> fileVec;
//    Utils::GetAllFiles(inputImageDir, fileVec);
//    if (fileVec.empty()) {
//        ERROR_LOG("Failed to read image from %s, and hidden dotfile "
//        "directory is ignored", inputImageDir.c_str());
//        return FAILED;
//    }




    ret = detect.Process();
    if (ret != SUCCESS) {
        ERROR_LOG("Detect process failed");
        return FAILED;
    }

    INFO_LOG("Detect process success");
    return SUCCESS;
}