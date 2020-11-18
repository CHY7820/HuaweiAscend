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
const char* kOpenPoseModelPath = "../model/openpose0.om";
const char* kGestureModelPath = "../model/stgcn.om";
const std::string kImageDir = "../data/";

}

int main()
{
    GestureDetect detect(kOpenPoseModelPath,kGestureModelPath);
    Result ret = detect.Init();
    if (ret != SUCCESS) {
        ERROR_LOG("Detect init resource failed");
        return FAILED;
    }

    ret = detect.Process();
    if (ret != SUCCESS) {
        ERROR_LOG("Detect process failed");
        return FAILED;
    }

    INFO_LOG("Detect process success");


    return SUCCESS;
}