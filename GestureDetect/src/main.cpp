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
const char* kGestureModelPath = "../model/msg3d.om";
uint32_t ImgWidth = 128;
uint32_t ImgHeight = 128;
}

extern int limbSeq[19][2] = { {2,3}, {2,6}, {3,4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10}, {10, 11}, {2, 12},
{12, 13}, {13, 14}, {1, 2}, {1, 15},{15, 17}, {1, 16},{16, 18},{3, 17},{6, 18} };

extern int mapIdx[19][2] = { {31,32}, {39,40}, {33,34}, {35,36}, {41,42}, {43,44}, {19,20}, {21,22}, {23,24},
{25,26}, {27,28}, {29,30}, {47,48},{49,50},{53,54},{51,52},{55,56},{37,38},{45,46} };

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