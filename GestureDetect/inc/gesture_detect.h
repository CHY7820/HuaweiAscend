#pragma once

#include "gesture_process.h"
#include "pose_process.h"
#include "dvpp_process.h"
#include "utils.h"
#include "acl/acl.h"

class GestureDetect {
public:
    GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath,uint32_t ImgWidth,uint32_t ImgHeight);
    ~GestureDetect();
    Result Init(); // didn't create input?
    Result Process();
//    void DeInit();

private:

    Result InitResource();
    Result InitModel();
//    Result OpenPoseProcess();
//    Result GestureProcess();

    OpenPoseProcess OpenposeModel_;
    GestureProcess GestureModel_;
    DvppProcess dvpp_;

    int32_t deviceId_;
    int32_t processed_img_num;

    aclrtContext context_;
    aclrtStream stream_;

//    uint32_t modelWidth_;
//    uint32_t modelHeight_;
    uint32_t inputDataSize_;
    //aclrtRunMode urnMode_;
    const char* OpenPoseModelPath_;
    const char* GestureModelPath_;
    const int FRAMES = 50;
    bool isInited_;
}