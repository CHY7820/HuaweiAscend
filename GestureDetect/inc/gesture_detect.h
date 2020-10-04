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
    Result Process(); // need to change
    void DeInit();

private:

    Result InitResource();
    Result InitModel(const char* OpenPoseModelPath, const char* GestureModelPath);
    Result ProcessMotionData();
    OpenPoseProcess OpenposeModel_;
    GestureProcess GestureModel_;
    DvppProcess dvpp_;

    int32_t deviceId_;
//    int32_t processed_img_num;

    aclrtContext context_;
    aclrtStream stream_;

//    uint32_t modelWidth_;
//    uint32_t modelHeight_;
    uint32_t inputDataSize_;
    aclrtRunMode runMode_;
    const char* OpenPoseModelPath_;
    const char* GestureModelPath_;
    const int FRAMES = 50;
    bool isInited_;

//    aclmdlDataset* input_;
//    aclmdlDataset* output_;
    std::shared_ptr<EngineTransNewT> motion_data_new = std::make_shared<EngineTransNewT>();
};