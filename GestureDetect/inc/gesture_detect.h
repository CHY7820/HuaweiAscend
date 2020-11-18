#pragma once

#include "gesture_process.h"
#include "pose_process.h"
#include "utils.h"
#include "acl/acl.h"


class GestureDetect {
public:
    GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath);
    ~GestureDetect();
    Result Init();
    Result Process();
    void DeInit();
    float motion_data [1][3][FRAME_LENGTH][18];

private:

    Result InitResource();
    Result InitModel(const char* OpenPoseModelPath, const char* GestureModelPath);
    Result ProcessMotionData();


    OpenPoseProcess OpenPoseModel_;
    GestureProcess GestureModel_;

    int32_t deviceId_;

    aclrtContext context_;
    aclrtStream stream_;

    const char* OpenPoseModelPath_;
    const char* GestureModelPath_;

    bool isInited_;


    aclmdlDataset* input_;
    aclmdlDataset* output_;
//    std::shared_ptr<EngineTransNewT> motion_data_new = std::make_shared<EngineTransNewT>();
};
