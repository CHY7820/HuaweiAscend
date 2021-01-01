#pragma once

#include "gesture_process.h"

#include "acl/acl.h"

#include "pose_process.h"
#include "utils.h"


class GestureDetect {
public:
    GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath);
    ~GestureDetect();
    Result Init();
    Result Process();
    void DeInit();


private:

    Result InitResource();
    Result InitModel(const char* OpenPoseModelPath, const char* GestureModelPath);
    Result ProcessMotionData();
    Result OpenPresenterChannel();
    void EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg);
    void SendImage(cv::Mat& image);

    OpenPoseProcess OpenPoseModel_;
    GestureProcess GestureModel_;

    int32_t deviceId_;

    aclrtContext context_;
    aclrtStream stream_;

    const char* OpenPoseModelPath_;
    const char* GestureModelPath_;

    bool isInited_;
    ascend::presenter::Channel* channel_;

    aclmdlDataset* input_;
    aclmdlDataset* output_;
};
