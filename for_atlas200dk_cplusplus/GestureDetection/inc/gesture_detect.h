#pragma once

#include "gesture_process.h"

#include "acl/acl.h"
#include "camera.h"
#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

#include "pose_process.h"
#include "utils.h"

using namespace ascend::presenter;
class GestureDetect {
public:
    GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath);
    ~GestureDetect();
    Result Init(bool use_dvpp);
    Result Process();
    Result Process(int channelId);
    void Test(string data_dir);
    void DeInit();



private:

    Result InitResource();
    Result InitModel(const char* OpenPoseModelPath, const char* GestureModelPath,bool use_dvpp);
    Result ProcessMotionData();
    Result OpenPresenterChannel();
    void EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg);
    void SendImage(cv::Mat& image);
    void SendImage(ImageData& jpegImage,vector<DetectionResult>& detRes);


    OpenPoseProcess OpenPoseModel_;
    GestureProcess GestureModel_;
    DvppProcess dvpp_;


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
