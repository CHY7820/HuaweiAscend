#pragma once

#include "gesture_process.h"
#include "pose_process.h"
#include "utils.h"
#include "acl/acl.h"

class GestureDetect {
public:
    GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath,uint32_t ImgWidth,uint32_t ImgHeight);
    ~GestureDetect();
    Result Init();
    Result Process();
    void DeInit();

private:

    Result InitResource();
    Result InitModel(const char* OpenPoseModelPath, const char* GestureModelPath);
    Result ProcessMotionData();
    void EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg);
    Result SendImage(cv::Mat& image);
    Result OpenPresenterChannel();

    OpenPoseProcess OpenposeModel_;
    GestureProcess GestureModel_;

    int32_t deviceId_;

    aclrtContext context_;
    aclrtStream stream_;

//    uint32_t modelWidth_;
//    uint32_t modelHeight_;
    uint32_t inputDataSize_;
    void* inputBuf_;
//    aclrtRunMode runMode_;
    const char* OpenPoseModelPath_;
    const char* GestureModelPath_;

    const int FRAMES = 50;
    bool isInited_;
//    Channel* channel_;

//    aclmdlDataset* input_;
//    aclmdlDataset* output_;
    std::shared_ptr<EngineTransNewT> motion_data_new = std::make_shared<EngineTransNewT>();
};