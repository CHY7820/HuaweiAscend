
#pragma once
#include "utils.h"
#include "model_process.h"
#include "acl/acl.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cstdint>

class OpenPoseProcess : public ModelProcess
{
public:
    OpenPoseProcess();
    Result Preprocess(cv::Mat& srcImage,cv::Mat& dstImage);
    Result Inference(aclmdlDataset*& openposeOutput,cv::Mat& frame);
    Result Postprocess(aclmdlDataset*& openposeOutput, std::shared_ptr<EngineTransNewT> motion_data_new); // * or &
    void ProcessMotionData();




private:
    // OpenPose model: (1,3,184,248) -- NCHW
    uint32_t modelHeight_ = 120;//184;
    uint32_t modelWidth_ = 160;//248;
    uint32_t inputDataSize_ = 120*160*3; // in byte

    // for openpose model taking input shape 128*128, output shape is 16*16

};
