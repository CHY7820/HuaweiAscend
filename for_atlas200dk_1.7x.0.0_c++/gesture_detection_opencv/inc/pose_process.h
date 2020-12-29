#pragma once
#include "utils.h"
#include "model_process.h"
#include "acl/acl.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <cstdint>
#define modelWidth_ 160
#define modelHeight_ 120

class OpenPoseProcess : public ModelProcess
{
public:
    OpenPoseProcess();

    Result InitModel(const char* modelPath);
    Result Preprocess(const std::string& imageFile);
    Result Inference(aclmdlDataset*& openposeOutput);
    Result Postprocess(aclmdlDataset*& openposeOutput, float motion_data[1][3][FRAME_LENGTH][18]); // * or &
    void ProcessMotionData();


private:
    // OpenPose model: (1,3,120,160) -- NCHW
    // (1,160,128,3) -- NHWC
//    const int modelHeight_;
//    const int modelWidth_ ;
    void* poseInputBuf_;
    uint32_t poseInputBufSize_;

};
