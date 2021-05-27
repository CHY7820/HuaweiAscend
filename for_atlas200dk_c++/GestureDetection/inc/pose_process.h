#pragma once

#include <cstdint>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "acl/acl.h"

#include "utils.h"
#include "model_process.h"
#include "dvpp_process.h"


class OpenPoseProcess : public ModelProcess
{
public:
    OpenPoseProcess();
    Result InitModel(const char* modelPath);
    Result InitModel(const char* modelPath,DvppProcess& dvpp);
    Result Preprocess(string& imageFile);
    Result Preprocess(cv::Mat image);
    Result Preprocess(ImageData& resizedImage,ImageData& srcImage);
    Result Inference(aclmdlDataset*& openposeOutput);
    Result Inference(aclmdlDataset*& openposeOutput, ImageData& resizedImage);
    Result Postprocess(aclmdlDataset*& openposeOutput,float motion_data[1][FRAME_LENGTH][18][3]);
    void ProcessMotionData();


private:
    void* poseInputBuf_;
    uint32_t poseInputBufSize_;
    DvppProcess dvpp_;

};
