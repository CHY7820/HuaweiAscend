#pragma once

#include <cstdint>

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "acl/acl.h"

#include "utils.h"
#include "model_process.h"


class OpenPoseProcess : public ModelProcess
{
public:
    OpenPoseProcess();

    Result InitModel(const char* modelPath);
    Result Preprocess(string& imageFile);
    Result Preprocess(cv::Mat image);
    Result Inference(aclmdlDataset*& openposeOutput);
    Result Postprocess(aclmdlDataset*& openposeOutput,float motion_data[1][3][FRAME_LENGTH][18]);
    void ProcessMotionData();


private:
    void* poseInputBuf_;
    uint32_t poseInputBufSize_;

};
