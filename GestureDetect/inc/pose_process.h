
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
    Result Preprocess(shared_ptr<ImageDesc>& imageData, const std::string& imageFile);
  //  Result Inference(aclmdlDataset*& openposeOutput,std::vector<cv::Mat>& chwImage);
    Result Postprocess(aclmdlDataset*& openposeOutput, std::shared_ptr<EngineTransNewT> motion_data_new); // * or &
    void ProcessMotionData();




private:
    // OpenPose model: (1,3,120,160) -- NCHW
    uint32_t modelHeight_ = 120;
    uint32_t modelWidth_ = 160;


};
