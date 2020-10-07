
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
//    ~OpenPoseProcess();
    Result Preprocess(void*& inputBuf_,cv::Mat& frame,uint32_t inputDatasize);
    Result Inference(aclmdlDataset*& openposeOutput,cv::Mat& resizedImage);
    Result Postprocess(aclmdlDataset* modelOutput, std::shared_ptr<EngineTransNewT> motion_data_new);

    void ProcessMotionData();
    void set_modelsize(uint32_t modelWidth,uint32_t modelHeight) {
        modelWidth_ = modelWidth;
        modelHeight_ = modelHeight;

    }




private:
    uint32_t modelWidth_;
    uint32_t modelHeight_;


//    aclrtRunMode runMode_;


};
