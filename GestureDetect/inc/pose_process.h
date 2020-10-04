
#pragma once
#include "utils.h"
#include "model_process.h"
#include "dvpp_process.h"
#include "acl/acl.h"

class OpenPoseProcess : public ModelProcess
{
public:
    OpenPoseProcess();
//    OpenPoseProcess(uint32_t modelId);
//    ~OpenPoseProcess();
    Result Preprocess(DvppProcess& dvpp, ImageData& resizedImage, ImageData& image);
    Result Inference(aclmdlDataset*& openposeOutput,ImageData& resizedImage);
    Result Postprocess(ImageData& image, aclmdlDataset* modelOutput, std::shared_ptr<EngineTransNewT> motion_data_new);

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
