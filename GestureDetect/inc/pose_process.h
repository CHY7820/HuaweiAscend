
#pragma once
#include "model_process.h"
#include "acl/acl.h"

class OpenPoseProcess : public ModelProcess
{
public:
    OpenPoseProcess();
    //~OpenPoseProcess();
    Result Preprocess(ImageData& resizedImage, ImageData& image);
    Result Inference(aclmdlDataset*& openposeOutput,ImageData& resizedImage);
    Result Postprocess(ImageData& image,aclmdlDataset*& inferenceOutput...);
    void SetPara(uint32_t modelWidth,uint32 modelHeight) {
        modelWidth_ = modelWidth;
        modelHeight_ = modelHeight;
    }


private:
    uint32_t modelWidth_;
    uint32_t modelHeight_;


};
