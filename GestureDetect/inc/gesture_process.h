#pragma once
#include "model_process.h"
#include "ascenddk/presenter/agent/channel.h"

class GestureProcess : public ModelProcess
{
public:
    GestureProcess();
    Result Inference(aclmdlDataset*& inferenceOutput, std::shared_ptr<EngineTransNewT> motion_data_new);
    Result Postprocess(const string &path,aclmdlDataset* modelOutput);
private:
    void EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg);
    Result SendImage(cv::Mat& image);
    Result OpenPresenterChannel();
    ascend::presenter::Channel* channel_ = nullptr;
};

