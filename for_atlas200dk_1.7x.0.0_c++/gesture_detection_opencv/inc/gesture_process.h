#pragma once
#include "model_process.h"
#include "ascenddk/presenter/agent/channel.h"
#include <bits/stdint-uintn.h>

class GestureProcess : public ModelProcess
{
public:
    GestureProcess();
    Result InitModel(const char* modelPath);
    Result Inference(aclmdlDataset*& inferenceOutput, float motionData[1][3][FRAME_LENGTH][18]);
    Result Postprocess(const string &path,aclmdlDataset* modelOutput);
private:
    void EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg);
    Result SendImage(cv::Mat& image);
    Result OpenPresenterChannel();
    ascend::presenter::Channel* channel_ = nullptr; // presenter server channel

};

