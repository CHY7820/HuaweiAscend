#pragma once
#include "model_process.h"

#include <bits/stdint-uintn.h>

#include "ascenddk/presenter/agent/channel.h"

class GestureProcess : public ModelProcess
{
public:
    GestureProcess();
    Result InitModel(const char* modelPath);
    Result Inference(aclmdlDataset*& inferenceOutput, float motionData[1][3][FRAME_LENGTH][18]);
    int Postprocess(aclmdlDataset* modelOutput,cv::Mat& image);

};

