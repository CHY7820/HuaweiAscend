#pragma once
#include "model_process.h"

#include <bits/stdint-uintn.h>

#include "ascenddk/presenter/agent/channel.h"

class GestureProcess : public ModelProcess
{
public:
    GestureProcess();
    Result InitModel(const char* modelPath);
    Result Inference(aclmdlDataset*& inferenceOutput, float motion_data[1][FRAME_LENGTH][18][3]);
    int Postprocess(aclmdlDataset* modelOutput);

};

