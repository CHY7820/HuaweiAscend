#pragma once
#include "model_process.h"

class GestureProcess : public ModelProcess
{
public:
    GestureProcess();
//    GestureProcess(uint32_t modelId);
    Result Inference(aclmdlDataset*& inferenceOutput, std::shared_ptr<EngineTransNewT> motion_data_new);
//    Result Postprocess(aclmdlDataset* modelOutput);
//    ~GestureProcess();
};

