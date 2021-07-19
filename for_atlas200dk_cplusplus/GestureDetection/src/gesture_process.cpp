#include "gesture_process.h"

#include <bits/stdint-uintn.h>
#include <cmath>
#include <fstream>

#include "presenter/agent/presenter_types.h"
#include "presenter/agent/errors.h"
#include "presenter/agent/presenter_channel.h"

#include "utils.h"

using namespace std;

GestureProcess::GestureProcess() : ModelProcess() {}

Result GestureProcess::InitModel(const char* modelPath)
{
    Result ret = LoadModelFromFileWithMem(modelPath);
    if(ret!=SUCCESS) {
        ERROR_LOG("model load failed");
        return FAILED;
    }
    INFO_LOG("model load success");

    ret = CreateDesc();
    if(ret!=SUCCESS) {
        ERROR_LOG("model CreateDesc failed");
        return FAILED;
    }
    INFO_LOG("model CreateDesc success");


    ret = CreateOutput();
    if(ret!=SUCCESS) {
        ERROR_LOG("model CreateOutPut failed");
        return FAILED;
    }

    INFO_LOG("model CreateOutPut success");


    INFO_LOG("STGCN Model initial success!");
    return SUCCESS;
}


Result GestureProcess::Inference(aclmdlDataset*& inferenceOutput, float motion_data[1][FRAME_LENGTH][18][3]) {

    uint32_t buffer_size = 3 * FRAME_LENGTH * 18 * sizeof(float);

    Result ret = CreateInput((void*) motion_data, buffer_size);
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateInput failed");
        return FAILED;
    }
    INFO_LOG("model CreateInput success");

    ret = Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GetModelOutputData();
    return SUCCESS;
}

int GestureProcess::Postprocess(aclmdlDataset* modelOutput) {
    // output the highest probability gesture id

    uint32_t ges_size = 0;
    int ges_num = 4;
    float* ges = (float*)GetInferenceOutputItem(ges_size, modelOutput, 0);
    int maxPos = max_element(ges, ges + ges_num) - ges;
    for (int i = 0; i < ges_num; i++)
        cout << "i: " << i << " score: " << ges[i] << endl;
    float total = 0;
    for (int i = 0; i < ges_num; i++)
        total += exp(ges[i]);
    for (int i = 0; i < ges_num; i++) {
      ges[i] = exp(ges[i]) / total;
      cout << gesture_labels[i] << ": " << ges[i] << endl;
    }
    if (ges[maxPos] > 0.8)
      return maxPos;

    return ges_num;
}
