#include "gesture_process.h"

#include <bits/stdint-uintn.h>
#include <cmath>
#include <fstream>

#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

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


Result GestureProcess::Inference(aclmdlDataset*& inferenceOutput, float motion_data[1][3][FRAME_LENGTH][18]) {

//    cout<<"in gesture inferences..."<<endl;
    uint32_t buffer_size = 3 * FRAME_LENGTH * 18 * sizeof(float);

//    Utils::write_motion_data(motion_data);
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
    cout<<"In gesture process inference"<<endl;
    return SUCCESS;
}

int GestureProcess::Postprocess(aclmdlDataset* modelOutput,cv::Mat &image) {

    uint32_t ges_size=0;
    float* ges =(float*)GetInferenceOutputItem(ges_size,modelOutput,0);
//    cout<<"-------------------------------------\n";
//    for(int i=0;i<24;i++)
//        cout<<"i: "<<ges[i]<<endl;
//    cout<<"-------------------------------------\n";
    int max_id=max_element(ges,ges+24)-ges;
    cout<<"gesture: "<<gesture_labels[max_id]<<endl;

    return max_id;
}


