/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File sample_process.h
* Description: handle acl resource
*/
#pragma once
#include "utils.h"
#include "acl/acl.h"
#include "model_process.h"
#include "gesture_process.h"
#include "dvpp_process.h"
#include <map>
#include <memory>

using namespace std;
//int LAST_GES = -1;
/**
* ClassifyProcess
*/
class GestureDetect {
public:
    GestureDetect(const char* OpenPose_modelPath, const char* Gesture_modelPath,
                 uint32_t modelWidth, uint32_t modelHeight);
    ~GestureDetect();

    Result Init();
    Result Preprocess(ImageData& resizedImage, ImageData& srcImage);
    Result OpenPoseInference(aclmdlDataset*& inferenceOutput, ImageData& resizedImage);
    Result GestureInference(aclmdlDataset*& inferenceOutput, std::shared_ptr<EngineTransNewT> motion_data_new);
    Result Postprocess(ImageData& image, aclmdlDataset* modelOutput, std::shared_ptr<EngineTransNewT> motion_data_new, int &success_num);
    Result PostGestureProcess(aclmdlDataset* modelOutput);
private:
    Result InitResource();
    Result InitModel(const char* omModelPath_openpose, const char* omModelPath_gesture);
    Result CreateImageInfoBuffer();
    void* GetInferenceOutputItem(uint32_t& itemDataSize,
                                 aclmdlDataset* inferenceOutput,
                                 uint32_t idx);
    void DrowBoundBoxToImage(vector<BBox>& detectionResults,
                             const string& origImagePath);
    void DestroyResource();

public:
    map<int, int>gesture_map;

private:
    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;
    uint32_t imageInfoSize_;
    void* imageInfoBuf_;
    ModelProcess model_;
    GestureProcess GestureModel_;

    const char* modelPath_OpenPose;
    const char* modelPath_Gesture;
    uint32_t modelWidth_;
    uint32_t modelHeight_;
    uint32_t inputDataSize_;
    DvppProcess dvpp_;
    aclrtRunMode runMode_;

    bool isInited_;


};

