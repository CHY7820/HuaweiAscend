/**
* @file model_process.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#pragma once
#include <iostream>

#include "utils.h"
#include "acl/acl.h"

/**
* ModelProcess
*/
class ModelProcess {
public:

    /**
    * @brief Constructor
    */
    ModelProcess();
//    ModelProcess(uint32_t modelId);

    /**
    * @brief Destructor
    */
    virtual ~ModelProcess();


    /**
    * @brief load model from file with mem
    * @param [in] modelPath: model path
    * @return result
    */
    Result LoadModelFromFileWithMem(const char *modelPath);

    /**
    * @brief unload model
    */
    void Unload();

    /**
    * @brief create model desc
    * @return result
    */
    Result CreateDesc();

    /**
    * @brief destroy desc
    */
    void DestroyDesc();

    /**
    * @brief create model input
    * @param [in] inputDataBuffer: input buffer
    * @param [in] bufferSize: input buffer size
    * @return result
    */
    Result CreateInput(void *inputDataBuffer, size_t bufferSize);
//    Result CreateInput(void *inputBuf1, size_t bufsize1,
//                        void* inputBuf2, size_t bufsize2);

    /**
    * @brief destroy input resource
    */
    void DestroyInput();

    /**
    * @brief create output buffer
    * @return result
    */
    Result CreateOutput();

    /**
    * @brief destroy output resource
    */
    void DestroyOutput();

    /**
    * @brief model execute
    * @return result
    */
    Result Execute();

    /**
    * @brief dump model output result to file
    */
    void DumpModelOutputResult();

    /**
    * @brief get model output result
    */
//    void OutputModelResult();

    aclmdlDataset * GetModelOutputData() { return output_; }

    void* GetInferenceOutputItem(uint32_t& itemDataSize,
    aclmdlDataset* inferenceOutput, uint32_t idx);

//    void set_runmode(aclrtRunMode runMode) { runMode_ = runMode; }
    void set_modelId(uint32_t modelId) { modelId_ = modelId; }

protected:
    uint32_t modelId_;
//    aclrtRunMode runMode_;

private:
    size_t modelMemSize_;
    size_t modelWeightSize_;
    void *modelMemPtr_;
    void *modelWeightPtr_;
    bool loadFlag_;  // model load flag
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;


};

