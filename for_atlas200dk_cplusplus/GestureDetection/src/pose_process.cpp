//
// Created by mind on 10/2/20.
//

#include "acl/acl.h"
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <cmath>
#include<fstream>


#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "presenter/agent/presenter_channel.h"

#include "pose_process.h"
#include "model_process.h"
#include "utils.h"


using namespace std;


OpenPoseProcess::OpenPoseProcess() : poseInputBuf_(nullptr), ModelProcess()
{
    poseInputBufSize_=RGB_IMAGE_SIZE_F32(modelWidth_,modelHeight_);

}

Result OpenPoseProcess::InitModel(const char* modelPath)
{
    Result ret = LoadModelFromFileWithMem(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("model load failed");
        return FAILED;
    }
    INFO_LOG("model load success");

    ret = CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateDesc failed");
        return FAILED;
    }
    INFO_LOG("model CreateDesc success");

    // register memory on the device
    aclrtMalloc(&poseInputBuf_, (size_t)(poseInputBufSize_), ACL_MEM_MALLOC_HUGE_FIRST);
    if (poseInputBuf_ == nullptr) {
        ERROR_LOG("Acl malloc image buffer failed.");
        return FAILED;
    }

    ret = CreateInput(poseInputBuf_, poseInputBufSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateInput failed");
        return FAILED;
    }
    INFO_LOG("model CreateInput success");

    ret = CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateOutPut failed");
        return FAILED;
    }

    INFO_LOG("model CreateOutPut success");

    INFO_LOG("OpenPose Model initial success!");

    return SUCCESS;

}


Result OpenPoseProcess::InitModel(const char* modelPath,DvppProcess& dvpp)
{
    // overload of InitModel

    Result ret = LoadModelFromFileWithMem(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("model load failed");
        return FAILED;
    }
    INFO_LOG("model load success");

    ret = CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateDesc failed");
        return FAILED;
    }
    INFO_LOG("model CreateDesc success");

    ret = CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateOutPut failed");
        return FAILED;
    }

    INFO_LOG("model CreateOutPut success");

    dvpp_ = dvpp;

    INFO_LOG("OpenPose Model initial success!");

    return SUCCESS;

}


Result OpenPoseProcess::Preprocess(string& imageFile)
{
    // preprocess of OpenPose model
    // read image from files and normalize, convert to NCHW format, and put into OpenPose input buffer
    cv::Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR);

    if (image.empty()) {
        ERROR_LOG("Read image %s failed", imageFile.c_str());
        return FAILED;
    }

    cv::resize(image, image, cv::Size(modelWidth_,modelHeight_),cv::INTER_CUBIC);

    image.convertTo(image, CV_32FC3); // uint8 -> float32
    image=image*(1/255.0)-0.5; // normalize

    std::vector<cv::Mat> channels;
    cv::split(image,channels);
    uint32_t channelSize=IMAGE_CHAN_SIZE_F32(modelWidth_,modelHeight_);

    int pos=0;
    for (int i = 0; i < 3; i++) {
        memcpy(static_cast<uint8_t*>(poseInputBuf_) + pos,(float*)channels[i].data, channelSize);
        pos+=channelSize;
        // ptr+idx: move pointer by byte
    }
    return SUCCESS;
}

Result OpenPoseProcess::Preprocess(cv::Mat image)
{
    // overload of Preprocess
    // process directly opencv format image
    cv::resize(image, image, cv::Size(modelWidth_,modelHeight_),cv::INTER_CUBIC);
    image.convertTo(image, CV_32FC3); // uint8 -> float32
    image=image*(1/255.0)-0.5;

    std::vector<cv::Mat> channels;
    cv::split(image,channels);
    uint32_t channelSize=IMAGE_CHAN_SIZE_F32(modelWidth_,modelHeight_);

    int pos=0;
    for (int i = 0; i < 3; i++) {
        memcpy(static_cast<uint8_t*>(poseInputBuf_) + pos,(float*)channels[i].data, channelSize);
        pos+=channelSize;
    }
    return SUCCESS;
}

Result OpenPoseProcess::Preprocess(ImageData& resizedImage,ImageData& srcImage)
{
    // overload of Preprocess
    // use dvpp to resize input image data
    Result ret = dvpp_.Resize(resizedImage, srcImage, modelWidth_, modelHeight_);
    if (ret == FAILED) {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }

    return SUCCESS;
}

Result OpenPoseProcess::Inference(aclmdlDataset*& inferenceOutput) {

    Result ret = CreateInput(poseInputBuf_, poseInputBufSize_);
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
Result OpenPoseProcess::Inference(aclmdlDataset*& inferenceOutput, ImageData& resizedImage) {

    // overload of Inference

    Result ret = CreateInput(resizedImage.data.get(),resizedImage.size);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    ret = Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GetModelOutputData();

    return SUCCESS;
}


Result OpenPoseProcess::Postprocess(aclmdlDataset*& openposeOutput,float motion_data[1][FRAME_LENGTH][18][3]) {

    // Process OpenPose output and obtain human skeleton keypoints' locations, with relavant scores

    static float motion_data_old[1][FRAME_LENGTH][18][3];
    uint32_t heatmapSize = 0;
    float* heatmap_=(float*)GetInferenceOutputItem(heatmapSize,openposeOutput,0);
    vector<key_pointsT> all_keypoints;
    float max_val=0.0;
    float threshold=0.4;

    for (int part = 0; part < 18; part++) {
        float *v = heatmap_ + part * 300;
        vector<float> vec(v, v+300);
        vector<float>::iterator biggest = max_element(vec.begin(), vec.end());
        int position=std::distance(vec.begin(), biggest);
        max_val=v[position];
        int maxRow=position / 20;
        int maxCol=position-maxRow*20;
        key_pointsT keypoints={0.0,0.0,0.0};
        if(max_val>threshold)
        {
            keypoints = {
             (float)maxCol*8,(float)maxRow*8,max_val
            };
        }
        all_keypoints.push_back(keypoints);
    }

    // normalize motion data
    float pose_data[18][3];
    memset(pose_data,0,sizeof(pose_data));
    for(int k=0;k<18;k++)
    {
        float tmp = all_keypoints[k].score;
        if(tmp<threshold) continue;
        else
        {
            pose_data[k][0] = all_keypoints[k].x/modelWidth_-0.5;
            pose_data[k][1] = 0.5-all_keypoints[k].y/modelHeight_;
            pose_data[k][2] = tmp;
        }
    }


    // dump pose result to motion data from tail to head.
    memcpy(motion_data[0][0],motion_data_old[0][1],sizeof(pose_data)*(FRAME_LENGTH-1)); // move out the data at the head
    memcpy(motion_data[0][FRAME_LENGTH-1],pose_data,sizeof(pose_data)); // update new data
    memcpy(motion_data_old,motion_data,sizeof(motion_data_old)); // update old data

    return SUCCESS;

}




