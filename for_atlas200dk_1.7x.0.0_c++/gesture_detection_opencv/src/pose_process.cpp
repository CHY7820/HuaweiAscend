//
// Created by mind on 10/2/20.
//
#include "model_process.h"
#include "utils.h"
#include "pose_process.h"
#include "acl/acl.h"
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <cmath>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"
#include "ascenddk/presenter/agent/presenter_channel.h"

#include<fstream>
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

int find_index(vector<int>::iterator begin, vector<int>::iterator end, int element){
    auto temp = begin;
    while(temp != end){
        if(*temp == element){
            return element;
        }
        temp += 1;
    }
    return -1;
}

Result OpenPoseProcess::Preprocess(string& imageFile)
{
    cout<<"in openpose preprocess"<<endl;
    // cv读进来的是BGR图像，通道是HWC
    cv::Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR); // BGR image:

    if (image.empty()) {
        ERROR_LOG("Read image %s failed", imageFile.c_str());
        return FAILED;
    }


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
        // ptr+idx: move pointer by byte
    }
    cout<<"openpose preprocess success"<<endl;
    // CHW
    return SUCCESS;
}

Result OpenPoseProcess::Preprocess(cv::Mat image)
{

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
        // ptr+idx: move pointer by byte
    }
    cout<<"openpose preprocess success"<<endl;
    // CHW
    return SUCCESS;
}


Result OpenPoseProcess::Inference(aclmdlDataset*& inferenceOutput) {
    Result ret = Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GetModelOutputData();

    return SUCCESS;
}



Result OpenPoseProcess::Postprocess(aclmdlDataset*& openposeOutput,float motion_data[1][3][FRAME_LENGTH][18]) {

    // note: possprocess assume there is only one person

    static float motion_data_old[1][3][FRAME_LENGTH][18];
    static bool flag_=false;
    if(!flag_)
    {
        memset(motion_data_old,0,sizeof(motion_data_old));
        flag_=true;
    }

    uint32_t heatmapSize = 0;
    float* heatmap_=(float*)GetInferenceOutputItem(heatmapSize,openposeOutput,1);
    Eigen::Matrix <float, 15, 20> resized_matrix;
    Eigen::MatrixXd::Index maxRow, maxCol;
    vector<key_pointsT> all_keypoints;
    float max_val=0.0;
    float threshold=0.4;

    for (int part = 0; part < 18; part++) {
        float *v = heatmap_ + part * 300;
        cv::Mat heatmap(15,20,CV_32FC1,v); //  Mat::Mat(int rows, int cols, int type, constScalar& s)
        cv::cv2eigen(heatmap, resized_matrix);
        max_val=resized_matrix.maxCoeff(&maxRow,&maxCol);
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
    float x[18]={0},y[18]={0},s[18]={0};
    for(int k=0;k<18;k++)
    {
        float tmp = all_keypoints[k].score;
        if(tmp<0.1) continue;
        else
        {
            x[k] = all_keypoints[k].x/modelWidth_-0.5;
            y[k] = 0.5-all_keypoints[k].y/modelHeight_;
            s[k] = tmp;
        }
    }

//    for(int i=0;i<18;i++)
//        cout<<"x: "<<x[i]<<"y: "<<y[i]<<"s: "<<s[i]<<endl;

    // dump pose result to motion data from tail to head
    memcpy(motion_data[0][0][FRAME_LENGTH-1], x, sizeof(x)); // add new pose data to the tail
    memcpy(motion_data[0][1][FRAME_LENGTH-1], y, sizeof(y));
    memcpy(motion_data[0][2][FRAME_LENGTH-1], s, sizeof(s));
    memcpy(motion_data[0][0][0],motion_data_old[0][0][1],sizeof(x)*(FRAME_LENGTH-1)); // move out the data at the head
    memcpy(motion_data[0][1][0],motion_data_old[0][1][1],sizeof(y)*(FRAME_LENGTH-1));
    memcpy(motion_data[0][2][0],motion_data_old[0][2][1],sizeof(s)*(FRAME_LENGTH-1));
    memcpy(motion_data_old,motion_data,sizeof(x)*FRAME_LENGTH*3); // update old data


    return SUCCESS;
}




