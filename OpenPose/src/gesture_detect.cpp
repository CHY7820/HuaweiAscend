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

* File sample_process.cpp
* Description: handle acl resource
*/
#include "gesture_detect.h"
#include "PracticalSocket.h"
#include <cstddef>
#include <iostream>

#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"
#include <cmath>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>


using namespace std;
using namespace cv;


int IMG_NUM;
int file_num = 0;
int LAST_GES = -1;
float temp_key_points[2][14] = {0};
Eigen::MatrixXf left_matrix(16, 16);
Eigen::MatrixXf right_matrix(16, 16);
Eigen::MatrixXf top_matrix(16, 16);
Eigen::MatrixXf bottom_matrix(16, 16);
Eigen::MatrixXf thre(16, 16);
//Eigen::MatrixXf thre_2(10, 1);
Eigen::MatrixXf thre_result(16, 16);

int limbSeq[13][2] = {{2,3}, {2,6}, {3,4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10}, {10, 11}, {2, 12},
{12, 13}, {13, 14}, {2, 1}};

int mapIdx[19][2] = {{31,32}, {39,40}, {33,34}, {35,36}, {41,42}, {43,44}, {19,20}, {21,22}, {23,24},
{25,26}, {27,28}, {29,30}, {47,48}};
int BAD_NUM;
float TOTAL_RIGHT, TOTAL_LEFT, TOTAL_TOP, TOTAL_BOTTOM;
float TEMP_LEFT, TEMP_RIGHT, TEMP_BOTTOM, TEMP_TOP;
std::shared_ptr<EngineTransNewT> motion_data_old = std::make_shared<EngineTransNewT>();



GestureDetect::GestureDetect(const char* OpenPose_modelPath,
                           const char* Gesture_modelPath,
                           uint32_t modelWidth, 
                           uint32_t modelHeight)
:deviceId_(0), context_(nullptr), stream_(nullptr), modelWidth_(modelWidth),
 modelHeight_(modelHeight), isInited_(false){
    imageInfoSize_ = 0;
    imageInfoBuf_ = nullptr;
    modelPath_OpenPose = OpenPose_modelPath;
    modelPath_Gesture = Gesture_modelPath;

    gesture_map[0] = 3;
    gesture_map[1] = 2;
    gesture_map[2] = -1;
    gesture_map[3] = 7;
    gesture_map[4] = 19;

}

GestureDetect::~GestureDetect() {
    DestroyResource();
}

Result GestureDetect::InitResource() {
    // ACL init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
//    INFO_LOG("acl init success");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
//    INFO_LOG("open device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
//    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
//    INFO_LOG("create stream success");

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }

    return SUCCESS;
}

Result GestureDetect::InitModel(const char* omModelPath_openpose, const char* omModelPath_gesture) {
    // 加载两个om模型文件
    Result ret = model_.LoadModelFromFileWithMem(omModelPath_openpose);
    ret = GestureModel_.LoadModelFromFileWithMem(omModelPath_gesture);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = model_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = GestureModel_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = model_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    ret = GestureModel_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }
    return SUCCESS;
}

Result GestureDetect::CreateImageInfoBuffer() {
    // 128 128 128 128
//    cout << "modelWidth_ " << modelWidth_ << "modelHeight_ " << modelHeight_ << "modelWidth_ " << modelWidth_ << "modelHeight_ " << modelHeight_ << endl;
    const float imageInfo[4] = {(float)modelWidth_, (float)modelHeight_,
    (float)modelWidth_, (float)modelHeight_};
    imageInfoSize_ = sizeof(imageInfo);
    if (runMode_ == ACL_HOST)
        imageInfoBuf_ = Utils::CopyDataHostToDevice((void *)imageInfo, imageInfoSize_);
    else
        imageInfoBuf_ = Utils::CopyDataDeviceToDevice((void *)imageInfo, imageInfoSize_);
    if (imageInfoBuf_ == nullptr) {
        ERROR_LOG("Copy image info to device failed");
        return FAILED;
    }

    return SUCCESS;
}

Result GestureDetect::Init() {
    if (isInited_) {
//        INFO_LOG("Classify instance is initied already!");
        return SUCCESS;
    }

    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    ret = InitModel(modelPath_OpenPose, modelPath_Gesture);
    if (ret != SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    ret = dvpp_.InitResource(stream_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init dvpp failed");
        return FAILED;
    }

    ret = CreateImageInfoBuffer();
    if (ret != SUCCESS) {
        ERROR_LOG("Create image info buf failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}

Result GestureDetect::Preprocess(ImageData& resizedImage, ImageData& srcImage) {
    ImageData imageDevice;
    Utils::CopyImageDataToDevice(imageDevice, srcImage, runMode_);
    ImageData yuvImage;
    Result ret = dvpp_.CvtJpegToYuv420sp(yuvImage, imageDevice);
    if (ret == FAILED) {
        ERROR_LOG("Convert jpeg to yuv failed");
        return FAILED;
    }

    //resize
    ret = dvpp_.Resize(resizedImage, yuvImage, modelWidth_, modelHeight_);
    if (ret == FAILED) {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }
    
    return SUCCESS;
}

// OpenPose推理函数
Result GestureDetect::OpenPoseInference(aclmdlDataset*& inferenceOutput, ImageData& resizedImage) {
    Result ret = model_.CreateInput(resizedImage.data.get(), resizedImage.size, imageInfoBuf_, imageInfoSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    ret = model_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = model_.GetModelOutputData();

    return SUCCESS;
}

// gesture推理函数
Result GestureDetect::GestureInference(aclmdlDataset*& inferenceOutput, std::shared_ptr<EngineTransNewT> motion_data_new){

    motion_data_new->buffer_size = 2 * FRAME_LENGTH * 14 * sizeof(float);

    Result ret = GestureModel_.CreateInput((void*) motion_data_new->data, motion_data_new->buffer_size);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    ret = GestureModel_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GestureModel_.GetModelOutputData();

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

// 自定义排序规则
bool cmp2(connectionT a,connectionT b) {
    return a.score>b.score;
}

// 处理输出的结果
Result GestureDetect::Postprocess(ImageData& image, aclmdlDataset* modelOutput, std::shared_ptr<EngineTransNewT> motion_data_new, int &success_num) {

    uint32_t dataSize = 0;
//
    float* newresult = (float *)GetInferenceOutputItem(dataSize, modelOutput, 1);

    cv::Mat temp_mat;
    cv::Mat temp_mat_0;
    cv::Mat temp_mat_1;
    Eigen::Matrix <float, 128, 128> resized_matrix;
    Eigen::Matrix <float, 128, 128> score_mid_0;
    Eigen::Matrix <float, 128, 128> score_mid_1;
    Eigen::MatrixXd::Index maxRow, maxCol;
    Eigen::MatrixXd::Index maxRow_F, maxCol_F;
    Eigen::MatrixXd::Index maxRow_new, maxCol_new;
    Eigen::MatrixXd::Index temp_maxRow, temp_maxCol;
    vector <key_pointsT> one_pic_key_points;
    vector <vector<key_pointsT>> all_key_points;
    vector <float> one_pic_peaks;
    float temp_key_points[2][14];
    int all_peak_index = 0;
    float temp_aa;
    bool if_valid = true;
    // 生成一张纯白图，用于画出结果图
    Mat out1(cv::Size(128,128), CV_8UC3, cv::Scalar(255, 255, 255));
    // 找出14个关键点（动作识别序列只要前14个关键点就够了）
    for (int pic_num = 0; pic_num < 14; pic_num++){
        float *v = newresult+pic_num*256;
        // 按照列映射到Matrix
        Eigen::Map<Eigen::MatrixXf> matrQQ(v, 16, 16);

        Eigen::Matrix <float, 16, 16> m=matrQQ;
        // 先找16x16的最大数值的下标
        temp_aa = m.maxCoeff(&maxRow_F, &maxCol_F);
        // 最大值都不大于0.1，认为本帧图像无效
        if (temp_aa < 0.1){
            if_valid = false;
//            cout << "Key point Index ======================== " << pic_num << endl;
//            continue;
            break;
        }

        // 获取矩阵左移的矩阵
        left_matrix.leftCols(15) = m.rightCols(15);
        left_matrix.col(15) = Eigen::MatrixXf::Zero(16, 1);
        // 右移
        right_matrix.rightCols(15) = m.leftCols(15);
        right_matrix.col(0) = Eigen::MatrixXf::Zero(16, 1);
        // 上移
        top_matrix.topRows(15) = m.bottomRows(15);
        top_matrix.row(15) = Eigen::MatrixXf::Zero(1, 16);
        // 下移
        bottom_matrix.bottomRows(15) = m.topRows(15);
        bottom_matrix.row(0) = Eigen::MatrixXf::Zero(1, 16);
        // 寻找16x16大小的局部最大值（与128x128对应起来）
        left_matrix = m - left_matrix;
        right_matrix = m - right_matrix;
        top_matrix = m - top_matrix;
        bottom_matrix = m - bottom_matrix;

        for (int aa = 0; aa < 16; aa++){
            for(int bb = 0; bb < 16; bb++){
                if(left_matrix(aa, bb) > 0 && right_matrix(aa, bb) > 0 && bottom_matrix(aa, bb) > 0 && top_matrix(aa, bb) > 0 && m(aa, bb) > 0.1){
                    one_pic_peaks.push_back(aa);
                    one_pic_peaks.push_back(bb);
                }
            }
        }
        // 扩展16x16为128x128大小
        cv::eigen2cv(m, temp_mat);
        cv::resize(temp_mat, temp_mat, cv::Size(128, 128), cv::INTER_CUBIC);
        cv::GaussianBlur(temp_mat, temp_mat, cv::Size(3,3), 5);
        cv::cv2eigen(temp_mat, resized_matrix);

        // 根据16x16大小的图像找到的局部最大值来寻找每张128x128图的局部最大值
        for (int aa = 0; aa < one_pic_peaks.size(); aa+=2){
            temp_maxRow = one_pic_peaks[aa] * 8 - 6;
            temp_maxCol = one_pic_peaks[aa+1] * 8 - 6;
            if(temp_maxRow < 0){
                temp_maxRow = 0;
            }
            if(temp_maxCol < 0){
                temp_maxCol = 0;
            }
            if(temp_maxRow > 121){
                temp_maxRow = 121;
            }
            if(temp_maxCol > 121){
                temp_maxCol = 121;
            }
            // 128中的局部最大值下标
            // 获取一个12x12大小的子矩阵，寻找最大值
            Eigen::MatrixXf small_matrix = resized_matrix.block<12, 12>(temp_maxRow, temp_maxCol);
            temp_aa = small_matrix.maxCoeff(&maxRow_new, &maxCol_new);
            // 子矩阵中的最大值认为是一个局部最大值（一个关键点）
            key_pointsT temp = {float(temp_maxRow + maxRow_new), float(temp_maxCol + maxCol_new), all_peak_index};
            all_peak_index++;
            one_pic_key_points.push_back(temp);
        }

        // 如果有一个部位一个点都没找到，人体缺失一个关键点，就不往下继续找了
        if(one_pic_key_points.size() == 0){
            return FAILED;
        }
        // 每张图计算出的keypoints存到一个vector，然后vector再存入总的keypoints
        all_key_points.push_back(one_pic_key_points);
        one_pic_peaks.clear();
        one_pic_key_points.clear();
    }

    // 只要有一个点找不到
    if(!if_valid){
        cout << "invalid image!!" << endl;
        return FAILED;
    }

    // =======================================================================
    // ==========================寻找关键点之间的关系============================
    // =======================================================================
    // 获取第一个输出数据
    vector <connectionT> connection_candidate;
    vector <vector<connectionT>> connection_all;
    float* newresult_0 = (float *)GetInferenceOutputItem(dataSize, modelOutput, 0);

    // 遍历mapIdx
    for (int kk = 0; kk < 13; kk++){
        float *v = newresult_0 + (mapIdx[kk][0] - 19)*256;
        // 按照列映射到Matrix
        Eigen::Map<Eigen::MatrixXf> matrQQ_0(v, 16, 16);

        Eigen::Map<Eigen::MatrixXf> matrQQ_1(v + 256, 16, 16);

        Eigen::Matrix <float, 16, 16> m_0 = matrQQ_0; // score_mid
        Eigen::Matrix <float, 16, 16> m_1 = matrQQ_1; // score_mid

        // 扩展成128x128大小
        cv::eigen2cv(m_0, temp_mat_0);
        cv::eigen2cv(m_1, temp_mat_1);
        cv::resize(temp_mat_0, temp_mat_0, cv::Size(128, 128), cv::INTER_CUBIC);
        cv::resize(temp_mat_1, temp_mat_1, cv::Size(128, 128), cv::INTER_CUBIC);
        cv::GaussianBlur(temp_mat_0, temp_mat_0, cv::Size(3,3), 3);
        cv::GaussianBlur(temp_mat_1, temp_mat_1, cv::Size(3,3), 3);
        cv::cv2eigen(temp_mat_0, score_mid_0); // score_mid

        cv::cv2eigen(temp_mat_1, score_mid_1); // score_mid

        vector <key_pointsT> temp_A = all_key_points[limbSeq[kk][0] -1];

        vector <key_pointsT> temp_B = all_key_points[limbSeq[kk][1] -1];

        int LA = temp_A.size();
        int LB = temp_B.size();

        if(LA != 0 && LB != 0){
            // 寻找la中每一个点与lb中关键点之间连接的可能性
            for (int aa = 0; aa < LA; aa++){
                for(int bb = 0; bb < LB; bb++){
                    float vec[2] = {temp_B[bb].point_x - temp_A[aa].point_x, temp_B[bb].point_y - temp_A[aa].point_y};
                    float norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1]);

                    vec[0] /= norm;
                    vec[1] /= norm;
                    Eigen::Matrix <float ,10 ,2> startend;
                    startend.col(0) = Eigen::ArrayXf::LinSpaced(10, temp_A[aa].point_x, temp_B[bb].point_x);
                    startend.col(1) = Eigen::ArrayXf::LinSpaced(10, temp_A[aa].point_y, temp_B[bb].point_y);

                    Eigen::Matrix <float, 10, 1> vec_x;
                    Eigen::Matrix <float, 10, 1> vec_y;

                    // TODO transformed!!!!
                    vec_x << score_mid_0(int(round(startend(0 ,0))), int(round(startend(0 ,1)))), score_mid_0(int(round(startend(1 ,0))), int(round(startend(1 ,1)))), score_mid_0(int(round(startend(2 ,0))), int(round(startend(2 ,1))))
                    , score_mid_0(int(round(startend(3 ,0))), int(round(startend(3 ,1)))), score_mid_0(int(round(startend(4 ,0))), int(round(startend(4 ,1)))), score_mid_0(int(round(startend(5 ,0))), int(round(startend(5 ,1))))
                    , score_mid_0(int(round(startend(6 ,0))), int(round(startend(6 ,1)))), score_mid_0(int(round(startend(7 ,0))), int(round(startend(7 ,1)))), score_mid_0(int(round(startend(8 ,0))), int(round(startend(8 ,1))))
                    , score_mid_0(int(round(startend(9 ,0))), int(round(startend(9 ,1))));

                    vec_y << score_mid_1(int(round(startend(0 ,0))), int(round(startend(0 ,1)))), score_mid_1(int(round(startend(1 ,0))), int(round(startend(1 ,1)))), score_mid_1(int(round(startend(2 ,0))), int(round(startend(2 ,1))))
                    , score_mid_1(int(round(startend(3 ,0))), int(round(startend(3 ,1)))), score_mid_1(int(round(startend(4 ,0))), int(round(startend(4 ,1)))), score_mid_1(int(round(startend(5 ,0))), int(round(startend(5 ,1))))
                    , score_mid_1(int(round(startend(6 ,0))), int(round(startend(6 ,1)))), score_mid_1(int(round(startend(7 ,0))), int(round(startend(7 ,1)))), score_mid_1(int(round(startend(8 ,0))), int(round(startend(8 ,1))))
                    , score_mid_1(int(round(startend(9 ,0))), int(round(startend(9 ,1))));

                    Eigen::Matrix <float, 10, 1>score_midpts = vec_x * vec[0] + vec_y * vec[1];

                    float score_with_dist_prior = score_midpts.sum() / (10.001) + min(64/(norm-1+1e-3), 0.0);

                    if (score_with_dist_prior > 0){
                        int bad_num = 0;
                        for (int fff = 0; fff < 10; fff++){
                            if(score_midpts(fff) < 0.05){
                                bad_num++;
                            }
                        }
                        if(bad_num < 2){
                            connectionT temp_connection{aa, bb, score_with_dist_prior};
                            connection_candidate.push_back(temp_connection);
                        }
                    }
                }
            }

            // 如果有一组关键点之间一个能连接的都没有，认为无效
            if(connection_candidate.size() == 0){
                return FAILED;
            }
            // 按照连接的可能性，从大到小排序，
            sort(connection_candidate.begin(), connection_candidate.end(), cmp2);
            vector<int> temp_1;
            vector<int> temp_2;
            temp_1.push_back(33);
            temp_2.push_back(33);

            int p_i;
            int p_j;
            // 获取所有的不重复的连接关系
            vector<connectionT> one_connection;
            for (int tt = 0; tt < connection_candidate.size(); tt++){
                int i = connection_candidate[tt].point_1;
                int j = connection_candidate[tt].point_2;
                float s = connection_candidate[tt].score;

                p_i = find_index(temp_1.begin(), temp_1.end(), i);

                p_j = find_index(temp_2.begin(), temp_2.end(), j);
                if(p_i != i && p_j != j){
                    temp_1.push_back(i);
                    temp_2.push_back(j);
                    connectionT temp{temp_A[i].num, temp_B[j].num};
                    one_connection.push_back(temp);
                    if (one_connection.size() >= min(LA, LB)){
                        break;
                    }
                }
            }
            connection_candidate.clear();
            connection_all.push_back(one_connection);
            one_connection.clear();
        }
    }

    // =======================================================================
    // ========================获取正中间人的完整关键点==========================
    // =======================================================================
    int mid_index = -1;
    int min_dis = 200;
    int temp_index[14] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    // 寻找正中间人的关键点
    int key_0_size = all_key_points[0].size();
    for (int aa = 0; aa < connection_all[0].size(); aa++){
        int this_point_x = all_key_points[1][connection_all[0][aa].point_1 % key_0_size].point_x;
        if (abs(this_point_x - 64) < min_dis){
            min_dis = abs(this_point_x - 64);
            mid_index = aa;
        }
    }

    // 正中间的人的1号和2号关键点
    temp_index[1] = connection_all[0][mid_index].point_1;
    temp_index[2] = connection_all[0][mid_index].point_2;
    // 0对应的就是1号和2号之间的连接关系，上面已经找到了，所以这里从“1”开始
    for (int aa = 1; aa < 13; aa ++){
        int index_A = limbSeq[aa][0] - 1;
        int index_B = limbSeq[aa][1] - 1;
        if(temp_index[index_A] == -1){
            return FAILED;
        }
        // connection_all 里面找index_A对应的序号的连接关系
        for (int bb = 0; bb < connection_all[aa].size(); bb++){
            if(connection_all[aa][bb].point_1 == temp_index[index_A]){
                temp_index[index_B] = connection_all[aa][bb].point_2;
            }
        }
        // 有一个连接点没找到，就认为这张照片无效，直接返回
        if(temp_index[index_B] == -1){
            return FAILED;
        }
    }

    // 找到的最中间的人的14个关键点保存在temp_key_points中
    for (int aa = 0; aa < 14; aa++){
        temp_key_points[0][aa] = all_key_points[aa][temp_index[aa]].point_x;
        temp_key_points[1][aa] = all_key_points[aa][temp_index[aa]].point_y;
        cv::Point p(temp_key_points[0][aa], temp_key_points[1][aa]);//初始化点坐标为(20,20)
        cv::circle(out1, p, 1, cv::Scalar(0, 0, 0), -1);  // 画半径为1的圆(画点）
        for(int bb = aa + 1; bb < 14; bb++){
            temp_index[bb] -= all_key_points[aa].size();
        }
    }

    cv::Point x0(temp_key_points[0][0], temp_key_points[1][0]);
    cv::Point x1(temp_key_points[0][1], temp_key_points[1][1]);
    cv::Point x2(temp_key_points[0][2], temp_key_points[1][2]);
    cv::Point x3(temp_key_points[0][3], temp_key_points[1][3]);
    cv::Point x4(temp_key_points[0][4], temp_key_points[1][4]);
    cv::Point x5(temp_key_points[0][5], temp_key_points[1][5]);
    cv::Point x6(temp_key_points[0][6], temp_key_points[1][6]);
    cv::Point x7(temp_key_points[0][7], temp_key_points[1][7]);
    cv::Point x8(temp_key_points[0][8], temp_key_points[1][8]);
    cv::Point x9(temp_key_points[0][9], temp_key_points[1][9]);
    cv::Point x10(temp_key_points[0][10], temp_key_points[1][10]);
    cv::Point x11(temp_key_points[0][11], temp_key_points[1][11]);
    cv::Point x12(temp_key_points[0][12], temp_key_points[1][12]);
    cv::Point x13(temp_key_points[0][13], temp_key_points[1][13]);
    cv::line(out1, x0, x1, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x1, x2, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x2, x3, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x3, x4, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x1, x5, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x5, x6, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x6, x7, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x1, x8, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x8, x9, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x9, x10, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x1, x11, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x11, x12, cv::Scalar(255, 0, 0), 1);
    cv::line(out1, x12, x13, cv::Scalar(255, 0, 0), 1);

    cv::imwrite("../result.jpg", out1);

    memcpy(motion_data_new->data[0][0][FRAME_LENGTH-1], temp_key_points[0], sizeof(float)*14);
    memcpy(motion_data_new->data[0][1][FRAME_LENGTH-1], temp_key_points[1], sizeof(float)*14);
    // x
    memcpy(motion_data_new->data[0][0][0], motion_data_old->data[0][0][1], sizeof(float)*14*(FRAME_LENGTH - 1));
    // y
    memcpy(motion_data_new->data[0][1][0], motion_data_old->data[0][1][1], sizeof(float)*14*(FRAME_LENGTH - 1));
    memcpy(motion_data_old->data, motion_data_new->data, sizeof(float)*2*FRAME_LENGTH*14);

    //求中心点坐标,并中心化
    float skeleton_center[2][FRAME_LENGTH]={0.0};
    for ( int c = 0; c < 2; c++ )
    {
        for ( int t = 0; t < FRAME_LENGTH; t++ )
        //        for ( int t = 0; t < 30; t++ )
        {
            skeleton_center[c][t] = float((motion_data_new->data[0][c][t][1]+motion_data_new->data[0][c][t][8]+motion_data_new->data[0][c][t][11])/float(3.0));
            for ( int v = 0; v < 14; v++ )
            {
                motion_data_new->data[0][c][t][v] = motion_data_new->data[0][c][t][v]-skeleton_center[c][t];
            }
        }
    }
    success_num ++;
    return SUCCESS;
}

void* GestureDetect::GetInferenceOutputItem(uint32_t& itemDataSize, aclmdlDataset* inferenceOutput, uint32_t idx) {
//    printf("get output id %d\n", idx);
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer from model "
                  "inference output failed", idx);
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer address "
                  "from model inference output failed", idx);
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ERROR_LOG("The %dth dataset buffer size of "
                  "model inference output is 0", idx);
        return nullptr;
    }

    void* data = nullptr;
    if (runMode_ == ACL_HOST) {
        data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            ERROR_LOG("Copy inference output to host failed");
            return nullptr;
        }
    }
    else {
        data = dataBufferDev;
    }

    itemDataSize = bufferSize;
    return data;
}

void GestureDetect::DestroyResource() {
	model_.DestroyResource();
	
	aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
//    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
//    INFO_LOG("end to destroy context");
	
    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }
//    INFO_LOG("end to reset device is %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }
//    INFO_LOG("end to finalize acl");
    aclrtFree(imageInfoBuf_);
}

Result GestureDetect::PostGestureProcess(aclmdlDataset* modelOutput){
    uint32_t dataSize = 0;
    //
    float* newresult = (float *)GetInferenceOutputItem(dataSize, modelOutput, 0);
    int maxPosition = max_element(newresult, newresult+5) - newresult;

    // 人工Softmax
    float down = 1.4;
    float result_total = pow(down, newresult[0]) + pow(down, newresult[1]) + pow(down, newresult[2]) + pow(down, newresult[3]) + pow(down, newresult[4]);
    newresult[0] = pow(down, newresult[0]) / result_total;
    newresult[1] = pow(down, newresult[1]) / result_total;
    newresult[2] = pow(down, newresult[2]) / result_total;
    newresult[3] = pow(down, newresult[3]) / result_total;
    newresult[4] = pow(down, newresult[4]) / result_total;

    bool if_need_pub = false;

    if (newresult[maxPosition] >= 0.5){
        switch (maxPosition){
            case 0:
                if(newresult[maxPosition] > 0.9){
                    if (LAST_GES != 0){
                        cout << " 鼓掌 " << newresult[0] << endl;
                        cout << " 挥手 " << newresult[1] << endl;
                        cout << " 站立 " << newresult[2] << endl;
                        cout << " 双手平举 " << newresult[3] << endl;
                        cout << " 踢腿 " << newresult[4] << endl;
                        cout << "=============================鼓掌" << endl;
                        if_need_pub = true;
                        LAST_GES = 0;
                    }
                }
                break;
            case 1:
                if(newresult[maxPosition] > 0.8){
                    if (LAST_GES != 1){
                        cout << " 鼓掌 " << newresult[0] << endl;
                        cout << " 挥手 " << newresult[1] << endl;
                        cout << " 站立 " << newresult[2] << endl;
                        cout << " 双手平举 " << newresult[3] << endl;
                        cout << " 踢腿 " << newresult[4] << endl;
                        cout << "=============================挥手" << endl;
                        if_need_pub = true;
                        LAST_GES = 1;
                    }
                }
                break;
            case 2:
                if(newresult[maxPosition] > 0.5){
                    if (LAST_GES != 2){
                        cout << " 鼓掌 " << newresult[0] << endl;
                        cout << " 挥手 " << newresult[1] << endl;
                        cout << " 站立 " << newresult[2] << endl;
                        cout << " 双手平举 " << newresult[3] << endl;
                        cout << " 踢腿 " << newresult[4] << endl;
                        cout << "=============================站立" << endl;
                        if_need_pub = false;
                        LAST_GES = 2;
                    }
                }
                break;
            case 3:
                if(newresult[maxPosition] > 0.95){
                    if (LAST_GES != 3){
                        cout << " 鼓掌 " << newresult[0] << endl;
                        cout << " 挥手 " << newresult[1] << endl;
                        cout << " 站立 " << newresult[2] << endl;
                        cout << " 双手平举 " << newresult[3] << endl;
                        cout << " 踢腿 " << newresult[4] << endl;
                        cout << "=============================双手平举" << endl;
                        if_need_pub = true;
                        LAST_GES = 3;
                    }
                }
                break;
            case 4:
                if(newresult[maxPosition] > 0.9){
                    if (LAST_GES != 4){
                        cout << " 鼓掌 " << newresult[0] << endl;
                        cout << " 挥手 " << newresult[1] << endl;
                        cout << " 站立 " << newresult[2] << endl;
                        cout << " 双手平举 " << newresult[3] << endl;
                        cout << " 踢腿 " << newresult[4] << endl;
                        cout << "==============================踢腿" << endl;
                        if_need_pub = true;
                        LAST_GES = 4;
                    }
                }
                break;
            default:
                cout << "max element==================nothing  " << maxPosition << "     " << newresult[maxPosition] << endl;
                break;
        }
    }
    // Socket发送指令
    if(if_need_pub){
        const int ECHOMAX = 255;
        string servAddress = "192.168.1.xxx";             // 输入指定IP
        int echoString[3] = {gesture_map[LAST_GES], 0, 0};// 要传输的数据
        int echoStringLen = 3;                            // 数据长度
        if (echoStringLen > ECHOMAX) {                    // 检查是否超过最大长度
            cerr << "Echo string too long" << endl;
            exit(1);
        }
        unsigned short echoServPort = 00000; // 输入指定端口
        try {
            UDPSocket sock;
            // Send the string to the server
            sock.sendTo(echoString, echoStringLen, servAddress, echoServPort);
            // Receive a response
            char echoBuffer[ECHOMAX + 1];                   // Buffer for echoed string + \0
            int respStringLen;                              // Length of received response

        } catch (SocketException &e) {
            cerr << e.what() << endl;
            exit(1);
        }
    }
    return SUCCESS;
}