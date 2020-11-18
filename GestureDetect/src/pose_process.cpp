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
int IMG_NUM;
int file_num = 0;
int LAST_GES = -1;
//float temp_key_points[3][18] = {-1};

//
//int limbSeq[19][2] = {{2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10},
//{10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 1}, {1, 15}, {15, 17},
//{1, 16}, {16, 18}, {3, 17}, {6, 18}};
//
//int mapIdx[19][2] = {{31, 32}, {39, 40}, {33, 34}, {35, 36}, {41, 42}, {43, 44}, {19, 20}, {21, 22},
//{23, 24}, {25, 26}, {27, 28}, {29, 30}, {47, 48}, {49, 50}, {53, 54}, {51, 52},
//{55, 56}, {37, 38}, {45, 46}};
int BAD_NUM;
float TOTAL_RIGHT, TOTAL_LEFT, TOTAL_TOP, TOTAL_BOTTOM;
float TEMP_LEFT, TEMP_RIGHT, TEMP_BOTTOM, TEMP_TOP;
//std::shared_ptr<EngineTransNewT> motion_data_old = std::make_shared<EngineTransNewT>();

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


// 自定义排序规则
bool cmp2(connectionT a,connectionT b) {
    return a.score>b.score;
}

Result OpenPoseProcess::Preprocess(const std::string& imageFile)
{

    cout<<"in openpose preprocess"<<endl;
    // cv读进来的是BGR图像，通道是HWC
    cout<<imageFile<<endl;
    cv::Mat image = cv::imread(imageFile, CV_LOAD_IMAGE_COLOR); // BGR image:
    if (image.empty()) {
        ERROR_LOG("Read image %s failed", imageFile.c_str());
        return FAILED;
    }

    //resize image to model size
    cv::Mat resizedImage; //(modelWidth_,modelHeight_,CV_8UC3);
//    cv::imwrite(imageFile.substr(15),image);
    cv::resize(image, resizedImage, cv::Size(modelWidth_,modelHeight_));
//    cv::resize(image, resizedImage, cv::Size(160,120));
//    cv::imwrite("rszd"+imageFile.substr(15),resizedImage);
    resizedImage.convertTo(resizedImage, CV_32FC3); // uint8 -> float32
    resizedImage=resizedImage/255.0;

    std::vector<cv::Mat> channels;
    cv::split(resizedImage,channels);
    uint32_t channelSize=IMAGE_CHAN_SIZE_F32(modelWidth_,modelHeight_);
    for (int i = 0; i < 3; i++) {
        memcpy(static_cast<float*>(poseInputBuf_) + i*channelSize/sizeof(float), channels[i].ptr<float>(0), channelSize);
        // ptr+idx: move pointer by byte
    }
    cout<<"openpose preprocess success"<<endl;
    // CHW
// cannot memcpy 3 channels together
    return SUCCESS; //nn_width; // resized_width
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



Result OpenPoseProcess::Postprocess(aclmdlDataset*& modelOutput,float motion_data[1][3][FRAME_LENGTH][18]) {

    static float motion_data_old[1][3][FRAME_LENGTH][18];

    uint32_t heatmapSize = 0, pafSize=0;
    float* paf=(float*)GetInferenceOutputItem(pafSize,modelOutput,0);
    float* heatmap=(float*)GetInferenceOutputItem(heatmapSize,modelOutput,1);
    cv::Mat temp_mat;

    Eigen::Matrix <float, 120, 160> resized_matrix;

    Eigen::Matrix <float,120, 160> left_matrix;
    Eigen::Matrix <float,120, 160> right_matrix;
    Eigen::Matrix <float, 120, 160> top_matrix;
    Eigen::Matrix <float,120, 160> bottom_matrix;
    Eigen::MatrixXd::Index maxRow, maxCol;
    vector <key_pointsT> keypoint;
    vector<key_pointsT> all_keypoints; // ASSUME one person only
    vector<float> peaks_pos;
    float max_val;


    float thre=0.1;
    // 找出18个关键点
    for (int part = 0; part < 18; part++) {
        float score=0,peak_row,peak_col; // hehe

        float *v = heatmap + part * 300;
        // 按照列映射到Matrix
        Eigen::Map<Eigen::MatrixXf> matmem(v, 15, 20); // matrix memory
        // m是16*16的矩阵
        Eigen::Matrix <float, 15, 20> m = matmem;


        // 先找16x16的最大数值的下标，temp_aa是最大值的数值
//        max_val = m.maxCoeff(&maxRow_F, &maxCol_F);
//        std::cout<<"16x16 max value: "<<max_val<<std::endl;

        // 最大值都不大于0.1，认为本帧图像无效
        // 特殊处理掉,todo
//        if (max_val < thre) {
//            if_valid = false;
//            cout<<max_val<<endl;
//            break;
//        }
        // 扩展15x20为120x160大小
        cv::eigen2cv(m, temp_mat);
        cv::resize(temp_mat, temp_mat, cv::Size(modelWidth_,modelHeight_), cv::INTER_CUBIC);
        cv::GaussianBlur(temp_mat, temp_mat, cv::Size(3, 3), 5);
        cv::cv2eigen(temp_mat, resized_matrix);
        max_val=resized_matrix.maxCoeff(&maxRow,&maxCol);
//        cout<<"max_val: "<<max_val<<endl;

        // 获取矩阵左移的矩阵
        left_matrix.leftCols(159) = resized_matrix.rightCols(159);
        left_matrix.col(159) = Eigen::MatrixXf::Zero(120, 1);
        // 右移
        right_matrix.rightCols(159) = resized_matrix.leftCols(159);
        right_matrix.col(0) = Eigen::MatrixXf::Zero(120, 1);
        // 上移
        top_matrix.topRows(119) =  resized_matrix.bottomRows(119);
        top_matrix.row(119) = Eigen::MatrixXf::Zero(1, 160);
        // 下移
        bottom_matrix.bottomRows(119) =  resized_matrix.topRows(119);
        bottom_matrix.row(0) = Eigen::MatrixXf::Zero(1, 160);
        //像素减去它右边像素的值
        left_matrix = resized_matrix - left_matrix;
        right_matrix = resized_matrix - right_matrix;
        top_matrix = resized_matrix - top_matrix;
        bottom_matrix = resized_matrix - bottom_matrix;

        for (int aa = 0; aa < 120; aa++) {
            for (int bb = 0; bb < 160; bb++) {
                if (left_matrix(aa, bb) > 0 && right_matrix(aa, bb) > 0 && bottom_matrix(aa, bb) > 0 && top_matrix(aa, bb) > 0 && resized_matrix(aa, bb) > thre) {
//                    cout<<"part number: "<<part<<"found"<<endl;
                    peaks_pos.push_back(aa);
                    peaks_pos.push_back(bb);
                }
            }
        }

        int peak_num=0;
        for (int i = 0; i < peaks_pos.size(); i += 2) {
            int tmp_row=peaks_pos[i];
            int tmp_col=peaks_pos[i+1];
            float tmp_score=resized_matrix(tmp_row,tmp_col);
            if(tmp_score>score)
            {
                score=tmp_score;
                peak_row=(float)tmp_row;
                peak_col=(float)tmp_col;
            }

        }

//        cout<<"score: "<<score<<endl;
        // 如果有一个部位一个点都没找到，人体缺失一个关键点，就不往下继续找了

//        if (score > thre)
//        {
            // 子矩阵中的最大值认为是一个局部最大值（一个关键点）
            // temp里面存了横纵坐标，和所有特征图找到的peak里，它是第几个找到的，all_peak_index
        key_pointsT keypoint = {
            peak_row,peak_col,score
        };
       // ASUME ONLY ONE PERSON : Take out the best part peak

        all_keypoints.push_back(keypoint);
        peaks_pos.clear();

    }

    // 生成一张纯白图，用于画出结果图
    cv::Mat out(cv::Size(modelWidth_,modelHeight_), CV_8UC3, cv::Scalar(255, 255, 255)); // cv::Scalar(b,g,r,alpha)
    for(int i=0;i<all_keypoints.size();i++)
    {
        key_pointsT kpt=all_keypoints[i];
        cv::Point p(kpt.x,kpt.y);
        cv::circle(out, p, kpt.score*10, cv::Scalar(0, 0, 255));
    }
//    cv::resize(out,out,cv::Size(modelWidth_,modelHeight_));
    cv::imwrite("../out/output/result.jpg", out);

    float x[18],y[18],s[18];
    for(int k=0;k<18;k+=3)
    {
        if(k>all_keypoints.size())
        {
            x[k]=0;y[k]=0;s[k]=0;
//            key_pointsT kpt={0,0,0};
//            all_keypoints.push_back(kpt);
        }
        else{
            x[k]=all_keypoints[k].x;
            y[k]=all_keypoints[k].y;
            s[k]=all_keypoints[k].score;
        }
    }


    // move and add at the queue tail
    memcpy(motion_data[0][0][FRAME_LENGTH-1], x, sizeof(x));
    memcpy(motion_data[0][1][FRAME_LENGTH-1], y, sizeof(y));
    memcpy(motion_data[0][2][FRAME_LENGTH-1], s, sizeof(s));
    memcpy(motion_data[0][0][0],motion_data_old[0][0][1],sizeof(x)*(FRAME_LENGTH-1));
    memcpy(motion_data[0][1][0],motion_data_old[0][1][1],sizeof(y)*(FRAME_LENGTH-1));
    memcpy(motion_data[0][2][0],motion_data_old[0][2][1],sizeof(s)*(FRAME_LENGTH-1));
    memcpy(motion_data_old,motion_data,sizeof(x)*FRAME_LENGTH*3);

    //求中心点坐标,并中心化
//    float skeleton_center[2][FRAME_LENGTH]={0.0};
//    for ( int c = 0; c < 3; c++ )
//    {
//        for ( int t = 0; t < FRAME_LENGTH; t++ )
//        //        for ( int t = 0; t < 30; t++ )
//        {
//            skeleton_center[c][t] = float((motion_data[0][c][t][1]+motion_data[0][c][t][8]+motion_data[0][c][t][11])/float(3.0));
//            for ( int v = 0; v < 18; v++ )
//            {
//                motion_data[0][c][t][v] = motion_data[0][c][t][v]-skeleton_center[c][t];
//            }
//        }
//    }
//    cout<<"sizeof motion data:"<<sizeof(motion_data)<<endl;
//    static int flag=1;
    //    flag++;
//    if(flag==100)
//    {
//        flag=-10000;
//        ofstream fout("motion_data.txt",ios::out);
//        for(int i=0;i<3;i++)
//        {
//            for(int j=0;j<FRAME_LENGTH;j++)
//            {
//                for(int k=0;k<18;k++)
//                    fout<<motion_data[0][i][j][k]<<" ";
//            }
//
//        }
//
//    }


    return SUCCESS;
}



