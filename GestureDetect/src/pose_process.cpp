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
//    cv::Mat resizedImage; //(modelWidth_,modelHeight_,CV_8UC3);
//    cv::imwrite(imageFile.substr(15),image);
    cv::resize(image, image, cv::Size(modelWidth_,modelHeight_),cv::INTER_CUBIC);

    image.convertTo(image, CV_32FC3); // uint8 -> float32
    image=image*(1/255.0)-0.5;

    //    float* input=(float*)image.data;
//    for(int i=0;i<10;i++)
//        printf("%f ",input[i]);
//
//    input=(float*)image.data;
//    cout<<endl<<"after"<<endl;
//    for(int i=0;i<10;i++)
//        printf("%f ",input[i]);

    std::vector<cv::Mat> channels;
    cv::split(image,channels);
    uint32_t channelSize=IMAGE_CHAN_SIZE_F32(modelWidth_,modelHeight_);
//    for (int i = 0; i < 3; i++) {
//        cout<<"i:"<<i<<endl;
//        float* input=(float*)channels[i].data;//.ptr<uchar>(0);
//        for(int j=100;j<110;j++)
//            printf("%.4f ",input[j]);
//        cout<<endl;
//    }
    int pos=0;
    for (int i = 0; i < 3; i++) {
//        memcpy(static_cast<float*>(poseInputBuf_) + i*channelSize/sizeof(float),(float*)channels[i].data, channelSize);// channels[i].ptr<float>(0)
        memcpy(static_cast<uint8_t*>(poseInputBuf_) + pos,(float*)channels[i].data, channelSize);
        pos+=channelSize;
        // ptr+idx: move pointer by byte
    }
    cout<<"openpose preprocess success"<<endl;
    // CHW
// cannot memcpy 3 channels together
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



Result OpenPoseProcess::Postprocess(aclmdlDataset*& modelOutput,float motion_data[1][3][FRAME_LENGTH][18]) {
    static float motion_data_old[1][3][FRAME_LENGTH][18];
    static bool flag_=false;
    if(!flag_)
    {
        memset(motion_data_old,0,sizeof(motion_data_old));
        flag_=true;
    }


    uint32_t heatmapSize = 0, pafSize=0;
    float* paf_=(float*)GetInferenceOutputItem(pafSize,modelOutput,0);
    float* heatmap_=(float*)GetInferenceOutputItem(heatmapSize,modelOutput,1);
//    cout<<"paf: "<<endl;
//    for(int i=0;i<10;i++)
//        printf("%.8f ",paf[i]);
//    cout<<endl<<"heatmap: "<<endl;
//    for(int i=0;i<10;i++)
//        printf("%.8f ",heatmap[i]);


    cv::Mat temp_mat;

    Eigen::Matrix <float, 120, 160> resized_matrix;
    Eigen::Matrix <float, 120, 160> left_matrix;
    Eigen::Matrix <float, 120, 160> right_matrix;
    Eigen::Matrix <float, 120, 160> top_matrix;
    Eigen::Matrix <float, 120, 160> bottom_matrix;
    Eigen::MatrixXd::Index maxRow, maxCol;
    vector <key_pointsT> keypoint;
    vector<key_pointsT> all_keypoints; // ASSUME one person only
    vector<float> peaks_pos;
    float max_val;
    float thre=0.3;


    // 找出18个关键点
    for (int part = 0; part < 18; part++) {


        float *v = heatmap_ + part * 300;
        // 按照列映射到Matrix
//        Eigen::Map<Eigen::MatrixXf> matmem(v, 15, 20); // matrix memory
//        Eigen::Matrix <float, 15, 20> m = matmem;
        cv::Mat heatmap(15,20,CV_32FC1,v); //  Mat::Mat(int rows, int cols, int type, constScalar& s)
        // 扩展15x20为120x160大小
//        cv::eigen2cv(m, temp_mat);
//        cv::resize(temp_mat, temp_mat, cv::Size(modelWidth_,modelHeight_), cv::INTER_CUBIC);

        cv::resize(heatmap,heatmap, cv::Size(modelWidth_,modelHeight_), cv::INTER_CUBIC);
        //        cv::GaussianBlur(temp_mat, temp_mat, cv::Size(3, 3), 5);
        cv::GaussianBlur(heatmap, heatmap, cv::Size(3, 3), 5);
        cv::cv2eigen(heatmap, resized_matrix);
        max_val=resized_matrix.maxCoeff(&maxRow,&maxCol);

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
                if (left_matrix(aa, bb) >= 0 && right_matrix(aa, bb) >= 0 && bottom_matrix(aa, bb) >= 0 && top_matrix(aa, bb) >= 0 && resized_matrix(aa, bb) > thre) {
//                    cout<<"part number: "<<part<<"found"<<endl;
                    peaks_pos.push_back(aa);
                    peaks_pos.push_back(bb);
                }
            }
        }

        float peak_score=0,peak_row=0,peak_col=0;
        int peak_num=0;
        int score=0;
        for (int i = 0; i < peaks_pos.size(); i += 2) {
            int tmp_row=peaks_pos[i];
            int tmp_col=peaks_pos[i+1];
            float tmp_score=resized_matrix(tmp_row,tmp_col);
            if(tmp_score>peak_score)
            {
                peak_score=tmp_score;
                peak_row=(float)tmp_row;
                peak_col=(float)tmp_col;
            }
//            cout<<"pos:"<<"x-"<<tmp_row<<"y-"<<tmp_col<<"score-"<<tmp_score<<endl;

        }
        key_pointsT keypoint = {
            peak_col,peak_row,peak_score
        };
        all_keypoints.push_back(keypoint);

       // ASUME ONLY ONE PERSON : Take out the best part peak


        peaks_pos.clear();

    }

    // TEST OPENPOSE
    // 生成一张纯白图，用于画出结果图
//    static int id=0;
//
//    cv::Mat out=cv::imread("../data/frames/"+to_string(id)+".jpg");//out(cv::Size(modelWidth_,modelHeight_), CV_8UC3, cv::Scalar(255, 255, 255)); // cv::Scalar(b,g,r,alpha)
//    int col=out.cols;
//    int row=out.rows;
//
//    cv::resize(out, out, cv::Size(modelWidth_,modelHeight_),cv::INTER_CUBIC);
//
//    if(id<5)
//    {
//        for(int i=0;i<all_keypoints.size();i++)
//        {
//            key_pointsT kpt=all_keypoints[i];
//            cv::Point p(kpt.x,kpt.y); // x is in the col direction!
//            cv::circle(out, p, kpt.score*5, cv::Scalar(0, 255, 255));
//        }
//        cv::resize(out,out,cv::Size(col,row),cv::INTER_CUBIC);
//        string imgname="../out/output/";
//        cv::imwrite(imgname+"result_"+to_string(id)+".jpg", out);
//        id++;
//    }




    float x[18]={0},y[18]={0},s[18]={0};
//    for(int k=0;k<18;k+=3)
    for(int k=0;k<18;k++)
    {
        if(k>all_keypoints.size())
            continue;
        else{
            x[k]=all_keypoints[k].x;
            y[k]=all_keypoints[k].y;
            s[k]=all_keypoints[k].score;
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



