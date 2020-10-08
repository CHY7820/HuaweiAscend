//
// Created by mind on 10/2/20.
//
#include "model_process.h"
#include "utils.h"
#include "pose_process.h"
#include "acl/acl.h"
#include <cstddef>
#include <cmath>
//#include <Eigen/Core>
//#include <Eigen/Dense>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"
#include "ascenddk/presenter/agent/presenter_channel.h"

int IMG_NUM;
int file_num = 0;
int LAST_GES = -1;
float temp_key_points[3][18] = {-1};
//Eigen::MatrixXf left_matrix(23, 16);
//Eigen::MatrixXf right_matrix(23, 16);
//Eigen::MatrixXf top_matrix(23, 16);
//Eigen::MatrixXf bottom_matrix(23, 16);
//Eigen::MatrixXf thre(23, 16);
//Eigen::MatrixXf thre_2(10, 1);
//Eigen::MatrixXf thre_result(23, 16);

//Eigen::MatrixXf thre(16, 16);
//Eigen::MatrixXf thre_result(16, 16);

int limbSeq[19][2] = {{2,3}, {2,6}, {3,4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10}, {10, 11}, {2, 12},
{12, 13}, {13, 14}, {2, 1}};

int mapIdx[19][2] = {{31,32}, {39,40}, {33,34}, {35,36}, {41,42}, {43,44}, {19,20}, {21,22}, {23,24},
{25,26}, {27,28}, {29,30}, {47,48}};
int BAD_NUM;
float TOTAL_RIGHT, TOTAL_LEFT, TOTAL_TOP, TOTAL_BOTTOM;
float TEMP_LEFT, TEMP_RIGHT, TEMP_BOTTOM, TEMP_TOP;
std::shared_ptr<EngineTransNewT> motion_data_old = std::make_shared<EngineTransNewT>();

OpenPoseProcess::OpenPoseProcess() : ModelProcess() {}
//OpenPoseProcess::OpenPoseProcess(uint32_t modelId) : ModelProcess(modelId) {};
//
//OpenPoseProcess::~OpenPoseProcess() {
//    Unload();
//    DestroyDesc();
//    DestroyInput();
//    DestroyOutput();
//}

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

Result OpenPoseProcess::Preprocess(cv::Mat& srcImage,cv::Mat& dstImage)
{
    // method 1
//    int new_width = int(srcImage.cols * 184.0 / srcImage.rows);
//    new_width += 8-new_width % 8; // padding so that H can divide 8
//    cv::resize(srcImage, dstImage, cv::Size(184, new_width), cv::INTER_CUBIC); // resize to 184 x 184*col/row
//    cv::imwrite("../out/method1.jpg",srcImage);
//    memcpy(inputBuf_, srcImage.ptr<uint8_t>(), inputDataSize);


//    std::cout<<srcImage.size()<<std::endl;
    // method 2
   // int new_width = int(srcImage.cols * 184.0 / srcImage.rows);
   // int padded_width = new_width+8-new_width % 8; // padding so that H can divide 8
    // resize and reshape from HWC to CHW
    cv::resize(srcImage, srcImage, cv::Size(modelWidth_,modelHeight_), cv::INTER_CUBIC);
    //Utils::hwc_to_chw(srcImage,dstImage);
//    std::vector<float> dst_data;
//    std::vector<cv::Mat> bgrChannels(3);
//    cv::split(dstImage, bgrChannels);
//    for (auto i = 0; i < bgrChannels.size(); i++)
//    {
//        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, 1));
//        dst_data.insert(dst_data.end(), data.begin(), data.end());
//    }
    cv::imwrite("../out/dstimg.jpg",dstImage);

//    cv::Mat singlec;
//    cv::Mat dstImage;
//    vector<cv::Mat> channels;
//    vector<cv::Mat> new_channels;
//    Eigen::Matrix <float,184,Eigen::Dynamic> resized_matrix;
//    resized_matrix.resize(184,new_width);
//
//    Eigen::Matrix <float, 184, Eigen::Dynamic> padded_matrix;
//    padded_matrix.resize(184,padded_width);
//
//    cv::split(srcImage, channels);
//
//    for (int i = 0; i < 3; i++)
//    {
//        singlec = channels.at(i);
//        if(i==1){
//            cv::imwrite("../out/singlec.jpg",singlec);
//        }

//        std::cout<<"singlec: "<<singlec.size<<std::endl;
//        cv::cv2eigen(singlec, resized_matrix);
//
//        padded_matrix.leftCols(new_width) = resized_matrix.leftCols(new_width);
//        for (int j = 0; j + new_width < padded_width; j++)
//        {
//            padded_matrix.col(j+new_width) = Eigen::MatrixXf::Ones(184, 1)*128;
////            std::cout<<padded_matrix.col(j+new_width)<<std::endl;
//        }
//        cv::Mat tmpmat;
       // std::cout<<padded_matrix.cols()<<" 1 "<<padded_matrix.rows()<<std::endl;
//        cv::eigen2cv(padded_matrix, tmpmat);
       // std::cout<<padded_matrix.cols()<<" 2 "<<padded_matrix.rows()<<std::endl;
//        std::cout<<tmpmat.rows<<std::endl;
//        new_channels.push_back(tmpmat);
//    }

//    std::cout<<new_channels[1].size()<<std::endl;
//        cv::merge(new_channels, dstImage);
//    cv::imwrite("../out/test.jpg",dstImage);

    return SUCCESS; //nn_width; // resized_width

}

Result OpenPoseProcess::Inference(aclmdlDataset*& inferenceOutput, cv::Mat& frame) {

    void* inputBuf;
    //std::cout<<frame.cols<<" "<<RGBU8_IMAGE_SIZE(frame.cols,frame.rows);
//    int32_t inputDataSize = RGBU8_IMAGE_SIZE(frame.cols,frame.rows); // datasize in byte
    aclrtMalloc(&inputBuf, inputDataSize_, ACL_MEM_MALLOC_HUGE_FIRST); // alloc memory
    if (inputBuf == nullptr) {
        ERROR_LOG("Acl malloc image buffer failed.");
        return FAILED;
    }

    memcpy(inputBuf, frame.ptr<uint8_t>(), inputDataSize_); // copy memory
    CreateInput(inputBuf,inputDataSize_); // create input: put inputBuf in input_ in model
    Result ret = Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GetModelOutputData();
    aclrtFree(inputBuf);
    return SUCCESS;
}



Result OpenPoseProcess::Postprocess(aclmdlDataset*& modelOutput, std::shared_ptr<EngineTransNewT> motion_data_new) {

    uint32_t dataSize = 0;
    // process heatmap
    float* newresult = (float *)GetInferenceOutputItem(dataSize, modelOutput, 1);
    for(int k=0;k<22;k++)
    {
        std::cout<<newresult[k]<<std::endl;
    }  // len = 19


    cv::Mat temp_mat;
    cv::Mat temp_mat_0;
    cv::Mat temp_mat_1;
    Eigen::Matrix <float, 120, 160> resized_matrix;
    Eigen::Matrix <float, 120, 160> score_mid_0;
    Eigen::Matrix <float, 120, 160> score_mid_1;

    //or:
//    Eigen::MatrixXf resized_matrix(modelWidth_,modelHeight_); // typedef Eigen::Matrix <float, Eigen::Dynamic, Eigen::Dynami>  MatrixXf
//    Eigen::MatrixXf score_mid_0(modelWidth_,modelHeight_);
//    Eigen::MatrixXf score_mid_1(modelWidth_,modelHeight_);
    Eigen::Matrix <float,15,20> left_matrix;
    Eigen::Matrix <float,15,20> right_matrix;
    Eigen::Matrix <float, 15, 20> top_matrix;
    Eigen::Matrix <float,15, 20> bottom_matrix;
    Eigen::MatrixXd::Index maxRow, maxCol;
    Eigen::MatrixXd::Index maxRow_F, maxCol_F;
    Eigen::MatrixXd::Index maxRow_new, maxCol_new;
    Eigen::MatrixXd::Index temp_maxRow, temp_maxCol;
    vector <key_pointsT> one_pic_key_points;
    vector <vector<key_pointsT>> all_key_points;
    vector <float> one_pic_peaks;
    float temp_key_points[3][18];
    int all_peak_index = 0;
    float temp_aa;
    bool if_valid = true;

    // 生成一张纯白图，用于画出结果图

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    cv::Mat out1(cv::Size(modelWidth_,modelHeight_), CV_8UC3, cv::Scalar(255, 255, 255)); // cv::Scalar(b,g,r,alpha)
    // 找出18个关键点
    for (int pic_num = 0; pic_num < 18; pic_num++) {

        float *v = newresult + pic_num * 300;
        // 按照列映射到Matrix
        //martQQ是

        Eigen::Map<Eigen::MatrixXf> matrQQ(v, 15, 20);
        // m是16*16的矩阵
        Eigen::Matrix <float, 15, 20> m = matrQQ;
        // 先找16x16的最大数值的下标，temp_aa是最大值的数值
        temp_aa = m.maxCoeff(&maxRow_F, &maxCol_F);

        // 最大值都不大于0.1，认为本帧图像无效
        // 特殊处理掉,todo
//        if (temp_aa < 0.1) {
//            if_valid = false;
//            //            cout << "Key point Index ======================== " << pic_num << endl;
//            //            continue;
//            break;
//        }

        // 获取矩阵左移的矩阵

        left_matrix.leftCols(19) = m.rightCols(19);
        left_matrix.col(20) = Eigen::MatrixXf::Zero(15, 1);
        // 右移
        right_matrix.rightCols(19) = m.leftCols(19);
        right_matrix.col(0) = Eigen::MatrixXf::Zero(15, 1);
        // 上移
        top_matrix.topRows(14) = m.bottomRows(14);
        top_matrix.row(15) = Eigen::MatrixXf::Zero(1, 20);
        // 下移
        bottom_matrix.bottomRows(14) = m.topRows(14);
        bottom_matrix.row(0) = Eigen::MatrixXf::Zero(1, 20);
        // 寻找16x16大小的局部最大值
        //像素减去它右边像素的值
        left_matrix = m - left_matrix;
        right_matrix = m - right_matrix;
        top_matrix = m - top_matrix;
        bottom_matrix = m - bottom_matrix;

        for (int aa = 0; aa < 15; aa++) {
            for (int bb = 0; bb < 20; bb++) {
                if (left_matrix(aa, bb) > 0 && right_matrix(aa, bb) > 0 && bottom_matrix(aa, bb) > 0 && top_matrix(aa, bb) > 0 && m(aa, bb) > 0.1) {
                    // one_pic_peaks是vector<float>，内部存着一张特征图内的peak的坐标
                    one_pic_peaks.push_back(aa);
                    one_pic_peaks.push_back(bb);
                }
            }
        }
        // 扩展16x16为128x128大小
        cv::eigen2cv(m, temp_mat);
        cv::resize(temp_mat, temp_mat, cv::Size(modelWidth_,modelHeight_), cv::INTER_CUBIC);
        cv::GaussianBlur(temp_mat, temp_mat, cv::Size(3, 3), 5);
        cv::cv2eigen(temp_mat, resized_matrix);

        // 根据16x16大小的图像找到的局部最大值来寻找每张128x128图的局部最大值
        for (int aa = 0; aa < one_pic_peaks.size(); aa += 2) {
            // 步长为8，6是啥
            temp_maxRow = one_pic_peaks[aa] * 8 - 6;
            temp_maxCol = one_pic_peaks[aa + 1] * 8 - 6;
            if (temp_maxRow < 0) {
                temp_maxRow = 0;
            }
            if (temp_maxCol < 0) {
                temp_maxCol = 0;
            }
            if (temp_maxRow > 113) {
                temp_maxRow = 113;
            }
            if (temp_maxCol > 153) {
                temp_maxCol = 153;
            }
            // 128中的局部最大值下标
            // 获取一个12x12大小的子矩阵，寻找最大值
            // 起始于temp_maxRow，temp_maxCol
            Eigen::MatrixXf small_matrix = resized_matrix.block<12, 12>(temp_maxRow, temp_maxCol);
            temp_aa = small_matrix.maxCoeff(&maxRow_new, &maxCol_new);
            // 子矩阵中的最大值认为是一个局部最大值（一个关键点）
            // temp里面存了横纵坐标，和所有特征图找到的peak里，它是第几个找到的，all_peak_index
            key_pointsT temp = { float(temp_maxRow + maxRow_new), float(temp_maxCol + maxCol_new),temp_aa ,all_peak_index };
            all_peak_index++;
            // temp是key_pointsT，one_pic_key_points是它的向量，这是一张特征图的
            one_pic_key_points.push_back(temp);
        }

        // 如果有一个部位一个点都没找到，人体缺失一个关键点，就不往下继续找了
        //这里要改
       // if (one_pic_key_points.size() == 0) {

         //   return FAILED;
       // }
        // 每张图计算出的keypoints存到一个vector，然后vector再存入总的keypoints
        //all_key_points是vector<key_pointsT>的vector
        all_key_points.push_back(one_pic_key_points);
        one_pic_peaks.clear();
        one_pic_key_points.clear();
    }

    // 只要有一个点找不到
  //  if (!if_valid) {
      //  cout << "invalid image!!" << endl;
      //  return FAILED;
   // }
    // 到此为止，已经有了所有的关键点了

    // =======================================================================
    // ==========================寻找关键点之间的关系============================
    // =======================================================================
    // 获取第一个输出数据，这个是表示连接是输出
    vector <connectionT> connection_candidate;
    vector <vector<connectionT>> connection_all;
    float* newresult_0 = (float *)GetInferenceOutputItem(dataSize, modelOutput, 0);
	vector<int> missing_con;

    // 遍历mapIdx
    for (int kk = 0; kk < 19; kk++) {
        // 进入这个for之后就是两种点之间关系的寻找了
        float *v = newresult_0 + (mapIdx[kk][0] - 19) * 300;
        // 按照列映射到Matrix
        Eigen::Map<Eigen::MatrixXf> matrQQ_0(v, 15, 20);
        // ???
        Eigen::Map<Eigen::MatrixXf> matrQQ_1(v + 300, 15, 20);

        Eigen::Matrix <float, 15, 20> m_0 = matrQQ_0; // score_mid
        Eigen::Matrix <float, 15, 20> m_1 = matrQQ_1; // score_mid

        // 扩展成128x128大小
        cv::eigen2cv(m_0, temp_mat_0);
        cv::eigen2cv(m_1, temp_mat_1);
        //temp开头的是cv格式的
        cv::resize(temp_mat_0, temp_mat_0, cv::Size(120, 160), cv::INTER_CUBIC);
        cv::resize(temp_mat_1, temp_mat_1, cv::Size(120, 160), cv::INTER_CUBIC);
        cv::GaussianBlur(temp_mat_0, temp_mat_0, cv::Size(3, 3), 3);
        cv::GaussianBlur(temp_mat_1, temp_mat_1, cv::Size(3, 3), 3);
        cv::cv2eigen(temp_mat_0, score_mid_0); // score_mid
        cv::cv2eigen(temp_mat_1, score_mid_1); // score_mid
        // score是eigen格式的，limbseq里面是关节index
        // temp_A和temp_B的代表有连接的关键点,其中的一组
        // 是vector <key_pointsT>
        vector <key_pointsT> temp_A = all_key_points[limbSeq[kk][0] - 1];
        vector <key_pointsT> temp_B = all_key_points[limbSeq[kk][1] - 1];

        int LA = temp_A.size();
        int LB = temp_B.size();

        if (LA != 0 && LB != 0) {
            // 寻找la中每一个点与lb中关键点之间连接的可能性
            for (int aa = 0; aa < LA; aa++) {
                for (int bb = 0; bb < LB; bb++) {
                    //这里应该在计算两个点之间的距离
                    float vec[2] = { temp_B[bb].point_x - temp_A[aa].point_x, temp_B[bb].point_y - temp_A[aa].point_y };
                    float norm = sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
                    //归一化
                    vec[0] /= norm;
                    vec[1] /= norm;

                    Eigen::Matrix <float, 10, 2> startend;
                    // 在两个关键点之间等间隔地取点
                    startend.col(0) = Eigen::ArrayXf::LinSpaced(10, temp_A[aa].point_x, temp_B[bb].point_x);
                    startend.col(1) = Eigen::ArrayXf::LinSpaced(10, temp_A[aa].point_y, temp_B[bb].point_y);

                    Eigen::Matrix <float, 10, 1> vec_x;
                    Eigen::Matrix <float, 10, 1> vec_y;

                    // TODO transformed!!!!
                    vec_x << score_mid_0(int(round(startend(0, 0))), int(round(startend(0, 1)))), score_mid_0(int(round(startend(1, 0))), int(round(startend(1, 1)))), score_mid_0(int(round(startend(2, 0))), int(round(startend(2, 1))))
                    , score_mid_0(int(round(startend(3, 0))), int(round(startend(3, 1)))), score_mid_0(int(round(startend(4, 0))), int(round(startend(4, 1)))), score_mid_0(int(round(startend(5, 0))), int(round(startend(5, 1))))
                    , score_mid_0(int(round(startend(6, 0))), int(round(startend(6, 1)))), score_mid_0(int(round(startend(7, 0))), int(round(startend(7, 1)))), score_mid_0(int(round(startend(8, 0))), int(round(startend(8, 1))))
                    , score_mid_0(int(round(startend(9, 0))), int(round(startend(9, 1))));

                    vec_y << score_mid_1(int(round(startend(0, 0))), int(round(startend(0, 1)))), score_mid_1(int(round(startend(1, 0))), int(round(startend(1, 1)))), score_mid_1(int(round(startend(2, 0))), int(round(startend(2, 1))))
                    , score_mid_1(int(round(startend(3, 0))), int(round(startend(3, 1)))), score_mid_1(int(round(startend(4, 0))), int(round(startend(4, 1)))), score_mid_1(int(round(startend(5, 0))), int(round(startend(5, 1))))
                    , score_mid_1(int(round(startend(6, 0))), int(round(startend(6, 1)))), score_mid_1(int(round(startend(7, 0))), int(round(startend(7, 1)))), score_mid_1(int(round(startend(8, 0))), int(round(startend(8, 1))))
                    , score_mid_1(int(round(startend(9, 0))), int(round(startend(9, 1))));

                    Eigen::Matrix <float, 10, 1>score_midpts = vec_x * vec[0] + vec_y * vec[1];
                    // 64就是0.5 * oriImg.shape[0]
                    float score_with_dist_prior = score_midpts.sum() / 10 + std::min(64.0 / (norm - 1), 0.0);

                    if (score_with_dist_prior > 0) {
                        int bad_num = 0;
                        for (int fff = 0; fff < 10; fff++) {
                            if (score_midpts(fff) < 0.05) {
                                bad_num++;
                            }
                        }
                        // 达成这个条件，则成为候选的连接关系
                        if (bad_num < 2) {
                            //加一个score_with_dist_prior + candA[aa][2] + candB[bb][2]]
                            //加一个score_with_dist_prior + candA[aa][2] + candB[bb][2]]
                            // aa是这个关键点，在这张特征图里被找到的次序
                            connectionT temp_connection{aa, bb, score_with_dist_prior,temp_A[aa].score,temp_B[bb].score};
                            // candidate后面还要过一轮关于重复性的筛选，所以只是中间变量
                            connection_candidate.push_back(temp_connection);
                        }
                    }
                }
            }//这个循环走完，就找到这一组点的所有可能的连接关系了
            //connection_candidate里是其中一组点的
            // 如果有一组关键点之间一个能连接的都没有，认为无效

            //if (connection_candidate.size() == 0) {
                //return FAILED;
            //}
            // 按照连接的可能性，从大到小排序
            sort(connection_candidate.begin(), connection_candidate.end(), cmp2);

            vector<int> temp_1;
            vector<int> temp_2;
            // ???
            temp_1.push_back(33);
            temp_2.push_back(33);

            int p_i;
            int p_j;
            // 获取所有的不重复的连接关系
            // one_connection是一组连接关系，是滤掉了重复的
            vector<connectionT> one_connection;
            for (int tt = 0; tt < connection_candidate.size(); tt++) {
                // i,j是可能有关联的关节，在peak中的序号，不是具体坐标
                int i = connection_candidate[tt].point_1;
                int j = connection_candidate[tt].point_2;
                float s = connection_candidate[tt].score;
                float s1 = connection_candidate[tt].score1;
                float s2 = connection_candidate[tt].score2;

                p_i = find_index(temp_1.begin(), temp_1.end(), i);
                p_j = find_index(temp_2.begin(), temp_2.end(), j);

                if (p_i != i && p_j != j) {
                    temp_1.push_back(i);
                    temp_2.push_back(j);
                    // temp_A和temp_B的代表有连接的关键点,其中的一组,i就是上面的aa,.num就是在所有特征图里的序号
                    //即all_peak_index
                    connectionT temp{ temp_A[i].num, temp_B[j].num,s,s1,s2};
                    one_connection.push_back(temp);
                    // 此时说明，已经所有的点都有了匹配
                    if (one_connection.size() >= min(LA, LB)) {
                        break;
                    }
                }
            }
            connection_candidate.clear();
            connection_all.push_back(one_connection);
            one_connection.clear();
        }
		else {
            missing_con.push_back(kk);
        }
    }//这里，已经把所有的连接关系都找到了

    // =======================================================================
    int flag = 0;
    //all_key_points里存了坐标和得分和全局序号
    //connection_all里存了连接的关键点的全局序号，连接得分，人体预备得分
    // 里面是vector<connectionT>
    //
    vector<human> subset;
    for (int k = 0; k < 19; k++) {
        vector<connectionT>temp_con = connection_all[k];
        //我的partAs里面要放第k种连接的全局序号
        vector<int> partAs;
        vector<int> partBs;

        for (int l = 0; l < temp_con.size(); l++) {
            partAs.push_back(temp_con[l].point_1);
            partBs.push_back(temp_con[l].point_2);
        }

        int indexA = limbSeq[k][0] - 1;
        int indexB = limbSeq[k][1] - 1;


        for (int i = 0; i < connection_all[k].size(); i++) {
            int found = 0;
            int subset_idx[2] = { -1,-1 };
            for (int j = 0; j < subset.size(); j++) {
                if(subset[j].p[indexA] == partAs[i] || subset[j].p[indexB] == partBs[i]) {subset_idx[found] = j;}
                found ++;
            }
            if (found == 1) {
                int j = subset_idx[0];
                if (subset[j].p[indexB] != partBs[i]) {
                    subset[j].p[indexB] = partBs[i];
                    subset[j].keynum += 1;
                    //connection_all[k]是vector<connectionT>，i之后是connectionT
                    subset[j].allscore += connection_all[k][i].score + connection_all[k][i].score2;
                }
            }
            else if (found == 1) {
                int j1 = subset_idx[0];
                int j2 = subset_idx[1];
                human h1 = subset[j1];
                human h2 = subset[j2];
                int count = 0;
                for (int h = 0; h < 18; h++) {
                    if (h1.p[h] >= 0 && h2.p[h] >= 0) count++;
                }
                if (count == 0)
                {
                    for (int ke = 0; ke < 18; ke++) {
                        subset[j1].p[ke] += subset[j2].p[ke];
                    }
                    subset[j1].keynum += subset[j2].keynum;
                    subset[j1].allscore += subset[j2].allscore + connection_all[k][i].score;
                    subset[j2].flag = -1;

                }
                else {
                    subset[j1].p[indexB] = partBs[i];
                    subset[j1].keynum += 1;
                    subset[j1].allscore+= connection_all[k][i].score + connection_all[k][i].score2;
                }

            }
            //connection_all里存了连接的关键点的全局序号，连接得分，人体预备得分
            else {
                human tem_new;
                tem_new.p[indexA] = partAs[i];
                tem_new.p[indexB] = partBs[i];
                tem_new.keynum = 2;
                tem_new.flag = 1;
                //all_peak_index的第一级是关键点编号,all_key_points[indexA]是indexA的关键点的所有的peak
                //all_key_points是one_pic_key_points的vector
                //one_pic_key_points是key_pointsT的vector
                // x, y,score ,all_peak_index，现在已知all_peak_index，所以可以通过全局序号，去找关键点的score
                tem_new.allscore = connection_all[k][i].score + connection_all[k][i].score2+ connection_all[k][i].score1;
                subset.push_back(tem_new);
            }
        }
    }


    //以及所有的人存下来之后还要再剔除
    int maxscore = -1;
    int best_id = -1;
    for (int h = 0; h < subset.size(); h++) {
//        if (subset[h].flag < 0) continue;
//        if (subset[h].keynum < 18 || subset[h].allscore / 18 < 0.4) continue;
        if (subset[h].allscore > maxscore)
        {
            maxscore = subset[h].allscore;
            best_id = h;
        }
    }
    // 找全局序号
    int temp_index[18] = { -1 };
    for (int aa = 0; aa < 18; aa++) {
        temp_index[aa] = subset[best_id].p[aa];
    }
   
    for (int aa = 0; aa < 18; aa++) {
        // aa是关键点序号，0,1是x与y轴
        // all_key_points里面是one_pic_key_points，可以通过关键点的编号搜索,即0-17
        // one_pic_key_points里面是key_pointsT类型，可以通过它压入one_pic_key_points时的序号搜索，全局序号
        // temp_index[aa]就是上面说的序号，all_peak_index
        temp_key_points[0][aa] = all_key_points[aa][temp_index[aa]].point_x;
        temp_key_points[1][aa] = all_key_points[aa][temp_index[aa]].point_y;
		temp_key_points[2][aa] = all_key_points[aa][temp_index[aa]].score;
        cv::Point p(temp_key_points[0][aa], temp_key_points[1][aa]);//初始化点坐标为(20,20)

        for (int bb = aa + 1; bb < 14; bb++) {
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

    cv::imwrite("../out/result.jpg", out1);



    memcpy(motion_data_new->data[0][0][FRAME_LENGTH - 1], temp_key_points[0], sizeof(float) * 18);
    memcpy(motion_data_new->data[0][1][FRAME_LENGTH - 1], temp_key_points[1], sizeof(float) * 18);
	memcpy(motion_data_new->data[0][2][FRAME_LENGTH - 1], temp_key_points[2], sizeof(float) * 18);
    // x
    memcpy(motion_data_new->data[0][0][0], motion_data_old->data[0][0][1], sizeof(float) * 18 * (FRAME_LENGTH - 1));
    // y
    memcpy(motion_data_new->data[0][1][0], motion_data_old->data[0][1][1], sizeof(float) * 18 * (FRAME_LENGTH - 1));
    memcpy(motion_data_old->data, motion_data_new->data, sizeof(float) * 2 * FRAME_LENGTH * 18);

    //求中心点坐标,并中心化
    float skeleton_center[2][FRAME_LENGTH] = { 0.0 };
    for (int c = 0; c < 2; c++)
    {
        for (int t = 0; t < FRAME_LENGTH; t++)
            for ( int t = 0; t < 30; t++ )
        {
            // data[0][0][t][v]是第t张图的第v个关节的x轴坐标
            skeleton_center[c][t] = float((motion_data_new->data[0][c][t][1] + motion_data_new->data[0][c][t][8] + motion_data_new->data[0][c][t][11]) / float(3.0));
            for (int v = 0; v < 14; v++)
            {
                motion_data_new->data[0][c][t][v] = motion_data_new->data[0][c][t][v] - skeleton_center[c][t];
            }
        }
    }

    return SUCCESS;
}


