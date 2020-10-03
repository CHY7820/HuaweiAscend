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

* File main.cpp
* Description: dvpp sample main func
*/

#include <bits/types/clock_t.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>

#include "gesture_detect.h"
#include "utils.h"
using namespace std;
int success_num = -4;
namespace {
uint32_t OpenPoseModelWidth = 128;
uint32_t OpenPoseModelHeight = 128;
const char* OpenPose_ModelPath = "../model/pose_deploy_final.om";
const char* Gesture_ModelPath = "../model/stgcn_fps30_sta_ho_ki432.om";
}

std::shared_ptr<EngineTransNewT> motion_data_new = std::make_shared<EngineTransNewT>();
int image_num = 0;

// 标准化骨骼关键点序列
Result ProcessOpenPoseData(){
    float temp_left = 128;
    float temp_right = 0;
    float temp_top = 128;
    float temp_bottom = 0;
    float total_left = 0;
    float total_right = 0;
    float total_top = 0;
    float total_bottom = 0;
    // 计算前五帧人体躯干的像素长度，作为标准，进行归一化
    for (int pic_num = FRAME_LENGTH-1; pic_num > FRAME_LENGTH-6; pic_num--){
        // 累加
        total_bottom +=  float(motion_data_new->data[0][1][pic_num][8] + motion_data_new->data[0][1][pic_num][11]) / 2 - float(motion_data_new->data[0][1][pic_num][1]);
    }

    total_bottom /= 5.0;
    for (int pic_num = 0; pic_num < FRAME_LENGTH; pic_num++){
        for(int key_num = 0; key_num < 14; key_num++){
            motion_data_new->data[0][0][pic_num][key_num] /= total_bottom;
            motion_data_new->data[0][1][pic_num][key_num] /= total_bottom;
        }
    }
    return SUCCESS;
}

// 保存输入动作识别模型的数据，为了离线测试（可以不需要）
Result SaveData(){
    string file_name = "./data/" + to_string(image_num) + "_raw_data.txt";
    ofstream file;
    file.open(file_name.c_str(), ios::trunc);
    for (int jj=0; jj < 2; jj++){
        for (int aa = 0; aa < FRAME_LENGTH; aa++){
            for (int qq = 0; qq < 14; qq++){
                file << motion_data_new->data[0][jj][aa][qq] << "\n";
            }
        }
    }
    file.close();
    return SUCCESS;
}

int main(int argc, char *argv[]) {
    //实例化目标检测对象,参数为分类模型路径,模型输入要求的宽和高 加载两个模型文件
    GestureDetect detect(OpenPose_ModelPath, Gesture_ModelPath, OpenPoseModelWidth, OpenPoseModelHeight);
    //初始化分类推理的acl资源, 模型和内存
    clock_t init_time = clock();
    Result ret = detect.Init();
    cout << "init time " << double(clock() - init_time) / CLOCKS_PER_SEC << endl;
    if (ret != SUCCESS) {
        ERROR_LOG("Classification Init resource failed");
        return FAILED;
    }

    //逐张图片推理
    ImageData image;
    clock_t start_time = clock();
    for (;;image_num++) {
        image_num %= 100;
        // 图像文件的路径
        string imageFile = "../data/" + to_string(image_num) + ".jpg";
        const char* tmp = imageFile.data();
        // 检查该文件是否存在
        if((access(tmp, 0)) == -1){
            image_num--;
            // 如果一秒30帧，平均一帧是0.033s，等待0.04s，差不多下一帧就到了
            usleep(40000);
            continue;
        }

        // 读取图片时间
        clock_t read_time = clock();
        // 读取图像文件
        int read_result = Utils::ReadImageFile(image, imageFile);
        if (read_result == FAILED){
            continue;
        }
        if (image.data == nullptr) {
            ERROR_LOG("Read image %s failed", imageFile.c_str());
            return FAILED;
        }
        //预处理图片:读取图片,讲图片缩放到模型输入要求的尺寸
        ImageData resizedImage;
        Result ret = detect.Preprocess(resizedImage, image);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next", imageFile.c_str());
            continue;
        }
        cout << "read time " << double(clock() - read_time) / CLOCKS_PER_SEC << endl;
        //将预处理的图片送入OpenPose模型推理,并获取OpenPose推理结果
        aclmdlDataset* inferenceOutput = nullptr;
        // 推理时间
        clock_t infer_time = clock();
        ret = detect.OpenPoseInference(inferenceOutput, resizedImage);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }

        cout << "infer time " << double(clock() - infer_time) / CLOCKS_PER_SEC << endl;
        // 删除之前的图片
        int before_index = image_num - 50;
        if(before_index < 0){
            before_index += 100;
        }
        string img_path = "../data/" + to_string(before_index) + ".jpg";
        const char * pre_img = img_path.c_str();
        // 删除图片文件
        unlink(pre_img);

        clock_t pose_time = clock();

        // 解析OpenPose推理输出
        ret = detect.Postprocess(image, inferenceOutput, motion_data_new, success_num);
        if (ret != SUCCESS) {
            continue;
        }
        cout << "pose_time time " << double(clock() - pose_time) / CLOCKS_PER_SEC << endl;
        // 每更新五帧进行一次动作识别
        if (success_num % 5 == 0) {
            ProcessOpenPoseData();
//            SaveData();
            //将人体骨架序列送入Gesture模型推理,并获取5种动作的可能性
            aclmdlDataset* gestureOutput = nullptr;

            clock_t ges_time = clock();

            ret = detect.GestureInference(gestureOutput, motion_data_new);
            if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
            cout << "ges_time time " << double(clock() - ges_time) / CLOCKS_PER_SEC << endl;
            // 数据后处理
            ret = detect.PostGestureProcess(gestureOutput);
            if (ret != SUCCESS) {
                ERROR_LOG("Process model inference output data failed");
                // 退出程序
                break;
            }
        }
    }
    return SUCCESS;
}
