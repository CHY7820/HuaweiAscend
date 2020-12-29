
#include "gesture_detect.h"
#include "acl/acl.h"
#include "gesture_process.h"
#include "utils.h"
#include <bits/types/clock_t.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

#define RGBF32_CHAN_SIZE(width, height) ((width) * (height) * 3)

using namespace std;

GestureDetect::GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath)
: deviceId_(0),context_(nullptr),stream_(nullptr),
isInited_(false),OpenPoseModelPath_(kOpenPoseModelPath), GestureModelPath_(kGestureModelPath) {
    OpenPoseModel_.set_modelId(0);
    GestureModel_.set_modelId(1);
}

GestureDetect::~GestureDetect() {
    DeInit();
}

Result GestureDetect::InitResource() {
    aclError ret = aclInit(nullptr);
    if(ret!=ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    ret = aclrtSetDevice(deviceId_);
    if(ret!=ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("acl open device %d success",deviceId_);

    ret = aclrtCreateContext(&context_,deviceId_);
    if(ret!=ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("acl create context success");

    ret = aclrtCreateStream(&stream_);
    if(ret!=ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("acl create stream success");

    return SUCCESS;


}

Result GestureDetect::InitModel(const char* OpenPoseModelPath, const char* GestureModelPath) {

    OpenPoseModel_.InitModel(OpenPoseModelPath);
    GestureModel_.InitModel(GestureModelPath);
    return SUCCESS;

}

Result GestureDetect::Init() {
    if (isInited_) {
        INFO_LOG("acl inited already");
        return SUCCESS;
    }

    Result ret = InitResource();
    if (ret!=SUCCESS) {
        ERROR_LOG("Init acl resource failed");
        return FAILED;
    }

    INFO_LOG("Init acl resource success");

    ret = InitModel(OpenPoseModelPath_,GestureModelPath_);
    if (ret!=SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    INFO_LOG("Init model success");

//    ret = GestureProcess::OpenPresenterChannel();
//    if (ret != SUCCESS) {
//        ERROR_LOG("Open presenter channel failed");
//        return FAILED;
//    }
    isInited_ = true;
    return SUCCESS;

}
//
//Result GestureDetect::ProcessMotionData() {
//    float temp_left = 120;
//    float temp_right = 0;
//    float temp_top = 160;
//    float temp_bottom = 0;
//    float total_left = 0;
//    float total_right = 0;
//    float total_top = 0;
//    float total_bottom = 0;
//    // 计算前五帧人体躯干的像素长度，作为标准，进行归一化
//    for (int pic_num = FRAME_LENGTH-1; pic_num > FRAME_LENGTH-6; pic_num--){
//        // 累加
//        total_bottom +=  float(motion_data[0][1][pic_num][8] + motion_data[0][1][pic_num][11]) / 2 - float(motion_data[0][1][pic_num][1]);
//    }
//
//    total_bottom /= 5.0;
//    for (int pic_num = 0; pic_num < FRAME_LENGTH; pic_num++){
//        for(int key_num = 0; key_num < 18; key_num++){
//            motion_data[0][0][pic_num][key_num] /= total_bottom;
//            motion_data[0][1][pic_num][key_num] /= total_bottom;
//        }
//    }
//    return SUCCESS;
//}

Result GestureDetect::Process() {
    bool start_flag=false;
    int success_num = -5;
    int image_num = 0;
    float motion_data[1][3][FRAME_LENGTH][18];
    memset(motion_data,0,sizeof(motion_data));

    clock_t start_time = clock();
    Result ret;
    while(1) {

        std::cout<<"image num: "<<image_num<<std::endl;
        if(image_num>=FRAME_LENGTH) // When pose infer count 100 , start gesture infer process.
        {
            image_num %= FRAME_LENGTH;
            if(!start_flag)
                start_flag=true;
        }

        // 图像文件的路径
        std::string imageFile = "../data/frames/" + to_string(image_num) + ".jpg";
        const char* tmp = imageFile.data();

        // 检查该文件是否存在
        if (access(tmp, 0) == -1)
        {
//            break;
            cout<<"image not found, wait..."<<endl;
            image_num--;
            std::cout<<"image_num-- "<<std::endl;
            // 如果一秒30帧，平均一帧是0.033s，等待0.04s，差不多下一帧就到了
            usleep(40000);
            continue;
        }

//        clock_t read_time = clock();
        ret = OpenPoseModel_.Preprocess(imageFile); // image pre-processing

        clock_t infer_time = clock();
        aclmdlDataset* inferenceOutput;
        OpenPoseModel_.Inference(inferenceOutput);

        std::cout << "openpose infer time " << double(clock() - infer_time) / CLOCKS_PER_SEC << std::endl;

         //解析OpenPose推理输出

        ret = OpenPoseModel_.Postprocess(inferenceOutput, motion_data);
        success_num++;

        if (ret != SUCCESS) {
            std::cout<<"Openpose Postprocess not success"<<std::endl;
            continue;
        }

//        if(start_flag) // 100->0->start, delete
//        {
//            int before_index = image_num - FRAME_LENGTH/2;
//            if (before_index < 0) {
//                before_index += FRAME_LENGTH;
//            } // before index = 50, 51, ... , 99, will be deleted
//
//            string img_path = "../data/frames/" + to_string(before_index) + ".jpg";
//            const char * pre_img = img_path.c_str();
//            unlink(pre_img); // delete image from 50 to 100
//
//        }


//        clock_t pose_time = clock();
//        std::cout << "pose_time time " << double(clock() - pose_time) / CLOCKS_PER_SEC << std::endl;

        // 每更新五帧进行一次动作识别

        if (start_flag && (success_num % 5 == 0)) {
            cout<<"Start Gesture Estimation..."<<endl;
            //ProcessMotionData();
            //SaveData();
            //将人体骨架序列送入Gesture模型推理,并获取400种动作的可能性
            aclmdlDataset* gestureOutput = nullptr;

            clock_t ges_time = clock();

            ret = GestureModel_.Inference(gestureOutput, motion_data);
            if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
            std::cout << "ges_time time " << double(clock() - ges_time) / CLOCKS_PER_SEC << std::endl;
            // 数据后处理
            ret = GestureModel_.Postprocess(imageFile,gestureOutput);
            if (ret != SUCCESS) {
                ERROR_LOG("Process model inference output data failed");
                // 退出程序
                break;
            }
        }
        unlink(imageFile.c_str()); // to sync with camera, delete it right after read and processed one image
        image_num++;
//        if(image_num==50)
//            break;
    }

    return SUCCESS;
}


void GestureDetect::DeInit() {

    //OpenPoseModel_.DestroyResource();
    //GestureModel_.DestroyResource();
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if(ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }

    ret = aclrtResetDevice(deviceId_);
    if(ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed");
    }

    ret = aclFinalize();
    if(ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed");
    }

//    aclrtFree(imageInfoBuf_);
}




