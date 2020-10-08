
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

GestureDetect::GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath,
                            uint32_t ImgWidth, uint32_t ImgHeight)
: deviceId_(0),context_(nullptr),stream_(nullptr),
isInited_(false),OpenPoseModelPath_(kOpenPoseModelPath), GestureModelPath_(kGestureModelPath) {
    OpenposeModel_.set_modelId(0);
//    OpenposeModel_.set_modelsize(ImgWidth,ImgHeight);
    GestureModel_.set_modelId(1);
//    inputDataSize_ = (ImgWidth, ImgHeight);

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
    Result ret = OpenposeModel_.LoadModelFromFileWithMem(OpenPoseModelPath);
    if(ret!=SUCCESS) {
        ERROR_LOG("openpose model load failed");
        return FAILED;
    }
    INFO_LOG("openpose model load success");

    ret = GestureModel_.LoadModelFromFileWithMem(GestureModelPath);
    if(ret!=SUCCESS) {
        ERROR_LOG("gesture model load failed");
        return FAILED;
    }
    INFO_LOG("gesture model load success");

    ret = OpenposeModel_.CreateDesc();
    if(ret!=SUCCESS) {
        ERROR_LOG("openpose model CreateDesc failed");
        return FAILED;
    }
    INFO_LOG("openpose model CreateDesc success");

    ret = GestureModel_.CreateDesc();
    if(ret!=SUCCESS) {
        ERROR_LOG("gesture model CreateDesc failed");
        return FAILED;
    }
    INFO_LOG("gesture model CreateDesc success");


    ret = OpenposeModel_.CreateOutput();
    if(ret!=SUCCESS) {
        ERROR_LOG("openpose model CreateOutPut failed");
        return FAILED;
    }
    INFO_LOG("openpose model CreateOutPut success");

    ret = GestureModel_.CreateOutput();
    if(ret!=SUCCESS) {
        ERROR_LOG("gesture model CreateOutPut failed");
        return FAILED;
    }
    INFO_LOG("gesture model CreateOutPut success");


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

//    ret = OpenPresenterChannel();
//    if (ret != SUCCESS) {
//        ERROR_LOG("Open presenter channel failed");
//        return FAILED;
//    }
    isInited_ = true;
    return SUCCESS;

}

Result GestureDetect::ProcessMotionData() {
    float temp_left = 120;
    float temp_right = 0;
    float temp_top = 160;
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
        for(int key_num = 0; key_num < 18; key_num++){
            motion_data_new->data[0][0][pic_num][key_num] /= total_bottom;
            motion_data_new->data[0][1][pic_num][key_num] /= total_bottom;
        }
    }
    return SUCCESS;
}

Result GestureDetect::Process() {
    int success_num = -4;
    int image_num = 0;
    clock_t start_time = clock();

    shared_ptr<ImageDesc> image_desc = nullptr;
    MAKE_SHARED_NO_THROW(image_desc, ImageDesc);

    if (image_desc == nullptr) {
        ERROR_LOG("Failed to MAKE_SHARED_NO_THROW for ImageDesc.");
        return FAILED;
    }


    for (;;image_num++) {
        std::cout<<"image num: "<<image_num<<std::endl;
        image_num %= FRAME_LENGTH;
        // 图像文件的路径
        std::string imageFile = "../data/frames/" + to_string(image_num) + ".jpg";
        const char* tmp = imageFile.data();
        std::cout<<"image path: "<<tmp<<std::endl;
        // 检查该文件是否存在
        if ((access(tmp, 0)) == -1) {
            break;
            image_num--;
            std::cout<<"image_num-- "<<std::endl;
            // 如果一秒30帧，平均一帧是0.033s，等待0.04s，差不多下一帧就到了
            usleep(40000);
            // continue;
        }

        clock_t read_time = clock();
//        cout<<image_desc<<endl;
//        cout<<image_desc.get()<<endl;

        Result ret = OpenposeModel_.Preprocess(image_desc, imageFile); // resize and padding the frame
//        printf("desc:%f\n",*(image_desc->data));

        // create input for the model inference.
        ret = OpenposeModel_.CreateInput((void*) image_desc->data.get(), image_desc->size);
//        cout<<"size"<<image_desc->size<<endl;
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }

        //将预处理的图片送入OpenPose模型推理,并获取OpenPose推理结果
        // 推理时间
        clock_t infer_time = clock();

        OpenposeModel_.Execute();

        std::cout << "infer time " << double(clock() - infer_time) / CLOCKS_PER_SEC << std::endl;

        aclmdlDataset*inferenceOutput = OpenposeModel_.GetModelOutputData();
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
         //解析OpenPose推理输出
        ret = OpenposeModel_.Postprocess(inferenceOutput, motion_data_new);
        success_num++;
        if (ret != SUCCESS) {
            std::cout<<"Openpose Postprocess not success"<<std::endl;
            continue;
        }
        // 删除之前的图片
        int before_index = image_num - 50;
        if (before_index < 0) {
            before_index += 100;
        }
        string img_path = "../data/frames/" + to_string(before_index) + ".jpg";
        const char * pre_img = img_path.c_str();
        // 删除图片文件
        unlink(pre_img);

        clock_t pose_time = clock();
        std::cout << "pose_time time " << double(clock() - pose_time) / CLOCKS_PER_SEC << std::endl;

        // 每更新五帧进行一次动作识别
        if (success_num % 5 == 0) {
            //ProcessMotionData();
            //SaveData();
            //将人体骨架序列送入Gesture模型推理,并获取400种动作的可能性
            aclmdlDataset* gestureOutput = nullptr;

            clock_t ges_time = clock();

            ret = GestureModel_.Inference(gestureOutput, motion_data_new);
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

    if (context_ != nullptr) {
        ret = aclrtDestroyStream(context_);
        if(ret != ACL_ERROR_NONE) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
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




