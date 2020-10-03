
#include "gesture_detect.h"

GestureDetect::GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath,
                            uint32_t ImgWidth, uint32_t ImgHeight)
: processed_imgs(0), deviceId(0),context_(nullptr),stream_(nullptr),
isInited_(false),OpenPoseModelPath_(kOpenPoseModelPath), GestureModelPath_(kGestureModelPath) {
    OpenposeModel_.SetPara(ImgWidth,ImgHeight);
}

GestureDetect::~GestureDetect() {
    //DeInit();
}

GestureDetect::InitResource() {
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

    ret = aclrtCreateStream(&stream);
    if(ret!=ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("acl create stream success");




}

Result GestureDetect::InitModel(const char* OpenPoseModelPath, const char* GestureModelPath) {
    Result ret = OpenposeModel_.LoaadModelFromFileWithMem(OpenPoseModelPath);
    if(ret!=SUCCESS) {
        ERROR_LOG("openpose model load failed");
        return FAILED;
    }
    INFO_LOG("openpose model load success");

    ret = GestureModel_.LoaadModelFromFileWithMem(GestureModelPath);
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


    ret = OpenposeModel_.CreateOutPut();
    if(ret!=SUCCESS) {
        ERROR_LOG("openpose model CreateOutPut failed");
        return FAILED;
    }
    INFO_LOG("openpose model CreateOutPut success");

    ret = GestureModel_.CreateOutPut();
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

    ret = InitResource();
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

//    ret = InitModel(OpenPoseModelPath_,GestureModelPath_);
//    if (ret!=SUCCESS) {
//        ERROR_LOG("Init model failed");
//        return FAILED;
//    }

    INFO_LOG("Init model success");

    isInited_ = true;
    return SUCCESS;

}

Result GestureDetect::Process(ImageData& image) {
    ImageData resizedImage;
    aclmdlDataset* openposeOutput = nullptr;
    Result ret = OpenposeModel_.Preprocess(resizedImage,image);
    ret = OpenposeModel_.Execute(); //openposeOutput,resizedImage
    ret = OpenposeModel_.Postprocess(image,inferenceOutput...);
    processed_imgs++;
    // update every T frame
    if(processed_imgs%FRAMES==0) {
        aclmdlDataset* gestureOutput = nullptr;
//        GestureModel_.Inference(gestureOutput,motion_data);
        GestureModel_.Execute();
        GestureModel_.Postprocess(gestureOutput);
    }


    return SUCCESS;
}

