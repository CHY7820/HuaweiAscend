//
// Created by mind on 10/2/20.
//
#include "utils.h"
#include "pose_process.h"

OpenPoseProcess::OpenPoseProcess() : ModelProcess() {}
//
//OpenPoseProcess::~OpenPoseProcess() {
//    Unload();
//    DestroyDesc();
//    DestroyInput();
//    DestroyOutput();
//}


Result OpenPoseProcess::Preprocess(ImageData& resizedImage, ImageData& srcImage)
{
    ImageData imageDevice;
    Utils:CopyImageDataToDevice(imageDevice,srcImage);
    ImageData yuvImage;
    Result ret = dvpp_.CvtJpegToYuv420sp(yuvImage, imageDevice);
    if(ret != SUCCESS) {
        ERROR_LOG("Convert jpeg to yuv failed");
        return FAILED;
    }
    INFO_LOG("Convert jpeg to yuv success");

    ret = dvpp_.Resize(resizedImage,yuvImage,modelWidth_,modelHeight_);
    if(ret != SUCCESS) {
        ERROR_LOG("Resize image failed");
    }
    INFO_LOG("Resize image success");

}
