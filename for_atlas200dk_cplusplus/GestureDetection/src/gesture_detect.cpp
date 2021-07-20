#include "gesture_detect.h"

#include <bits/types/clock_t.h>
#include <ctime>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <unistd.h>


#include "acl/acl.h"
#include "gesture_process.h"
#include "utils.h"
#include "camera.h"

#define RGBF32_CHAN_SIZE(width, height) ((width) * (height) * 3)

using namespace std;
using namespace ascend::presenter;

namespace {
    const char *aclConfigPath = "../src/acl.json";
    const char *kConfigFile = "../script/param.conf";
}

GestureDetect::GestureDetect(const char* kOpenPoseModelPath,const char* kGestureModelPath)
: deviceId_(0),context_(nullptr),stream_(nullptr),channel_(nullptr),
isInited_(false),OpenPoseModelPath_(kOpenPoseModelPath), GestureModelPath_(kGestureModelPath) {
    OpenPoseModel_.set_modelId(0);
    GestureModel_.set_modelId(1);

}

GestureDetect::~GestureDetect() {
    DeInit();
}


Result GestureDetect::OpenPresenterChannel() {
    // will listen on host_ip:port, and present on web_ip:web_port

//    param.host_ip = "192.168.1.134";  //IP address of Presenter Server
//    param.port = 7008;  //port of present service
//    param.channel_name = "Gesture Recognization";
//    param.content_type = ascend::presenter::ContentType::kVideo;  //content type is Video
    INFO_LOG("OpenChannel start");
    ascend::presenter::PresenterErrorCode errorCode =ascend::presenter::OpenChannelByConfig(channel_, kConfigFile);
//    INFO_LOG("OpenChannel param");
    if (errorCode != ascend::presenter::PresenterErrorCode::kNone) {
        ERROR_LOG("OpenChannel failed %d", static_cast<int>(errorCode));
        return FAILED;
    }

    return SUCCESS;
}

Result GestureDetect::InitResource() {

    aclError ret = aclInit(aclConfigPath);
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

    ret = dvpp_.InitResource(stream_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init dvpp failed\n");
        return FAILED;
    }

    return SUCCESS;

}

Result GestureDetect::InitModel(const char* OpenPoseModelPath, const char* GestureModelPath,bool use_dvpp) {
    // use_dvpp == true: for strawberry camera, flase for USB camera
    if(use_dvpp)
    {
        OpenPoseModel_.InitModel(OpenPoseModelPath,dvpp_);
    }
    else
    {
        OpenPoseModel_.InitModel(OpenPoseModelPath);
    }
    GestureModel_.InitModel(GestureModelPath);

    return SUCCESS;

}

Result GestureDetect::Init(bool use_dvpp) {
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

    ret = InitModel(OpenPoseModelPath_,GestureModelPath_,use_dvpp);
    if (ret!=SUCCESS) {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    INFO_LOG("Init model success");

    ret = OpenPresenterChannel();
    if (ret != SUCCESS) {
        ERROR_LOG("Open presenter channel failed");
        return FAILED;
    }
    isInited_ = true;
    return SUCCESS;

}


void GestureDetect::EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg) {
    vector<int> param = vector<int>(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 95; //default(95) 0-100
    cv::imencode(".jpg", origImg, encodeImg, param);
}


void GestureDetect::SendImage(cv::Mat& image) {
    // send opencv format image to the presenter server
    vector<uint8_t> encodeImg;
    EncodeImage(encodeImg, image); // BGR --> Jpeg

    ascend::presenter::ImageFrame imageParam;
    imageParam.format = ascend::presenter::ImageFormat::kJpeg;
    imageParam.width = image.cols;
    imageParam.height = image.rows;
    imageParam.size = encodeImg.size();
    imageParam.data = reinterpret_cast<uint8_t*>(encodeImg.data());
    std::vector<ascend::presenter::DetectionResult> gestureResults;
    imageParam.detection_results = gestureResults;

    ascend::presenter::PresenterErrorCode errorCode = ascend::presenter::PresentImage(channel_, imageParam);
    if (errorCode != ascend::presenter::PresenterErrorCode::kNone) {
        ERROR_LOG("PresentImage failed %d", static_cast<int>(errorCode));
    }

}

void GestureDetect::SendImage(ImageData& jpegImage,vector<DetectionResult>& detRes) {
    // send ImageData format image to the presenter server
    ascend::presenter::ImageFrame frame;
    frame.format = ImageFormat::kJpeg;
    frame.width = jpegImage.width;
    frame.height = jpegImage.height;
    frame.size = jpegImage.size;
    frame.data = jpegImage.data.get();
    frame.detection_results = detRes;

    ascend::presenter::PresenterErrorCode errorCode = ascend::presenter::PresentImage(channel_, frame);
    if (errorCode != ascend::presenter::PresenterErrorCode::kNone) {
        ERROR_LOG("PresentImage failed %d", static_cast<int>(errorCode));
    }

}


Result GestureDetect::Process() {

    // Use OpenPose model to extract human key points from a frame
    // when OpenPose processed FRAME_LENGHT (e.g. 30) frames, start STGCN model to analyse gesture
    // trigger STGCN every gesture_step (e.g. 5) frame
    // finally put the gesture estimation result on the frame and send to the presenter server

    bool start_flag=false;
    int gesture_id=0;
    int gesture_step=5;

    int success_num = 0;
    int image_num = 0;

    float motion_data[1][FRAME_LENGTH][18][3];
    memset(motion_data,0,sizeof(motion_data));

    while(1)
    {
        clock_t start_time = clock();
        if(image_num>=FRAME_LENGTH)
        {
            image_num %= FRAME_LENGTH;

            if(!start_flag)
                start_flag=true;
        }

        string imageFile = "../data/frames/" + to_string(image_num) + ".jpg";
        while (access(imageFile.data(), 0) == -1)
        {
            cout<<"waiting... "+imageFile<<endl;
            // check if the image exists
            // if not, waiting until the camera sends
        }

        cv::Mat frame = cv::imread(imageFile);

        if(frame.empty())
        {
            unlink(imageFile.c_str());
            image_num++;
            continue;
        }

        Result ret = OpenPoseModel_.Preprocess(frame);

        aclmdlDataset* inferenceOutput;
        OpenPoseModel_.Inference(inferenceOutput);

        ret = OpenPoseModel_.Postprocess(inferenceOutput, motion_data);
        success_num++;

        if (ret != SUCCESS) {
            std::cout<<"Openpose Postprocess not success"<<std::endl;
            continue;
        }

        if (start_flag && (success_num % gesture_step == 0)) {
            success_num%=gesture_step;
            aclmdlDataset* gestureOutput = nullptr;
            ret = GestureModel_.Inference(gestureOutput, motion_data);
            if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
            gesture_id = GestureModel_.Postprocess(gestureOutput);

        }

        Utils::put_text(frame,gesture_labels[gesture_id]);
        SendImage(frame);

        // to sync with camera, delete each frame right after processing
        unlink(imageFile.c_str());
        image_num++;

        INFO_LOG("process time:%lf",double(clock() - start_time) / CLOCKS_PER_SEC);
    }

    return SUCCESS;
}

Result GestureDetect::Process(int channelId) {

    // overload function of Process
    // start strawberry camera and use captured frames for gesture detection
    // channelId should match the camera insert number on atlas 200dk board

    Camera camera(channelId);
    if (camera.Open(channelId)) {
        ERROR_LOG("Failed to open channelId =%d.\n", channelId);
        return FAILED;
    }
    INFO_LOG("Open camera success, channelID = %d.\n",channelId);

    bool start_flag=false;
    int gesture_id=0;
    int gesture_step=10; // 24
    int success_num = 0;
    int image_num = 0;

    float motion_data[1][FRAME_LENGTH][18][3];
    memset(motion_data,0,sizeof(motion_data));

    void * buffer = nullptr;
    int size = camera.GetCameraDataSize(channelId);

    aclError aclRet = acldvppMalloc(&buffer, size);
    shared_ptr<ImageData> g_imagedata = make_shared<ImageData>();
    g_imagedata->data.reset((uint8_t*)buffer, [](uint8_t* p) { aclrtFree((void *)p); }); // the second para is deleter for buffer
    while(1)
    {
        clock_t start_time = clock();
        if(image_num>=FRAME_LENGTH)
        {
            image_num %= FRAME_LENGTH;

            // When pose infer count to FRAME_LENGTH , start gesture infer process.
            if(!start_flag)
                start_flag=true;
        }

        camera.Read(channelId, *(g_imagedata.get()));
        if (g_imagedata->data == nullptr) {
            ERROR_LOG("Read image %d failed\n", channelId);
            return FAILED;
        }

        ImageData resizedImage;
        Result ret = OpenPoseModel_.Preprocess(resizedImage,*(g_imagedata.get()));

        aclmdlDataset* inferenceOutput;
        OpenPoseModel_.Inference(inferenceOutput,resizedImage);

        ret = OpenPoseModel_.Postprocess(inferenceOutput, motion_data);
        success_num++;


        if (ret != SUCCESS) {
            std::cout<<"Openpose Postprocess not success"<<std::endl;
            continue;
        }

        if (start_flag && (success_num % gesture_step == 0)) {
            success_num%=gesture_step;
            aclmdlDataset* gestureOutput = nullptr;
            ret = GestureModel_.Inference(gestureOutput, motion_data);
            if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
            gesture_id = GestureModel_.Postprocess(gestureOutput);

        }
        ImageData jpgImage;
        ret = dvpp_.CvtYuv420spToJpeg(jpgImage, *(g_imagedata.get()));

        if(ret == FAILED) {
            ERROR_LOG("Convert jpeg to yuv failed\n");
            return FAILED;
        }
        DetectionResult detect_result;
        vector<DetectionResult> detectResults;
        Point point_lt, point_rb;
        point_lt.x=50;
        point_lt.y=50;
        point_rb.x=200;
        point_rb.y=50;
        detect_result.lt = point_lt;
        detect_result.rb = point_rb;
        detect_result.result_text = gesture_labels[gesture_id];
        detectResults.emplace_back(detect_result);
        SendImage(jpgImage,detectResults);

        image_num++;
        INFO_LOG("process time:%lf",double(clock() - start_time) / CLOCKS_PER_SEC);
    }

    return SUCCESS;
}

void GestureDetect::Test(string data_dir) {
    // function for debugging
    // process .txt files in data_dir and output gesture inference results

    std::vector<string> filenames;
    std::vector<int> gesture_res;
    Utils::GetAllFiles(data_dir,filenames);
    for(int i=0;i<filenames.size();i++)
    {
        cout<<filenames[i]<<endl;
        float motion_data[1][FRAME_LENGTH][18][3];
        memset(motion_data,0,sizeof(motion_data)); // necessary
        Utils::read_motion_data(filenames[i],motion_data);
        for(int k=0;k<18;k++)
            cout<<motion_data[0][0][k][0]<<" "<<motion_data[0][0][k][1]<<" "<<motion_data[0][0][k][2]<<endl;
        aclmdlDataset* gestureOutput = nullptr;
        Result ret = GestureModel_.Inference(gestureOutput, motion_data);
        if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
        }
        gesture_res.push_back(GestureModel_.Postprocess(gestureOutput));
    }

    cout<<"[ ";
    for(int i=0;i<gesture_res.size();i++)
    {
        cout<<gesture_res[i]<<" ";
    }
    cout<<"]"<<endl;


    cout<<"[ ";
    for(int i=0;i<gesture_res.size();i++)
    {
        cout<<gesture_labels[gesture_res[i]]<<" ";
    }
    cout<<"]"<<endl;


}

void GestureDetect::DeInit() {

    delete channel_;
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

}

