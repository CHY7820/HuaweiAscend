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
//#include <mutex>
//#include <pthread.h>

#include "acl/acl.h"
#include "gesture_process.h"
#include "utils.h"
#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

#define RGBF32_CHAN_SIZE(width, height) ((width) * (height) * 3)

using namespace std;
/*
struct Attr{
    string imageFile;
    bool start_flag;
    GestureDetect* t;
};*/
//void process(struct Attr* a);
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
    // will listen on ip:port, and present on web_ip:web_port

    ascend::presenter::OpenChannelParam param;
    param.host_ip = "192.168.1.134";  //IP address of Presenter Server
    param.port = 7008;  //port of present service
    param.channel_name = "Gesture Recognization";
    param.content_type = ascend::presenter::ContentType::kVideo;  //content type is Video
    INFO_LOG("OpenChannel start");
    ascend::presenter::PresenterErrorCode errorCode =ascend::presenter::OpenChannel(channel_, param);
    INFO_LOG("OpenChannel param");
    if (errorCode != ascend::presenter::PresenterErrorCode::kNone) {
        ERROR_LOG("OpenChannel failed %d", static_cast<int>(errorCode));
        return FAILED;
    }

    return SUCCESS;
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
    // send image to the presenter server
    vector<uint8_t> encodeImg;
    EncodeImage(encodeImg, image);

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


Result GestureDetect::Process() {

    // use OpenPose model to extract human key points from a frame
    // when OpenPose processed FRAME_LENGHT (i.e. 100) frames, start STGCN model to analyse gesture
    // trigger STGCN every 5 frame
    // finally put the gesture estimatation result on the frame and send to the presenter server

    bool start_flag=false;
    int gesture_id=0;

    int success_num = 0;
    int image_num = 0;

    float motion_data[1][3][FRAME_LENGTH][18];
    memset(motion_data,0,sizeof(motion_data));

//    clock_t start_time = clock();
    while(1)
    {
        std::cout<<"image num: "<<image_num<<std::endl;
        if(image_num>=FRAME_LENGTH)
        {
            image_num %= FRAME_LENGTH;

            // When pose infer count 100 , start gesture infer process.
            if(!start_flag)
                start_flag=true;
        }

        string imageFile = "../data/frames/" + to_string(image_num) + ".jpg";
        cout<<imageFile<<endl;
        while (access(imageFile.data(), 0) == -1)
        {
            // check if the image exists
            // if not, waiting until the camera sends
            sleep(1); // sleep 2 s
            cout<<"waiting..."<<endl;
        }

        cv::Mat frame = cv::imread(imageFile,CV_LOAD_IMAGE_COLOR);

        Result ret = OpenPoseModel_.Preprocess(frame); // image pre-processing
        if(ret!=SUCCESS)
        {
            continue;
        }

//        clock_t infer_time = clock();
        aclmdlDataset* inferenceOutput;
        OpenPoseModel_.Inference(inferenceOutput); // openpose inferring

//        std::cout << "openpose infer time " << double(clock() - infer_time) / CLOCKS_PER_SEC << std::endl;

        ret = OpenPoseModel_.Postprocess(inferenceOutput, motion_data); // post-process, getting key points of human body and put into motion data array
        success_num++;

        if (ret != SUCCESS) {
            std::cout<<"Openpose Postprocess not success"<<std::endl;
            continue;
        }

//        clock_t pose_time = clock();
//        std::cout << "pose_time time " << double(clock() - pose_time) / CLOCKS_PER_SEC << std::endl;

        if (start_flag && (success_num % 5 == 0)) {
            success_num%=5;
            cout<<"Start Gesture Estimation..."<<endl;

            aclmdlDataset* gestureOutput = nullptr;

//            clock_t ges_time = clock();
//            Utils::ProcessMotionData(motion_data); //

            ret = GestureModel_.Inference(gestureOutput, motion_data);
            if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
                ERROR_LOG("Inference model inference output data failed");
                return FAILED;
            }
            //std::cout << "ges_time time " << double(clock() - ges_time) / CLOCKS_PER_SEC << std::endl;
            gesture_id = GestureModel_.Postprocess(gestureOutput,frame);

        }

        Utils::put_text(frame,gesture_labels[gesture_id]);
        SendImage(frame);

        // to sync with camera, delete each frame right after processing
        unlink(imageFile.c_str());
        image_num++;
    }

    return SUCCESS;
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




/*void process(void* attr)
{
    struct Attr* a=(struct Attr*) attr;
    float motion_data[1][3][FRAME_LENGTH][18];
    memset(motion_data,0,sizeof(motion_data));
    int ret = a->t->OpenPoseModel_.Preprocess(a->imageFile,ges_type); // image pre-processing
    if(ret!=SUCCESS)
    {
        return;
    }

    clock_t infer_time = clock();
    aclmdlDataset* inferenceOutput;
    a->t->OpenPoseModel_.Inference(inferenceOutput);

    //std::cout << "openpose infer time " << double(clock() - infer_time) / CLOCKS_PER_SEC << std::endl;

    //解析OpenPose推理输出

    ret = a->t->OpenPoseModel_.Postprocess(inferenceOutput, motion_data);
    m.lock();

    success_num++;
    m.unlock();

    if (ret != SUCCESS) {
        std::cout<<"Openpose Postprocess not success"<<std::endl;
        return;
    }

    //        clock_t pose_time = clock();
    //        std::cout << "pose_time time " << double(clock() - pose_time) / CLOCKS_PER_SEC << std::endl;

    // 每更新五帧进行一次动作识别

    if (a->start_flag && (success_num % 5 == 0)) {
        m.lock();
        success_num%=5;
        m.unlock();
        cout<<"Start Gesture Estimation..."<<endl;

        aclmdlDataset* gestureOutput = nullptr;

        clock_t ges_time = clock();
        Utils::ProcessMotionData(motion_data);

        ret =a->t->GestureModel_.Inference(gestureOutput, motion_data);
        if ((ret != SUCCESS) || (gestureOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return;
        }
        //std::cout << "ges_time time " << double(clock() - ges_time) / CLOCKS_PER_SEC << std::endl;
        // 数据后处理
        ret = a->t->GestureModel_.Postprocess(a->imageFile,gestureOutput,ges_type);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            // 退出程序
            return ;
        }
        //            Utils::write_motion_data(motion_data);

    }


    unlink(a->imageFile.c_str()); // to sync with camera, delete it right after read and processed one image

}*/


