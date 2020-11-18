#include "gesture_process.h"
#include "utils.h"
#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"
#include <bits/stdint-uintn.h>
#include <cmath>

using namespace std;

GestureProcess::GestureProcess() : ModelProcess() {}


Result GestureProcess::InitModel(const char* modelPath)
{
    Result ret = LoadModelFromFileWithMem(modelPath);
    if(ret!=SUCCESS) {
        ERROR_LOG("model load failed");
        return FAILED;
    }
    INFO_LOG("model load success");

    ret = CreateDesc();
    if(ret!=SUCCESS) {
        ERROR_LOG("model CreateDesc failed");
        return FAILED;
    }
    INFO_LOG("model CreateDesc success");


    ret = CreateOutput();
    if(ret!=SUCCESS) {
        ERROR_LOG("model CreateOutPut failed");
        return FAILED;
    }

    INFO_LOG("model CreateOutPut success");

    INFO_LOG("STGCN Model initial success!");
    return SUCCESS;
}


Result GestureProcess::Inference(aclmdlDataset*& inferenceOutput, float motion_data[1][3][FRAME_LENGTH][18]) {

    uint32_t buffer_size = 3 * FRAME_LENGTH * 18 * sizeof(float);
    cout<<"x: "<<motion_data[0][0][FRAME_LENGTH-1][0]<<" y: "<<motion_data[0][1][FRAME_LENGTH-1][0]<<" score: "<<motion_data[0][2][FRAME_LENGTH-1][0]<<endl;
    Result ret = CreateInput((void*) motion_data, buffer_size);
    if (ret != SUCCESS) {
        ERROR_LOG("model CreateInput failed");
        return FAILED;
    }
    INFO_LOG("model CreateInput success");

    ret = Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GetModelOutputData();
    cout<<"In gesture process inference"<<endl;
    return SUCCESS;
}

void GestureProcess::EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg) {
    vector<int> param = vector<int>(2);
    //param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[0] = 95;
    param[1] = 95;//default(95) 0-100

    cv::imencode(".jpg", origImg, encodeImg, param);
}


Result GestureProcess::SendImage(cv::Mat& image) {
    vector<uint8_t> encodeImg;
    EncodeImage(encodeImg, image);

    ascend::presenter::ImageFrame imageParam;
    imageParam.format = ascend::presenter::ImageFormat::kJpeg;
    imageParam.width = image.cols;
    imageParam.height = image.rows;
    imageParam.size = encodeImg.size();
    imageParam.data = reinterpret_cast<uint8_t*>(encodeImg.data());

    std::vector<ascend::presenter::DetectionResult> detectionResults;
    imageParam.detection_results = detectionResults;

    ascend::presenter::PresenterErrorCode errorCode = ascend::presenter::PresentImage(channel_, imageParam);
    if (errorCode != ascend::presenter::PresenterErrorCode::kNone) {
        ERROR_LOG("PresentImage failed %d", static_cast<int>(errorCode));
        return FAILED;
    }

    return SUCCESS;
}

Result GestureProcess::OpenPresenterChannel() {
    ascend::presenter::OpenChannelParam param;
    param.host_ip = "192.168.1.134";  //IP address of Presenter Server
    param.port = 7008;  //port of present service
    param.channel_name = "colorization-video";
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
//
Result GestureProcess::Postprocess(const string &path, aclmdlDataset* modelOutput){

    uint32_t ges_size=0;
    float* ges_types =(float*)GetInferenceOutputItem(ges_size,modelOutput,0);
//    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(modelOutput,0);
//    if (dataBuffer == nullptr) {
//        ERROR_LOG("get model output aclmdlGetDatasetBuffer failed");
//        return FAILED;
//    }
//    void* data = aclGetDataBufferAddr(dataBuffer);
//    if (data == nullptr) {
//        ERROR_LOG("aclGetDataBufferAddr from dataBuffer failed.");
//        return FAILED;
//    }
//
//    std::vector<int> res;
//    aclError ret = aclrtMemcpy(&res, sizeof(res), data, sizeof(res), ACL_MEMCPY_DEVICE_TO_DEVICE);
//    if (ret != ACL_ERROR_NONE) {
//        ERROR_LOG("result labels aclrtMemcpy failed!");
//        return FAILED;
//    }
//    cout<<res[0]<<endl;
//
//    std::vector<int>::iterator maxVal = std::max_element(std::begin(res), std::end(res));
//    int k=std::distance(std::begin(res),maxVal);



    cout<<"ges_size: "<<ges_size<<endl;
    int max_id=max_element(ges_types,ges_types+40)-ges_types;
//    for(int i=0;i<40;i++)
//        cout<<" "<<ges_types[i]<<" ";
    cout<<"max_id"<<max_id<<" "<<"max_val "<<ges_types[max_id]<<endl;
    cout<<"gesture: "<<gesture_labels[max_id]<<endl;

    cv::Point origin;
    cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    origin.x=img.rows/6;
    origin.y=img.cols/6*5;


    cv::putText(img,gesture_labels[max_id],origin,cv::FONT_HERSHEY_COMPLEX,2.0,cv::Scalar(0,255,250));
    cout<<"put text success"<<endl;
//    cv::imwrite("../out/output/putted_img",img);
    SendImage(img);

//    std::cout<<"hello"<<std::endl;
    return SUCCESS;
}


