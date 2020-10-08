#include "gesture_process.h"
#include "utils.h"
#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"
using namespace std;

GestureProcess::GestureProcess() : ModelProcess() {}

Result GestureProcess::Inference(aclmdlDataset*& inferenceOutput, std::shared_ptr<EngineTransNewT> motion_data_new) {

    motion_data_new->buffer_size = 3 * FRAME_LENGTH * 18 * sizeof(float);

    Result ret = CreateInput((void*) motion_data_new->data, motion_data_new->buffer_size);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    ret = Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }

    inferenceOutput = GetModelOutputData();

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

    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(modelOutput,0);
    if (dataBuffer == nullptr) {
        ERROR_LOG("get model output aclmdlGetDatasetBuffer failed");
        return FAILED;
    }
    void* data = aclGetDataBufferAddr(dataBuffer);
    if (data == nullptr) {
        ERROR_LOG("aclGetDataBufferAddr from dataBuffer failed.");
        return FAILED;
    }

    std::vector<int> res;
    aclError ret = aclrtMemcpy(&res, sizeof(res), data, sizeof(res), ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("result labels aclrtMemcpy failed!");
        return FAILED;
    }
    cout<<res[0]<<endl;

    std::vector<int>::iterator maxVal = std::max_element(std::begin(res), std::end(res));
    int k=std::distance(std::begin(res),maxVal);

    cv::Point origin;
    cv::Mat img = cv::imread(path);
    origin.x=img.rows/6;
    origin.y=img.cols/6*5;


    cv::putText(img,gesture_labels[k],origin,cv::FONT_HERSHEY_COMPLEX,2.0,cv::Scalar(0,255,250));
    cout<<"put text success"<<endl;

    SendImage(img);
    std::cout<<"hello"<<std::endl;
    return SUCCESS;
}


