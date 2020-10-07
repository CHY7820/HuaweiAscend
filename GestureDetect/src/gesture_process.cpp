
#include "gesture_process.h"




GestureProcess::GestureProcess() : ModelProcess() {}
//GestureProcess::GestureProcess(uint32_t modelId) : ModelProcess(modelId) {}


Result GestureProcess::Inference(aclmdlDataset*& inferenceOutput, std::shared_ptr<EngineTransNewT> motion_data_new) {

    motion_data_new->buffer_size = 2 * FRAME_LENGTH * 14 * sizeof(float);

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


//
Result GestureProcess::Postprocess(aclmdlDataset* modelOutput){
//    uint32_t dataSize = 0;
//    //
//    float* newresult = (float *)GetInferenceOutputItem(dataSize, modelOutput, 0);
//    int maxPosition = max_element(newresult, newresult+5) - newresult;
//
//    // 人工Softmax
//    float down = 1.4;
//    float result_total = pow(down, newresult[0]) + pow(down, newresult[1]) + pow(down, newresult[2]) + pow(down, newresult[3]) + pow(down, newresult[4]);
//    newresult[0] = pow(down, newresult[0]) / result_total;
//    newresult[1] = pow(down, newresult[1]) / result_total;
//    newresult[2] = pow(down, newresult[2]) / result_total;
//    newresult[3] = pow(down, newresult[3]) / result_total;
//    newresult[4] = pow(down, newresult[4]) / result_total;
//
//    bool if_need_pub = false;
//
//    if (newresult[maxPosition] >= 0.5){
//        switch (maxPosition){
//            case 0:
//            if(newresult[maxPosition] > 0.9){
//                if (LAST_GES != 0){
//                    cout << " 鼓掌 " << newresult[0] << endl;
//                    cout << " 挥手 " << newresult[1] << endl;
//                    cout << " 站立 " << newresult[2] << endl;
//                    cout << " 双手平举 " << newresult[3] << endl;
//                    cout << " 踢腿 " << newresult[4] << endl;
//                    cout << "=============================鼓掌" << endl;
//                    if_need_pub = true;
//                    LAST_GES = 0;
//                }
//            }
//            break;
//            case 1:
//            if(newresult[maxPosition] > 0.8){
//                if (LAST_GES != 1){
//                    cout << " 鼓掌 " << newresult[0] << endl;
//                    cout << " 挥手 " << newresult[1] << endl;
//                    cout << " 站立 " << newresult[2] << endl;
//                    cout << " 双手平举 " << newresult[3] << endl;
//                    cout << " 踢腿 " << newresult[4] << endl;
//                    cout << "=============================挥手" << endl;
//                    if_need_pub = true;
//                    LAST_GES = 1;
//                }
//            }
//            break;
//            case 2:
//            if(newresult[maxPosition] > 0.5){
//                if (LAST_GES != 2){
//                    cout << " 鼓掌 " << newresult[0] << endl;
//                    cout << " 挥手 " << newresult[1] << endl;
//                    cout << " 站立 " << newresult[2] << endl;
//                    cout << " 双手平举 " << newresult[3] << endl;
//                    cout << " 踢腿 " << newresult[4] << endl;
//                    cout << "=============================站立" << endl;
//                    if_need_pub = false;
//                    LAST_GES = 2;
//                }
//            }
//            break;
//            case 3:
//            if(newresult[maxPosition] > 0.95){
//                if (LAST_GES != 3){
//                    cout << " 鼓掌 " << newresult[0] << endl;
//                    cout << " 挥手 " << newresult[1] << endl;
//                    cout << " 站立 " << newresult[2] << endl;
//                    cout << " 双手平举 " << newresult[3] << endl;
//                    cout << " 踢腿 " << newresult[4] << endl;
//                    cout << "=============================双手平举" << endl;
//                    if_need_pub = true;
//                    LAST_GES = 3;
//                }
//            }
//            break;
//            case 4:
//            if(newresult[maxPosition] > 0.9){
//                if (LAST_GES != 4){
//                    cout << " 鼓掌 " << newresult[0] << endl;
//                    cout << " 挥手 " << newresult[1] << endl;
//                    cout << " 站立 " << newresult[2] << endl;
//                    cout << " 双手平举 " << newresult[3] << endl;
//                    cout << " 踢腿 " << newresult[4] << endl;
//                    cout << "==============================踢腿" << endl;
//                    if_need_pub = true;
//                    LAST_GES = 4;
//                }
//            }
//            break;
//            default:
//            cout << "max element==================nothing  " << maxPosition << "     " << newresult[maxPosition] << endl;
//            break;
//        }
//    }
//  resultImage = resultImage * 255;
    //    SendImage(resultImage);
    return SUCCESS;
}


