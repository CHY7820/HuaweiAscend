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

* File utils.h
* Description: handle file operations
*/
#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <vector>

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include "opencv2/imgproc/types_c.h"
#include "acl/acl.h"
#include "ascenddk/presenter/agent/presenter_types.h"
#include "ascenddk/presenter/agent/errors.h"
#include "ascenddk/presenter/agent/presenter_channel.h"

using namespace std;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define RGB_IMAGE_SIZE_U8(width, height) ((width) * (height) * 3)
#define RGB_IMAGE_SIZE_F32(width, height) ((width) * (height) * 3 * 4)
#define IMAGE_CHAN_SIZE_F32(width, height) ((width) * (height) * 4)

#define FRAME_LENGTH 100
#define PEOPLE_MOST 10
#define modelWidth_ 160
#define modelHeight_ 120

template<class Type>
std::shared_ptr<Type> MakeSharedNoThrow() {
    try {
        return std::make_shared<Type>();
    }
    catch (...) {
        return nullptr;
    }
}

#define MAKE_SHARED_NO_THROW(memory, memory_type) \
    do { \
            memory = MakeSharedNoThrow<memory_type>(); \
    }while(0);


typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
}Result;



typedef struct key_point{
    float x;
    float y;
	float score;
//    int num;
} key_pointsT;




//const string gesture_labels[] = {"applauding","bending back","blowing nose","carrying baby",
//"celebrating","clapping","crawling baby","crying","drinking","exercising arm","headbanging",
//"headbutting","high kick","jogging","laughing","lunge","pull ups","punching person (boxing)",
//"push up","shaking hands","shaking head","side kick","sign language interpreting","singing",
//"situp","slapping","smoking","sneezing","sniffing","somersaulting","squat","stretching arm",
//"stretching leg","swinging legs","swinging on something","tai chi","tasting food","washing hands",
//"writing","yawning"};

const string gesture_labels[] = {"applauding","bending back","celebrating","clapping","crying",
"drinking","headbanging","headbutting","high kick","jogging","laughing","pull ups","push up",
"shaking head","side kick","sign language interpreting","sit up","squat","stretching arm",
"stretching leg","tai chi","tasting food","writing","yawning"};

//const string gesture_labels[] {" ","swiping down","swiping left","swiping right","swiping up"};

/**
 * Utils
 */
namespace Utils {

    /**
    * @brief create device buffer of pic
    * @param [in] picDesc: pic desc
    * @param [in] PicBufferSize: aligned pic size
    * @return device buffer of pic
    */
    bool IsDirectory(const std::string &path);

    bool IsPathExist(const std::string &path);

    void SplitPath(const std::string &path, std::vector<std::string> &path_vec);

    void GetAllFiles(const std::string &path, std::vector<std::string> &file_vec);

    void GetPathFiles(const std::string &path, std::vector<std::string> &file_vec);
    void* CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy);
    void* CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize);
    void* CopyDataHostToDevice(void* deviceData, uint32_t dataSize);
    void* CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize);

    void write_motion_data(float motion_data[1][3][FRAME_LENGTH][18]);
    void write_array_data(float* array,int n,string name);

    void ProcessMotionData(float motion_data[1][3][FRAME_LENGTH][18]);

    void put_text(cv::Mat& frame,string text);

}; // namespace Utils

