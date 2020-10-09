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

using namespace std;

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)

#define RGB_IMAGE_SIZE_U8(width, height) ((width) * (height) * 3)
#define RGB_IMAGE_SIZE_F32(width, height) ((width) * (height) * 3 * 4)
#define IMAGE_CHAN_SIZE_F32(width, height) ((width) * (height) * 4)


#define FRAME_LENGTH 100
#define PEOPLE_MOST 10


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

struct Resolution {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ImageDesc {
    uint32_t img_width = 0;
    uint32_t img_height = 0;
    int32_t size = 0;
   // std::string input_path = "";
//    std::shared_ptr<float> data;
    std::shared_ptr<uint8_t> data;
};



struct Rect {
    uint32_t ltX = 0;
    uint32_t ltY = 0;
    uint32_t rbX = 0;
    uint32_t rbY = 0;
};



typedef struct key_points{
    float point_x;
    float point_y;
	float score;
    int num;
} key_pointsT;



typedef struct connection{
    int point_1;
    int point_2;
	// score是连接的分数
    float score;
	//它两头的关键点分数
	float score1;
	float score2;
} connectionT;

typedef struct ohuamn {
	//全部为负数
	// 存的是全局序号
	int p[18] = { -1 };
	float allscore = 0;
	int keynum = 0;
	int flag = -1;
} human;

// jiashi changed
typedef struct EngineTransNew
{
    //    std::vector<std::vector<std::vector<float>>> data;
	// 最后一维是关键点
	// 倒数第二维是帧
	// 正数第二维是x与y
	// 那第一维应该是人？
    float data [1][3][FRAME_LENGTH][18];
    //    float data [2][30][14];
    size_t buffer_size ;   // buffer size
}EngineTransNewT;


struct BBox {
    Rect rect;
    uint32_t score;
    string text;
};

const string gesture_labels[] = {"applauding","bending back","blowing nose","carrying baby",
"celebrating","clapping","crawling baby","crying","drinking","exercising arm","headbanging",
"headbutting","high kick","jogging","laughing","lunge","pull ups","punching person (boxing)",
"push up","shaking hands","shaking head","side kick","sign language interpreting","singing",
"situp","slapping","smoking","sneezing","sniffing","somersaulting","squat","stretching arm",
"stretching leg","swinging legs","swinging on something","tai chi","tasting food","washing hands",
"writing","yawning"};

/**
 * Utils
 */
class Utils {
public:

    /**
    * @brief create device buffer of pic
    * @param [in] picDesc: pic desc
    * @param [in] PicBufferSize: aligned pic size
    * @return device buffer of pic
    */
    static bool IsDirectory(const std::string &path);

    static bool IsPathExist(const std::string &path);

    static void SplitPath(const std::string &path, std::vector<std::string> &path_vec);

    static void GetAllFiles(const std::string &path, std::vector<std::string> &file_vec);

    static void GetPathFiles(const std::string &path, std::vector<std::string> &file_vec);
    static void* CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy);
    static void* CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize);
    static void* CopyDataHostToDevice(void* deviceData, uint32_t dataSize);
    static void* CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize);

    static void ImageNchw(shared_ptr<ImageDesc>& imageData, std::vector<cv::Mat>& nhwcImageChs, uint32_t size);

    //    static int ReadImageFile(ImageData& image, std::string fileName);
//    static Result CopyImageDataToDevice(ImageData& imageDevice, ImageData srcImage, aclrtRunMode mode);
};

