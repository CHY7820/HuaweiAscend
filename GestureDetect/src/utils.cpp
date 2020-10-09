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

* File utils.cpp
* Description: handle file operations
*/
#include "utils.h"
#include <bits/stdint-uintn.h>
#include <map>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstring>
#include <dirent.h>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

using namespace std;

namespace {
const std::string kImagePathSeparator = ",";
const int kStatSuccess = 0;
const std::string kFileSperator = "/";
const std::string kPathSeparator = "/";
// output image prefix
const std::string kOutputFilePrefix = "out_";

}

bool Utils::IsDirectory(const string &path) {
    // get path stat
    struct stat buf;
    if (stat(path.c_str(), &buf) != kStatSuccess) {
        return false;
    }

    // check
    if (S_ISDIR(buf.st_mode)) {
        return true;
    } else {
    return false;
    }
}

bool Utils::IsPathExist(const string &path) {
    ifstream file(path);
    if (!file) {
        return false;
    }
    return true;
}

void Utils::SplitPath(const string &path, vector<string> &path_vec) {
    char *char_path = const_cast<char*>(path.c_str());
    const char *char_split = kImagePathSeparator.c_str();
    char *tmp_path = strtok(char_path, char_split);
    while (tmp_path) {
        path_vec.emplace_back(tmp_path);
        tmp_path = strtok(nullptr, char_split);
    }
}

void Utils::GetAllFiles(const string &path, vector<string> &file_vec) {
    // split file path
    vector<string> path_vector;
    SplitPath(path, path_vector);

    for (string every_path : path_vector) {
        // check path exist or not
        if (!IsPathExist(path)) {
        ERROR_LOG("Failed to deal path=%s. Reason: not exist or can not access.",
                every_path.c_str());
        continue;
        }
        // get files in path and sub-path
        GetPathFiles(every_path, file_vec);
    }
}

void Utils::GetPathFiles(const string &path, vector<string> &file_vec) {
    struct dirent *dirent_ptr = nullptr;
    DIR *dir = nullptr;
    if (IsDirectory(path)) {
        dir = opendir(path.c_str());
        while ((dirent_ptr = readdir(dir)) != nullptr) {
            // skip . and ..
            if (dirent_ptr->d_name[0] == '.') {
            continue;
            }

            // file path
            string full_path = path + kPathSeparator + dirent_ptr->d_name;
            // directory need recursion
            if (IsDirectory(full_path)) {
                GetPathFiles(full_path, file_vec);
            } else {
                // put file
                file_vec.emplace_back(full_path);
            }
        }
    } 
    else {
        file_vec.emplace_back(path);
    }
}

void* Utils::CopyDataDeviceToLocal(void* deviceData, uint32_t dataSize) {
    uint8_t* buffer = new uint8_t[dataSize];
    if (buffer == nullptr) {
        ERROR_LOG("New malloc memory failed");
        return nullptr;
    }

    aclError aclRet = aclrtMemcpy(buffer, dataSize, deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("Copy device data to local failed, aclRet is %d", aclRet);
        delete[](buffer);
        return nullptr;
    }

    return (void*)buffer;
}

void* Utils::CopyDataToDevice(void* data, uint32_t dataSize, aclrtMemcpyKind policy) {
    void* buffer = nullptr;
    aclError aclRet = aclrtMalloc(&buffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return nullptr;
    }

    aclRet = aclrtMemcpy(buffer, dataSize, data, dataSize, policy);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("Copy data to device failed, aclRet is %d", aclRet);
        (void)aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void* Utils::CopyDataDeviceToDevice(void* deviceData, uint32_t dataSize) {
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
}

void* Utils::CopyDataHostToDevice(void* deviceData, uint32_t dataSize) {
    return CopyDataToDevice(deviceData, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}


void Utils::ImageNchw(shared_ptr<ImageDesc>& imageData, std::vector<cv::Mat>& nhwcImageChs, uint32_t size) {
    uint8_t* nchwBuf = new uint8_t[size];
    int channelSize = IMAGE_CHAN_SIZE_F32(nhwcImageChs[0].rows, nhwcImageChs[0].cols);
    int pos = 0;
    for (int i = 0; i < nhwcImageChs.size(); i++) {
        memcpy(static_cast<uint8_t *>(nchwBuf) + pos,  nhwcImageChs[i].ptr<float>(0), channelSize);
        pos += channelSize;
    }
    imageData->size = size;
    imageData->data.reset((uint8_t *)nchwBuf, [](uint8_t* p) { delete[](p);} );
}

//Result Utils::CopyImageDataToDevice(ImageData& imageDevice, ImageData srcImage, aclrtRunMode mode) {
//    void * buffer;
//    if (mode == ACL_HOST)
//        buffer = Utils::CopyDataHostToDevice(srcImage.data.get(), srcImage.size);
//    else
//        buffer = Utils::CopyDataDeviceToDevice(srcImage.data.get(), srcImage.size);
//
//    if (buffer == nullptr) {
//        ERROR_LOG("Copy image to device failed");
//        return FAILED;
//    }
//
//    imageDevice.width = srcImage.width;
//    imageDevice.height = srcImage.height;
//    imageDevice.size = srcImage.size;
//    imageDevice.data.reset((uint8_t*)buffer, [](uint8_t* p) { aclrtFree((void *)p); });
//
//    return SUCCESS;
//}
//
//int Utils::ReadImageFile(ImageData& image, std::string fileName)
//{
//    //uint32_t width = 0, height = 0;
//    //GetJPEGWidthHeight(fileName.c_str(), &width, &height);
//    //INFO_LOG("jpeg width %d, height %d", width, height);
//
//    struct stat sBuf;
//    int fileStatus = stat(fileName.data(), &sBuf);
//    if (fileStatus == -1) {
//        ERROR_LOG("failed to get file");
//        return FAILED;
//    }
//    if (S_ISREG(sBuf.st_mode) == 0) {
//        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
//        return FAILED;
//    }
//    std::ifstream binFile(fileName, std::ifstream::binary);
//    if (binFile.is_open() == false) {
//        ERROR_LOG("open file %s failed", fileName.c_str());
//        return FAILED;
//    }
//
//    binFile.seekg(0, binFile.end);
//    uint32_t binFileBufferLen = binFile.tellg();
////    cout << "binFileBufferLen   " << binFileBufferLen << endl;
//    if (binFileBufferLen == 0) {
////        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
//        binFile.close();
//        return FAILED;
//    }
//
//    binFile.seekg(0, binFile.beg);
//
//    uint8_t* binFileBufferData = new(std::nothrow) uint8_t[binFileBufferLen];
//    if (binFileBufferData == nullptr) {
//        ERROR_LOG("malloc binFileBufferData failed");
//        binFile.close();
//        return FAILED;
//    }
//    binFile.read((char *)binFileBufferData, binFileBufferLen);
//    binFile.close();
//
//    int32_t ch = 0;
//    acldvppJpegGetImageInfo(binFileBufferData, binFileBufferLen,
//              &(image.width), &(image.height), &ch);
//    image.data.reset(binFileBufferData, [](uint8_t* p) { delete[](p); });
//    image.size = binFileBufferLen;
//
//    return SUCCESS;
//}
//int Utils::ReadImageFile(std::string fileName)
//{
//
//}