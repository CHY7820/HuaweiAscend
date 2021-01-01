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

void Utils::write_motion_data(float motion_data[1][3][FRAME_LENGTH][18])
{
    static int dataid=0;

    cout<<"writing motion data..."<<"id: "<<dataid<<endl;
    ofstream fout;
    string filename="./output/motion_data_";
    filename+=to_string(dataid)+".txt";
    fout.open(filename);
    fout.setf(ios::fixed);
    fout.precision(8);
    if(dataid<32)
    {
        for(int i=0;i<3;i++)
        {
            for(int j=0;j<FRAME_LENGTH;j++)
            {
                for(int k=0;k<18;k++)
                {
                    fout<<motion_data[0][i][j][k]<<" ";
                }

            }
        }
        //cout<<"write motion data success!"<<endl;
    }
    dataid++;

}

void Utils::write_array_data(float* array,int n,string name)
{
    ofstream fout;
    string filename="./output/"+name+".txt";
    fout.open(filename);
    fout.setf(ios::fixed);
    fout.precision(8);
    for(int i=0;i<n;i++)
    {
        fout<<array[i]<<" ";
    }

}

// 标准化骨骼关键点序列
void Utils::ProcessMotionData(float motion_data[1][3][FRAME_LENGTH][18]) {
    for(int i=0;i<2;i++) {
        for(int j=0;j<FRAME_LENGTH;j++) {
            for(int k=0;k<18;k++)
            {
                if(motion_data[0][i][j][k]==0) continue;
                if(i==0)
                    motion_data[0][i][j][k]=motion_data[0][i][j][k]/modelWidth_-0.5;
                else
                    motion_data[0][i][j][k]=-0.5+motion_data[0][i][j][k]/modelHeight_;
            }
        }

    }

//    for(int k=0;k<18;k++)
//    {
//        cout<<"x-y-s: "<<motion_data[0][0][0][k]<<"-"<<motion_data[0][1][0][k]<<"-"<<motion_data[0][2][0][k]<<endl;
//    }

}

void Utils::put_text(cv::Mat& image,string text)
{
    cv::Point origin;
    origin.x=image.cols/8;
    origin.y=image.rows/10;
    cv::putText(image,text,origin,cv::FONT_HERSHEY_COMPLEX,1.0,cv::Scalar(100,50,130));
}
