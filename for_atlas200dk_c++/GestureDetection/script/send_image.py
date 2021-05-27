#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import os
# 传输图片
import paramiko

MAX_FRAME = 100
# 修改Atlas 200DK的IP以及要传输的目标位置
Atlas_IP = "192.168.1.2"
Atlas_path = "HIAI_PROJECTS/workspace_mind_studio/GestureDetect_e8aa832f/data/frames/"

def read_capture():
    num = 0
    send_time = 0
    # 读取指定端口视频流
    cap = cv2.VideoCapture(0) # 端口修改为适配于您主机的摄像头ID
    success, frame = cap.read()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(Atlas_IP,  22, "HwHiAiUser", "Mind@123")
    # 打开SSH端口
    sftp = ssh.open_sftp()
    while success:
        flag=0
        start = time.time()
        send_time += 1
        # 读取图片
        img_np = frame.copy()
        # 图片保存的位置
        # os.mkdir("../data/frames") # 创建文件夹
        img_path = '%s%d.jpg'%('../data/frames/',num)
        cv2.imwrite(img_path, img_np)
        img_path1 = '%s%d.jpg'%('../data/frames/',(num-1)%100)
        # while os.path.exists(img_path):
        #     print ('waiting to be read')
        #     pass
        # 传输
        try:
            sftp.put(img_path, Atlas_path+'%d.jpg'%num)
            os.unlink(img_path)
        except IOError:
            print ("==ERROR transport==")
            flag=1
           # sftp.put(img_path1, Atlas_path+'%d.jpg'%num)
            #continue
        print ("发送第%d张图片"%num)
        print ("send time: ",time.time()-start,'s')
        if flag==0:
            num += 1
        if num >= MAX_FRAME:
            num = num % MAX_FRAME # number from 0 to MAX_FRAME-1

        cv2.namedWindow('capture gesture', cv2.WINDOW_NORMAL)
        cv2.imshow('capture gesture', img_np)
        if cv2.waitKey(1) >= 0:
            break
        success, frame = cap.read()
    cv2.destroyAllWindows()
    cap.release()

def clear_image_file():
    for num in range(MAX_FRAME):
        path = '%s%d.jpg'%('../data/frames/',num)
        if os.path.exists(path):
            os.unlink(path)

def read_capture2():
    num = 0
    # send_time = 0
    # 读取指定端口视频流
    cap = cv2.VideoCapture(0) # 端口修改为适配于您主机的摄像头ID
    success, frame = cap.read()
    trans=paramiko.Transport((Atlas_IP,22))
    trans.connect(username='HwHiAiUser',password='Mind@123')
    sftp=paramiko.SFTPClient.from_transport(trans)
    while success:
        start = time.time()
        if not os.path.exists('../data/frames'):
            os.mkdir("../data/frames") # 创建文件夹
        local_image_path = '%s%d.jpg'%('../data/frames/',num)
        cv2.imwrite(local_image_path, frame)
        # 传输
        try:
            sftp.put(localpath=local_image_path, remotepath=Atlas_path+'%d.jpg'%num) # local file, remote file
            os.unlink(local_image_path)
        except IOError:
            print ("Transport Error")
            success, frame = cap.read()
            continue
        print ("发送第%d张图片"%num)
        num += 1
        if num >= MAX_FRAME:
            num = num % MAX_FRAME # number from 0 to MAX_FRAME-1

        cv2.namedWindow('capture face detection', cv2.WINDOW_NORMAL)
        cv2.imshow('capture face detection', frame)
        if cv2.waitKey(1) >= 0:
            break
        success, frame = cap.read()
        print ('trans time: ',time.time()-start)
    cv2.destroyAllWindows()
    cap.release()
    trans.close()



if __name__ == '__main__':
    clear_image_file()
    read_capture2()

