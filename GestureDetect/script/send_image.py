#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
# 传输图片
import paramiko

# 修改Atlas 200DK的IP以及要传输的目标位置
Atlas_IP = "192.168.1.2"
Atlas_path = "HIAI_PROJECTS/workspace_mind_studio/GestureDetect_e8aa832f$/data/"

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
    start = time.time()
    while success:
        send_time += 1
        # 读取图片
        img_np = frame.copy()
        # 图片保存的位置
        os.mkdir("./frames") # 创建文件夹
        img_path = '%s%d.jpg'%('./frames/', num)
        cv2.imwrite(img_path, img_np)
        # 传输
        try:
            sftp.put(img_path, Atlas_path+'%d.jpg'%num)
        except IOError:
            print ("==ERROR transport==")
            success, frame = cap.read()
            continue
        print ("发送第%d张图片"%num)
        num += 1
        # 一百张图片一次循环
        num = num % 100

        cv2.namedWindow('capture face detection', cv2.WINDOW_NORMAL)
        cv2.imshow('capture face detection', img_np)
        if cv2.waitKey(1) >= 0:
            break
        success, frame = cap.read()
        # 一秒发送25张
        time.sleep(0.028)
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    read_capture()

