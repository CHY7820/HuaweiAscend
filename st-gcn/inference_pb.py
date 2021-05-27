import sys
sys.path.extend(['./'])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import tensorflow as tf

tf.device('/gpu:0')

from solver import Solver
from gen_tfrecord_dataset import get_data

if __name__ == '__main__':

    model_path='./pb/stgcn.pb'
    data_path='./dataset/joint_data'
    
    solver = Solver()
    frame = solver.num_frames

    # 读取数据集
    datac = list()
    if not ((Path(data_path)).exists()):
        print('Data file does not exist')
    for root, dirs, files in os.walk(data_path):
        for f in files:
            filename = os.path.join(root, f)
            print(filename)
            data = get_data(filename, frame)
            datac.append(data)
    data = np.array(datac)

    # # txt读取测试
    # print(data[0, 50, 6, 0:3])
    # print(data.shape)

    solver.inference_pb(model_path, data)
