import sys
sys.path.extend(['./'])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.device('/gpu:0')

from solver import Solver

if __name__ == "__main__":
    train_data_path = './dataset/record_data/train_data_record/'
    test_data_path = './dataset/record_data/val_data_record/'
    train_num = 3096
    test_num = 774
    epochs = 10
    batch_size = 32
    N = 1
    solver = Solver()
    solver.train(train_data_path, test_data_path, train_num, test_num, epochs, batch_size, N)