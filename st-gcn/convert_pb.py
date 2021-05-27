import sys
sys.path.extend(['./'])

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.device('/gpu:0')

from solver import Solver

if __name__ == "__main__":
    epoch = 10
    output_graph = './pb/stgcn.pb'
    solver = Solver()
    solver.convert_pb(epoch, output_graph)