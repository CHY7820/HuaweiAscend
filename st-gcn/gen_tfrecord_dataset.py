import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

act_list = ['swiping_up', 'swiping_down', 'swiping_left', 'swiping_right', 'do_nothing']


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(features, label):
    feature = {
        'features':
        _bytes_feature(tf.io.serialize_tensor(features.astype(np.float32))),
        'label':
        _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(
        feature=feature)).SerializeToString()


def get_data(filename, num_frames):
    with open(filename) as f:
        tmp = f.read().split()
        tmp2 = []
        for j in tmp:
            tmp2.append(float(j))
    return np.array(tmp2).reshape(num_frames, 18, 3)


def gen_tfrecord_data(num_shards, joint_data_path, record_data_path, shuffle, num_frames=100):
    # 初始化
    datac = list()
    labelsc = list()

    # 读取数据集
    for act in act_list:
        data_path = Path(os.path.join(joint_data_path, act))
        if not (data_path.exists()):
            print('Data file does not exist')
            return
        for root, dirs, files in os.walk(data_path):
            for f in files:
                filename = os.path.join(root, f)
                data = get_data(filename, num_frames)
                datac.append(data)
                labelsc.append(act_list.index(act))
    
    # 转换数据
    data = np.array(datac)
    labels = np.array(labelsc)

    # 检查数据长度
    if len(labels) != len(data):
        print("Data and label lengths didn't match!")
        return

    # 数据洗牌
    if shuffle:
        p = np.random.permutation(len(labels))
        labels = labels[p]
        data = data[p]

    # 打印shape
    print("Data shape:", data.shape)

    # 划分train/val
    len_train = int(len(data) * 0.8)
    train_data = data[:len_train]
    val_data = data[len_train:]
    train_labels = labels[:len_train]
    val_labels = labels[len_train:]

    # 检查数据长度
    if len(train_data) != len(train_labels) or len(val_data) != len(val_labels):
        print("Data and label lengths didn't match!")
        return

    # 分别打印shape
    print("Train Data Shape:", train_data.shape)
    print("Val Data Shape:", val_data.shape)

    # 建立目录
    if not ((Path(record_data_path)).exists()):
        os.mkdir(record_data_path)

    # 生成数据集
    for tar in ('train', 'val'):
        # 选择数据
        if tar == 'train':
            tar_data = train_data
            tar_labels = train_labels
        else:
            tar_data = val_data
            tar_labels = val_labels

        # 选择目录
        dest_path = os.path.join(record_data_path, tar + '_data_record/')
        if not ((Path(dest_path)).exists()):
            os.mkdir(dest_path)

        # 生成文件
        step = len(tar_labels) // num_shards  # step是每个tfrecord文件中保存的样本个数
        for shard in range(num_shards):
            # 循环生成tfrecord文件，一共生成num_shards个
            tfrecord_data_path = os.path.join(dest_path, str(shard) + ".tfrecord")
            with tf.io.TFRecordWriter(tfrecord_data_path) as writer:
                for i in tqdm(range(shard * step, (shard * step) + step if shard < num_shards - 1 else len(tar_labels))):
                    writer.write(serialize_example(tar_data[i], tar_labels[i]))


if __name__ == '__main__':
    tf.enable_eager_execution() # 动态图机制
    # 参数设置
    from solver import Solver
    solver = Solver()
    num_frames = solver.num_frames
    num_shards = 1
    joint_data_path = './dataset/joint_data/'
    record_data_path = './dataset/record_data/'
    shuffle = True
    # 数据集生成
    gen_tfrecord_data(num_shards, joint_data_path, record_data_path, shuffle, num_frames)
    print('...done')
