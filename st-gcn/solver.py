import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from model.stgcn import stgcn as Model

# 训练过程中的硬件设置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  #是否采用增长的方式分配显存
config.allow_soft_placement = True  #如果你指定的设备不存在，允许TF自动分配设备

# 变量，值为True或者False，用来标记是否在训练还是在测试阶段，因为模型中含有BN层和dropout层
is_training = tf.Variable(True, name='input_is_traing', trainable=False)
# 定义op，用来改变标记位is_training的数值
assign_true_training = tf.assign(is_training, True)
assign_false_training = tf.assign(is_training, False)

# 导入tfrecord格式数据集
def get_dataset(directory, num_classes, batch_size, drop_remainder=False, shuffle=False, shuffle_size=2000):
    # 数据集里面保存的特征字典列表
    feature_description = {
        'features': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    # parse each proto and the features within
    # 相当于写入时的逆过程,解析dataset的数据，解析出data, label
    # 函数里面定义函数，从外部代码隐藏，get_dataset函数外不能调用这个函数
    def _parse_feature_function(example_proto):
        # 解析tfrecord文件的每条记录，即序列化后的tf.train.Example；使用tf.parse_single_example来解析：
        features = tf.io.parse_single_example(example_proto, feature_description)
        data = tf.io.parse_tensor(features['features'], tf.float32)
        label = tf.one_hot(features['label'], num_classes)
        # data = tf.reshape(data, (3, 100, 18, 1))
        # data = tf.squeeze(data, 3)
        return data, label

    # records是一个数组，包含num_shards个tfrecord文件的地址
    records = [
        os.path.join(directory, file) for file in os.listdir(directory)
        if file.endswith("tfrecord")
    ]
    #读取所有tfrecord文件得到dataset，num_parallel_reads是线程数，也可以不用开
    dataset = tf.data.TFRecordDataset(records, num_parallel_reads=len(records))
    dataset = dataset.map(_parse_feature_function)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(shuffle_size)
    return dataset


class Solver:
    def __init__(self,
                 num_classes=5,
                 num_frames=30,
                 checkpoints_path="./checkpoints",
                 log_dir='./logs',
                 ):

        # 参数
        self.num_classes = num_classes
        self.num_frames = num_frames

        # 文件
        self.checkpoints_path = checkpoints_path
        self.log_dir = log_dir

        # 模型
        self.model = Model(self.num_classes)

    def train(self,
              train_data_path,
              test_data_path,
              train_num,
              test_num,
              epochs,
              batch_size,
              N,
              base_lr=0.01
              ):
            
        train_batch_num = train_num // batch_size
        test_batch_num = test_num // batch_size

        # 生成读取测试集和测试集的迭代器
        train_dataset = get_dataset(train_data_path,
                                    self.num_classes,
                                    batch_size,
                                    drop_remainder=True,
                                    shuffle=True,
                                    shuffle_size=train_num)
        train_data_iterator = train_dataset.make_initializable_iterator()
        train_data = train_data_iterator.get_next()

        test_dataset = get_dataset(test_data_path,
                                   self.num_classes,
                                   batch_size,
                                   drop_remainder=False,
                                   shuffle=False)
        test_data_iterator = test_dataset.make_initializable_iterator()
        test_data = test_data_iterator.get_next()

        # 模型输入输出的占位
        x = tf.placeholder(tf.float32, [batch_size, self.num_frames, 18, 3], name='input_features')
        y = tf.placeholder(tf.int32, [batch_size, self.num_classes], name='input_labels')

        # 定义损失函数和优化器
        logits = self.model.call(x, is_training=is_training)
        softmax_prob = tf.nn.softmax(logits, name='softmax_prob')
        class_id = tf.argmax(logits, 1, name='output_class_id')
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        optimizer = tf.train.AdamOptimizer(learning_rate=base_lr)

        # train_op必须在update_ops完成之后在进行
        with tf.name_scope('train_op'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss)  # 将损失函数降到最低

        loss_summary = tf.summary.scalar('train_loss', loss)
        acc_summary = tf.summary.scalar('train_acc', accuracy)
        merged_summary = tf.summary.merge([loss_summary, acc_summary])

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_all_variables())
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            train_iter = 0
            for epoch in range(epochs):
                sess.run([
                    train_data_iterator.initializer,
                    test_data_iterator.initializer
                ])
                print("Epoch: {}".format(epoch + 1))
                print("Training: ")
                for i in tqdm(range(train_batch_num)):
                    [features, labels] = sess.run(train_data)
                    # print(labels)
                    sess.run(assign_true_training)
                    train_loss, train_acc, _, logits_train, class_id_train, merged_summary_train = sess.run(
                        [loss, accuracy, train_op, logits, class_id, merged_summary],
                        feed_dict={
                            x: features,
                            y: labels
                        }
                    )
                    # print('*************训练准确率********', train_acc)
                    summary_writer.add_summary(merged_summary_train, train_iter + 1)
                    train_iter += 1

                print("Testing: ")
                # 下列两个参数用于统计测试时准确率和损失的平均值
                test_acc_average = 0
                test_loss_average = 0
                for i in tqdm(range(test_batch_num)):
                    [features, labels] = sess.run(test_data)
                    # print(labels)
                    sess.run(assign_false_training)
                    test_loss, test_acc, correct_prediction_test, logits_test = sess.run(
                        [loss, accuracy, correct_prediction, logits],
                        feed_dict={
                            x: features,
                            y: labels
                        }
                    )
                    test_acc_average += test_acc
                    test_loss_average += test_loss
                print('测试准确率 : ', test_acc_average / test_batch_num)

                test_summary_loss = tf.Summary(value=[tf.Summary.Value(tag='test_loss',
                                                                       simple_value=test_loss_average / test_batch_num)])
                test_summary_acc = tf.Summary(value=[tf.Summary.Value(tag='test_acc',
                                                                      simple_value=test_acc_average / test_batch_num)])
                summary_writer.add_summary(test_summary_loss, epoch + 1)
                summary_writer.add_summary(test_summary_acc, epoch + 1)

                # 每N个批次保存一次模型
                if epoch % N == 0:
                    checkpoint_write_path = os.path.join(self.checkpoints_path, 'ckpt-%05d' % (epoch + 1))
                    if not os.path.exists(checkpoint_write_path):
                        os.makedirs(checkpoint_write_path)
                    saver = tf.train.Saver(max_to_keep=1000)
                    saver.save(sess, os.path.join(checkpoint_write_path, 'ckpt-%05d' % (epoch + 1)))
                print('model saved to ckpt-%05d' % (epoch + 1))

    def convert_pb(self, epoch, output_graph):

        checkpoint_path = self.checkpoints_path + "/ckpt-%05d" % epoch  #要固化为pb的checkpoints模型的路径

        # 模型输入输出的占位
        x = tf.placeholder(tf.float32, [None, self.num_frames, 18, 3], name='input_features')
        y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_labels')

        # 定义损失函数和优化器
        logits = self.model.call(x, is_training=is_training)
        softmax_prob = tf.nn.softmax(logits, name='softmax_prob')
        class_id = tf.argmax(logits, 1, name='output_class_id')

        with tf.Session(config=config) as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, os.path.join(checkpoint_path, 'ckpt-%05d' % epoch))
            print('success')
            graph = tf.get_default_graph()  # 获得默认的图
            input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
            print("%d ops in the input graph." % len(input_graph_def.node))  # 得到当前图有几个操作节点
            output_node_names = "output_logits/BiasAdd"
            output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess=sess,
                input_graph_def=input_graph_def,  # 等于:sess.graph_def
                output_node_names=output_node_names.split(",")
            )  # 如果有多个输出节点，以逗号隔开
            print('****************output_graph_def***********************')
            print("%d ops in the out graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点
            output_graph_def_end = tf.graph_util.remove_training_nodes(output_graph_def)
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def_end.SerializeToString())  # 序列化输出

    def inference_pb(self, model_path, data):
        
        with gfile.FastGFile(model_path,'rb') as f:
            g = tf.GraphDef()
            g.ParseFromString(f.read())
            tf.import_graph_def(g,name='')

        np.set_printoptions(threshold=np.inf)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("input_features:0")
            out = graph.get_tensor_by_name("output_logits/BiasAdd:0")
            pred = sess.run(out, feed_dict={x: data})
            print('******predict_inferance*********\n',pred)
            predict_lable = np.argmax(pred, axis=1)
            print('******predict_lable*********\n',predict_lable)
