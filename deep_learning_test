import numpy as np
import os
import cv2
import random
import tensorflow.compat.v1 as tf
import pandas as pd

#gaussian noise + normalize

#防止报错
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def normalize(img):
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[0])):
            tmp.append(img[i][j])
    average=np.mean(tmp)
    std=np.std(tmp, ddof=1)
    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j]=(img[i][j]-average)/std
    return(img)

def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv

def add_gaussian_noise(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            s = np.random.normal(0, 5, 1)  # 产生一个正态分布
            img[i][j]=clamp(s+img[i][j])
    return(img)

# 调用该方法，获取文件夹中猫狗的像素矩阵为数据，并为每条数据添加标签
def load_data(data_dir0,data_dir1):
    # 存放所有图片像素矩阵数据
    data_all = []
    # 存放其对应的标签
    labels_all = []
    # i 为每个图片数据的名字
    for i in os.listdir(data_dir0):
        # 拼接路径 保证cv2读取正确
        img = cv2.imread(os.path.join(data_dir0, i), 0)
        img = add_gaussian_noise(img)
        # 将读取到的大小不一的图像进行统一缩放
        img = cv2.resize(img, (128, 128))
        img=normalize(img)
        tmp = []
        tmp.append(img)
        tmp.append(img)
        tmp.append(img)
        # 转为数字矩阵
        img_array = np.asarray(tmp)
        # 直接对数据进行归一化
        img_array = img_array / 127.5 - 1
        # 将这个数据存放进列表
        data_all.append(img_array)
        labels_all.append(0)

    for i in os.listdir(data_dir1):
        # 拼接路径 保证cv2读取正确
        img = cv2.imread(os.path.join(data_dir1, i),0)
        img=add_gaussian_noise(img)
        # 将读取到的大小不一的图像进行统一缩放
        img = cv2.resize(img, (128, 128))
        img = normalize(img)
        tmp=[]
        tmp.append(img)
        tmp.append(img)
        tmp.append(img)
        # 转为数字矩阵
        img_array = np.asarray(tmp)
        # 直接对数据进行归一化
        img_array = img_array / 127.5 - 1
        # 将这个数据存放进列表
        data_all.append(img_array)
        labels_all.append(1)
    # 将列表作为转为数组
    data_all = np.asarray(data_all)
    labels_all = np.asarray(labels_all)
    labels_all = np.vstack(labels_all)
    return data_all, labels_all

# 调用该方法，对数据进行洗牌
def shuffle(data, labels):
    m = len(data)
    o = np.random.permutation(m)
    data = data[o]
    labels = labels[o]
    return data, labels

# 调用该方法对数据进行切分
def data_split(data, labels):
    m = len(data)
    l = int(m * 0.9)
    x_train, x_test = data[0:l], data[l:]
    y_train, y_test = labels[0:l], labels[l:]
    return x_train, x_test, y_train, y_test


# 调用load_data处理数据
data_all, labels_all = load_data("D:/test/images-0712/negtive","D:/test/images-0712/positive")

# 调用洗牌函数对数据进行打乱
data_all, labels_all = shuffle(data_all, labels_all)
# 调用数据切分方法，对数据进行切分
x_train, x_test, y_train, y_test = data_split(data_all, labels_all)

# 设置网络的参数，学习率设置为0.0001，训练周期设为100，batch_szie设为100
learning_rate = 0.00003
n_times = 300
batch_size = 100

# 定义网络占位符
X = tf.placeholder(tf.float32, [None, 3, 128, 128])
X_img = tf.transpose(X, perm=[0, 2, 3, 1])
Y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)

# 定义神经网络结构模型
# 首先构建卷积层，用于特征提取，示例如下，可以调整卷积参数及层数，这里选择两层，激活函数选择relu
conv1 = tf.layers.conv2d(X_img, 32, (3, 3), strides=(1, 1), padding='valid', activation=tf.nn.relu)
# 加入batch_normalization防止卷积层过拟合
conv1 = tf.layers.batch_normalization(conv1,momentum=0.9)
pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='valid')
conv2 = tf.layers.conv2d(pool1, 64, (3, 3), (1, 1), padding='valid', activation=tf.nn.relu)
conv2 = tf.layers.batch_normalization(conv2,momentum=0.9)
pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='valid')
# 将卷积数据展平，用于构建全连接层，用于分类
flatten = tf.layers.flatten(pool2)
fc1 = tf.layers.dense(flatten, 128, activation=tf.nn.relu)
# 全连接层可以加入dropout函数用于切断一些神经元的联系，防止过拟合
fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
# 二分类问题，最终归于一个输出即可，激活函数选择sigmoid
fc2 = tf.layers.dense(fc1, 1, activation=tf.nn.sigmoid)
# 编写定义损失函数，网络优化器采用Adam
loss = - tf.reduce_mean(Y * tf.log(fc2) + (1 - Y) * tf.log(1 - fc2))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 预测
predict = tf.cast(fc2 > 0.5, dtype=tf.float32)
# 准确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))
# 开启会话1
record=[]
with tf.Session() as sess:
    # 创建会话并初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 大循环训练100次
    for n in range(n_times):
        # 起始位置
        start = 0
        # 结束位置
        end_ = batch_size
        # acc_all存放准确率
        acc_all = []
        # 每次分批次循环训练
        # 调用洗牌函数对数据进行打乱
        data_all, labels_all = shuffle(data_all, labels_all)
        # 调用数据切分方法，对数据进行切分
        x_train, x_test, y_train, y_test = data_split(data_all, labels_all)
        for j in range(len(x_train) // batch_size):
            # 获取数据
            # 执行一个逻辑判断，如果下标越界，则重置起始与结束位置
            if start >= len(x_train):
                start = 0
                end_ = batch_size
            # 分批次获取训练数据中的数据
            data_batch, labels_batch = x_train[start:end_], y_train[start:end_]
            # 执行训练
            los, _, acc = sess.run([loss,train_op,accuracy], feed_dict={X: data_batch, Y: labels_batch, keep_prob: 0.9})
            acc_all.append(acc)
        accs = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1})
        # 每个大批次训练完毕，打印训练集平均准确率和测试集准确率
        print('大批次', n + 1)
        print('训练集准确率：', np.mean(acc_all))
        print('测试集的准确率:', accs)
        #为了作图
        tmp=[]
        tmp.append(n+1)
        tmp.append(np.mean(acc_all))
        tmp.append(accs)
        record.append(tmp)

    #将记录的内容转化为csv
    name=["批次","训练集准确率","测试集准确率"]
    csv = pd.DataFrame(columns=name, data=record)
    csv.to_csv("D:/test/images-0712/deep_learning_record_valid.csv", encoding='gbk')

    # 测试集中随机抽一个样本并进行测试，输出结果
    r = random.randint(0, len(x_test) - 1)
    print('随机标签', y_test[r])
    print('预测标签', sess.run(predict, feed_dict={X: x_test[r], keep_prob: 1}))
