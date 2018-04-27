# coding=utf-8
import os
import numpy as np
import tensorflow as tf


def get_files(file_dir):

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if 'cat' in name[0]:
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            if 'dog' in name[0]:
                dogs.append(file_dir + file)
                label_dogs.append(1)
        image_list = np.hstack((cats, dogs))
        label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    print(temp)
    temp = temp.transpose()
    # 打乱顺序
    print(temp)
    np.random.shuffle(temp)

    # 取出第一个元素作为 image 第二个元素作为 label
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    print(label_list)
    label_list = [int(i) for i in label_list]
    print(image_list)

    return image_list, label_list


# image_W ,image_H 指定图片大小，batch_size 每批读取的个数 ，capacity队列中 最多容纳元素的个数
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换数据为 ts 能识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    # 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
    image = tf.image.decode_jpeg(image_contents, channels=3) #jpeg或者jpg格式都用decode_jpeg函数
    # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
    image = tf.image.per_image_standardization(image)
    # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    # 重新定义下 label_batch 的形状
    label_batch = tf.reshape(label_batch, [batch_size])
    # 转化图片
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


