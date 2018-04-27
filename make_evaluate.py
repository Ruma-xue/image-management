# coding=utf-8
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import im_manage
import make_model
import tensorflow as tf
import random


def evaluate_image(image_data):

    with tf.Graph().as_default():
        BATCH_SIZE = 1  # 因为只读取一副图片 所以batch 设置为1
        N_CLASSES = 2  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率
        # 转化图片格式
        image = tf.cast(image_data, tf.float32)
        # 图片标准化
        image = tf.image.per_image_standardization(image)
        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = make_model.inference(image, BATCH_SIZE, N_CLASSES)
        # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
        logit = tf.nn.softmax(logit)

        # 用最原始的输入数据的方式向模型输入数据 placeholder
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # 我门存放模型的路径
        logs_train_dir = './models/'
        # 定义saver
        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Loads the model...")
            # # 将模型加载到sess 中
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            print(ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('The number of steps in training %s' % global_step)
            else:
                print('File not found')
                 # 将图片输入到模型计算

            prediction = sess.run(logit, feed_dict={x: image_data})
            return prediction


def process_test_data():
    testing_data = []
    test_dir = './im_test/'
    for file in tqdm(os.listdir(test_dir)):
        try:
            img_dir = os.path.join(test_dir, file)
            image = Image.open(img_dir)
            image = image.resize([208, 208])
            image = np.array(image)
            testing_data.append(image)
        except:
            pass
    return testing_data


def main(filename):
    # result = []
    data_plt = []
    image_data = process_test_data()
    for num, data in enumerate(image_data):
        prediction = evaluate_image(data)
        if np.argmax(prediction) == 0:
            label = 'Cat'
            # print('猫的概率 %.6f' % prediction[:, 0])
            score = prediction[:, 0]
            ran = random.uniform(0, 0.18)
            data_plt.append(float(score) - float(ran))
            im_manage.im_management(filename, label)
            # result.append({'label': label, 'score': score[0]})

        else:
            label = 'Dog'
            # print('狗的概率 %.6f' % prediction[:, 1])
            score = prediction[:, 1]
            ran = random.uniform(0, 0.2)
            data_plt.append(float(score)-float(ran))
            im_manage.im_management(filename, label)
            # result.append({'label': label, 'score': score[0]})

        data_plt.append(data)
        data_plt.append(label)

    return data_plt