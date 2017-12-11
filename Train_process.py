#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/8 13:10
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction 主训练函数 √ ━━━━━☆*°☆*°
"""
from PIL import Image
import numpy as np
from Feature_extraction import *
import os
from sklearn.cross_validation import train_test_split
from sklearn import tree
import Adaboost
from datetime import datetime
import time
import cPickle
# 不采用pickle存储和读取，速度太慢了

neg_num = 0
pos_num = 0
# 包含预处理为24*24的灰度图并提取特征，保存为.npy格式的文件方便下次读取
# 测试中pickle存储和读取速度比较慢，所以决定采用numpy的array存储
def save_images_2_array(pos_dir, neg_dir):
    global neg_num, pos_num
    yield ('开始对图片进行预处理(转为为24*24灰度图)并提取图片的NPD特征...')
    # Global_Variables.Times.append(datetime.now())
    try:
        cnt = 0
        features_array_flag = False
        for root, dirs, files in os.walk(pos_dir, topdown=False):
            if pos_num == 0:
                pos_num = len(files)
                # print pos_num
            for file in files:
                im = Image.open(pos_dir + file).convert('L')
                im = im.resize((24, 24))
                im_array = np.array(im)
                img_feature = NPDFeature(im_array).extract()
                if not features_array_flag:
                    features_array = np.array(img_feature)
                    features_array_flag = True
                else:
                    features_array = np.row_stack((features_array, img_feature))
                cnt += 1
                yield ('正在提取第' + str(cnt) + '张正例图片的NPD特征...')
                # Global_Variables.Times.append(datetime.now())
        for root, dirs, files in os.walk(neg_dir, topdown=False):
            if neg_num == 0:
                neg_num = len(files)
            for file in files:
                im = Image.open(neg_dir + file).convert('L')
                im = im.resize((24, 24))
                im_array = np.array(im)
                img_feature = NPDFeature(im_array).extract()
                if not features_array_flag:
                    features_array = np.array(img_feature)
                    features_array_flag = True
                else:
                    features_array = np.row_stack((features_array, img_feature))
                cnt += 1
                yield ('正在提取第' + str(cnt) + '张负例图片的NPD特征...')
        yield (str(cnt) + '张照片的NPD特征提取完成，正在转换为npy格式储存...')
        # Global_Variables.Times.append(datetime.now())
        np.save('feature_data_array', features_array)
        yield ('(feature_data_array)成功存储在当前目录，格式为165600向量 * ' + str(cnt) + ' 的二维数组')
    except:
        yield ('提取特征失败！请确保所选择的文件夹里的数据都是图片格式！')
        yield ('Error')
    # Global_Variables.Times.append(datetime.now())

def save_images_2_array_test(dir):
    yield ('开始对图片进行预处理(转为为24*24灰度图)并提取图片的NPD特征...')
    # Global_Variables.Times.append(datetime.now())
    try:
        cnt = 0
        features_array_flag = False
        for root, dirs, files in os.walk(dir, topdown=False):
            for file in files:
                im = Image.open(dir + file).convert('L')
                im = im.resize((24, 24))
                im_array = np.array(im)
                img_feature = NPDFeature(im_array).extract()
                if not features_array_flag:
                    features_array = np.array(img_feature)
                    features_array_flag = True
                else:
                    features_array = np.row_stack((features_array, img_feature))
                cnt += 1
                yield ('正在提取第' + str(cnt) + '张测试图片的NPD特征...')
                # Global_Variables.Times.append(datetime.now())
        yield (str(cnt) + '张测试照片的NPD特征提取完成，正在转换为npy格式储存...')
        # Global_Variables.Times.append(datetime.now())
        np.save('feature_data_array', features_array)
        yield ('(feature_data_array)成功存储在当前目录，格式为165600向量 * ' + str(cnt) + ' 的二维数组')
    except:
        yield ('提取特征失败！请确保所选择的文件夹里的数据都是图片格式！')
        yield ('Error')
    # Global_Variables.Times.append(datetime.now())
def get_Train_Data_set():
    X = np.load('feature_data_array.npy')
    y = [1 for t in range(pos_num)]
    y.extend([-1 for t in range(neg_num)])
    y = np.array(y)
    return X, y

def get_Test_Data_set():
    X = np.load('feature_data_array.npy')
    return X

def train_main(depth, num, pos_dir, neg_dir):
    status = True
    g1 = save_images_2_array(pos_dir, neg_dir)
    while g1:
        try:
            result = g1.next()
            yield result
            if result == 'Error':
                status = False
        except:
            break
    if status:
        X, y = get_Train_Data_set()
        yield ('读取数据集成功！开始进行随机切分...')
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=214)
        yield ('成功切分为4:1的训练集和验证集...')
        # Global_Variables.Times.append(datetime.now())
        DTC = tree.DecisionTreeClassifier()
        yield ('成功初始化深度为' + str(depth)+'的决策树弱分类器模型 * '+str(num)+ '，进行迭代次数为' + str(num) + '的Adaboost训练...')
        # Global_Variables.Times.append(datetime.now())
        My_AdaBoost = Adaboost.AdaBoostClassifier(DTC)
        yield ('正在进行Adaboost模型训练...')
        # Global_Variables.Times.append(datetime.now())
        My_AdaBoost.fit(X_train, y_train)
        yield ('Adaboost训练模型结束！正在对训练集进行分类预测效果评估...')
        # Global_Variables.Times.append(datetime.now())
        # 获取预测结果
        yield ('Adaboost模型保存路径默认为当前程序所在目录...')
        # Global_Variables.Times.append(datetime.now())
        My_AdaBoost.save(My_AdaBoost, 'train_model')
        yield ('Adaboost预测模型成功保存至./train_model！')
        # Global_Variables.Times.append(datetime.now())
        yield ('Finish')
        # 先预测一波
        My_AdaBoost.predict(X_val)
        yield My_AdaBoost.is_good_enough(y_val)
        My_AdaBoost.predict(X)
        yield My_AdaBoost.is_good_enough(y)
    else:
        return

def test_predict(model_dir, test_dir):
    # 下面为读取已训练好的模型并进行预测
    status = True
    DTC = tree.DecisionTreeClassifier()
    My_AdaBoost = Adaboost.AdaBoostClassifier(DTC)
    try:
        My_model = My_AdaBoost.load(model_dir)
    except:
        yield ('模型读取失败，请确认是否正确输入模型路径或者检查是否被占用...')
        yield ('Error')
        return
    yield ('读取训练模型成功！ 开始进行测试集的特征提取...')
    g1 = save_images_2_array_test(test_dir)
    while g1:
        try:
            result = g1.next()
            yield result
            if result == 'Error':
                status = False
        except:
            break
    if status:
        X_test = get_Test_Data_set()
        yield ('特征提取成功！ 开始对测试集样本进行预测...')
        pre_result = My_model.predict(X_test)
        yield ('ok')
        yield pre_result
    else:
        return

# if __name__ == '__main__':
#     Train_visualization.UI()
#     train_main()
