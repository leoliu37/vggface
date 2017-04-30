#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 22:08:33 2017

@author: leo
"""

import numpy as np
import os
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')

import cv2.cv as cv
from skimage import transform as tf

from PIL import Image, ImageDraw
import threading
from time import ctime,sleep
import time
import sklearn
import matplotlib.pyplot as plt
import skimage

import caffe
import sklearn.metrics.pairwise as pw


# from caffe offical tutorial
 #安装Python环境、numpy、matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy.io as sio
#matplotlib inline

    #设置默认显示参数
plt.rcParams['figure.figsize'] = (10, 10)        # 图像显示大小
plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值: 像素为正方形
plt.rcParams['image.cmap'] = 'gray'  # 使用灰度输出而不是彩色输出
    

#我把GPU加速注释掉了,所以没有GPU加速,速度有点慢,你要在学校有条件找个有GeForce显卡的电脑
#caffe.set_mode_gpu()
caffe_root = '/home/leo/caffe'
sys.path.insert(0, caffe_root + 'python')

root='/home/leo/caffe/examples/exercise3/'   #根目录
deploy=root + 'deploy.prototxt'    #deploy文件
caffe_model=root + 'vggface1_snapshot_iter_200.caffemodel'   #训练好的 caffemodel
img1=root+'SGFS/test/151.bmp'    #随机找的一张待测图片
img2=root+'SGFS/train/159.jpg'    #随机找的一张待测图片

testimgpath = root+'SGFS/test'
Photolistloc = root+'photolocation.txt'
Sketchlistloc = root+'sketchlocation.txt'

#加载caffe模型
global net
net=caffe.Classifier(deploy,caffe_model)


def extra_fea(path):
    
    global net
    #加载验证图片
    X=read_image(path)
    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    # out = net.forward_all(data = X)
    out = net.forward_all(blobs=['fc7'],data = X)
    fc7=out['fc7']
    
    #fc7是模型的输出,也就是特征值
    feature1 = np.float64(out['fc7'])
    feature1=np.reshape(feature1,(test_num,4096))
#    #加载注册图片
#    X=read_image(path2)
#    #X  作为 模型的输入
#    # out = net.forward_all(data=X)
#    out = net.forward_all(blobs=['fc7'],data = X)
#    fc7=out['fc7']
#    #fc7是模型的输出,也就是特征值
#    feature2 = np.float64(out['fc7'])
#    feature2=np.reshape(feature2,(test_num,4096))
#    #求两个特征向量的cos值,并作为是否相似的依据
#    predicts=pw.cosine_similarity(feature1, feature2)
    return  feature1


def compar_pic(path1,path2):
    global net
    #加载验证图片
    X=read_image(path1)
    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    # out = net.forward_all(data = X)
    out = net.forward_all(blobs=['fc7'],data = X)
    fc7=out['fc7']
    
    #fc7是模型的输出,也就是特征值
    feature1 = np.float64(out['fc7'])
    feature1=np.reshape(feature1,(test_num,4096))
    #加载注册图片
    X=read_image(path2)
    #X  作为 模型的输入
    # out = net.forward_all(data=X)
    out = net.forward_all(blobs=['fc7'],data = X)
    fc7=out['fc7']
    #fc7是模型的输出,也就是特征值
    feature2 = np.float64(out['fc7'])
    feature2 = np.reshape(feature2,(test_num,4096))
    #求两个特征向量的cos值,并作为是否相似的依据
    predicts=pw.cosine_similarity(feature1, feature2)
    return  predicts



def read_image(filelist):

    averageImg = [129.1863,104.7624,93.5940]
    X=np.empty((1,3,224,224))
    word=filelist.split('\n')
    filename=word[0]
    im1=skimage.io.imread(filename,as_grey=False)
    #归一化
    image =skimage.transform.resize(im1,(224, 224))*255
    X[0,0,:,:]=image[:,:,0]-averageImg[0]
    X[0,1,:,:]=image[:,:,1]-averageImg[1]
    X[0,2,:,:]=image[:,:,2]-averageImg[2]
    return X

    
def read_imagelist(filelist):
        '''
        @brief：从列表文件中，读取图像数据到矩阵文件中
        @param： filelist 图像列表文件
        @return ：4D 的矩阵
        '''
        fid=open(filelist)
        lines=fid.readlines()
        test_num=len(lines)
        fid.close()
        X=np.empty((test_num,3,224,224))
        i =0
        for line in lines:
            word=line.split('\n')
            filename=word[0]
            im1=skimage.io.imread(filename,as_grey=False)
            image =skimage.transform.resize(im1,(224, 224))*255
            if image.ndim<3:
                print 'gray:'+filename
                X[i,0,:,:]=image[:,:]
                X[i,1,:,:]=image[:,:]
                X[i,2,:,:]=image[:,:]
            else:
                X[i,0,:,:]=image[:,:,0]
                X[i,1,:,:]=image[:,:,1]
                X[i,2,:,:]=image[:,:,2]
            i=i+1
        return X
    
    
def extra_vggfea(imglist):
    '''from imglist each row represents each img, 
    obtain the fealist which each row represents each fea '''
    averageImg = [129.1863,104.7624,93.5940] #from calculation by date
 

    X = imglist

    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    # out = net.forward_all(data = X)
    out = net.forward_all(blobs=['fc7'],data = X)
    fc7=out['fc7']
    
    #fc7是模型的输出,也就是特征值
    feature = np.float64(out['fc7'])
    feature = np.reshape(feature,(test_num,4096))    
    
    return feature
    
    


        
if __name__ == '__main__':
    
    
    
#    #设置阈值,大于阈值是同一个人,反之
#    thershold = 0.85
#    #加载注册图片与验证图片
#    #注意:人脸图像必须是N*N的!!!如果图片的高和宽不一样,进行归一化的时候会对图片进行拉伸,影响识别效果
#    reg_path = img1
#    rec_path = img2
    
#    fea1 = extra_fea(reg_path)
#    fea2 = extra_fea(rec_path)
#    fea = np.concatenate((fea1,fea2))
    
    Photolist = read_imagelist(Photolistloc)
    Sketchlist = read_imagelist(Sketchlistloc)
    Photofea = extra_vggfea(Photolist)
    Sketchfea = extra_vggfea(Sketchlist)
    
    save_file = root+'vggfea.mat'
    sio.savemat(save_file, {'photofea': Photofea, 'sketchfea': Sketchfea}) #同理，只是存入了两个不同的变量供使用
    
    Distance = pw.cosine_similarity(Sketchfea, Photofea)
    
    
    print Distance[0,]
    
#    #计算注册图片与验证图片的相似度
#    result=compar_pic(reg_path,rec_path)
#    print "%s和%s两张图片的相似度是:%f\n\n"%(reg_path,rec_path,result)
#    if result>=thershold:
#        print '是一个人!!!!\n\n'
#    else:
#        print '不是同一个人!!!!\n\n'