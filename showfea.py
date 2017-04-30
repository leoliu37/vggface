#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:33:08 2017

@author: leo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

Created on Mon Apr 24 10:55:40 2017

@author: leo
"""


import numpy as np
import sys,os,caffe
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

caffe_root = '/home/leo/caffe'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import sklearn.metrics.pairwise as pw


# from caffe offical tutorial
 #安装Python环境、numpy、matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math
#matplotlib inline

#设置默认显示参数
plt.rcParams['figure.figsize'] = (10, 10)        # 图像显示大小
plt.rcParams['image.interpolation'] = 'nearest'  # 最近邻差值: 像素为正方形
plt.rcParams['image.cmap'] = 'gray'  # 使用灰度输出而不是彩色输出
    

#我把GPU加速注释掉了,所以没有GPU加速,速度有点慢,你要在学校有条件找个有GeForce显卡的电脑
#caffe.set_mode_gpu()

root='/home/leo/caffe/examples/exercise3/'   #根目录
deploy=root + 'deploy.prototxt'    #deploy文件
caffe_model=root + 'vggface1_snapshot_iter_200.caffemodel'   #训练好的 caffemodel
img1=root+'SGFS/test/151.bmp'    #随机找的一张待测图片
img2=root+'SGFS/train/159.jpg'    #随机找的一张待测图片


#加载caffe模型
global net
net=caffe.Classifier(deploy,caffe_model)


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
    feature2=np.reshape(feature2,(test_num,4096))
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

    # visual feature
def vis_square(data):
        """输入一个形如：(n, height, width) or (n, height, width, 3)的数组，并对每一个形如(height,width)的特征进行可视化sqrt(n) by sqrt(n)"""



    # 对输入的图像进行normlization
        data = (data - data.min()) / (data.max() - data.min())

    # 强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
        n = int(np.ceil(np.sqrt(data.shape[0])))
    # 每幅小图像之间加入小空隙
        padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
        data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)   

        # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        # data的一个小例子,e.g., (3,120,120)
        # 即，这里的data是一个2d 或者 3d 的ndarray
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        # 显示data所对应的图像
        plt.imshow((data))
        plt.axis('off')
        plt.show()    
#        img = cv2.imread(imgpath)
        
#        cv2.imshow("Image", data)
#        cv2.waitKey (0)  
#        cv2.destroyAllWindows()  
    
def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )
    
#　编写一个函数，用于显示各层数据
def show_data(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
#    imgplot = plt.imshow(data)
#    imgplot.set_cmap('nipy_spectral')
    plt.imshow(data,cmap='gray')
    plt.axis('off')
    plt.show()
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_data1(data):

  datanum = data.shape[0]
  for num in range(0,datanum-1):
    data -= data.min()
    data /= data.max()
    data1 = data[num,]

    plt.subplot(math.ceil(math.sqrt(datanum)),math.ceil(math.sqrt(datanum)),num+1)
    imgplot = plt.imshow(data1)
    imgplot.set_cmap('nipy_spectral') 
#    plt.colorbar()
    plt.axis('off')
    
  plt.show()
  plt.figure() 
    
    
if __name__ == '__main__':
    
#利用提前训练好的模型，设置测试网络
    net = caffe.Net(deploy,caffe_model,caffe.TEST)
    
    net.blobs['data'].data.shape  
    #加载测试图片，并显示
    im = caffe.io.load_image(img1)

#    im=np.array(Image.open(img1).convert('RGB'))
#    print im.shape


#　编写一个函数，将二进制的均值转换为python的均值
    binMean = root+'mean.binaryproto'
    npyMean = root+'mean.npy'
    convert_mean(binMean,npyMean)    
    
#将图片载入blob中,并减去均值
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(npyMean).mean(1).mean(1)) # 减去均值
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))
    net.blobs['data'].data[...] = transformer.preprocess('data',im)
    inputData=net.blobs['data'].data
#显示减去均值前后的数据
    plt.figure()
    plt.subplot(1,2,1),plt.title("origin")
    plt.imshow(im)
    plt.axis('off')
    plt.subplot(1,2,2),plt.title("subtract mean")
    plt.imshow(transformer.deprocess('data', inputData[0]))
    plt.axis('off')
    plt.show()


#运行测试模型，并显示各层数据信息
    net.forward()
    [(k, v.data.shape) for k, v in net.blobs.items()]


#显示各层的参数信息
    [(k, v[0].data.shape) for k, v in net.params.items()]

#显示第一个卷积层的输出数据和权值（filter）
    print net.blobs['pool5'].data.shape
    show_data1(net.blobs['pool5'].data[0])
    show_data(net.blobs['pool5'].data[0])
    print net.params['conv3_3'][0].data.shape
    show_data(net.params['conv3_3'][0].data[:64],)
    


#==============================================================================
# #显示第一次pooling后的输出数据
#     show_data(net.blobs['pool1'].data[0])
#     net.blobs['pool1'].data.shape
# #显示第二次卷积后的输出数据以及相应的权值（filter）
#     show_data(net.blobs['conv2_1'].data[0],padval=0.5)
#     print net.blobs['conv2_1'].data.shape
#     show_data(net.params['conv2_1'][0].data)
#     print net.params['conv2_1'][0].data.shape
# #显示第三次卷积后的输出数据以及相应的权值（filter）,取前１024个进行显示
#     show_data(net.blobs['conv3_1'].data[0],padval=0.5)
#     print net.blobs['conv3_1'].data.shape
#     show_data(net.params['conv3_1'][0].data)
#     print net.params['conv3_1'][0].data.shape
# 
# 
#     #显示第三次池化后的输出数据
#     show_data(net.blobs['pool3'].data[0],padval=0.2)
#     print net.blobs['pool3'].data.shape
#     # 最后一层输入属于某个类的概率
#     feat = net.blobs['prob'].data[0]
#     print feat
#     plt.plot(feat.flat)
#    
#     show_data(net.blobs['conv4_1'].data[0],padval=0.5)
#     print net.blobs['conv4_1'].data.shape
#     show_data(net.blobs['pool4'].data[0],padval=0.2)
#     print net.blobs['pool4'].data.shape
#     show_data(net.blobs['conv5_1'].data[0],padval=0.5)
#     print net.blobs['conv5_1'].data.shape
#     show_data(net.blobs['pool5'].data[0],padval=0.2)
#     print net.blobs['pool5'].data.shape
# 
# 
#==============================================================================

#==============================================================================
# for layer_name, blob in net.blobs.iteritems():
#     print layer_name + '\t' + str(blob.data.shape)
#     
# for layer_name, param in net.params.iteritems():
#     print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
# 
#     filters = net.params['conv1_1'][0].data
#     vis_square(filters.transpose(0, 2, 3, 1))
# 
#     feat = net.blobs['conv1_1'].data[0,]
#     cv2.imshow(feat)
#==============================================================================
