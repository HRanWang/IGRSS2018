from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Input,Conv2D,UpSampling2D,Permute
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K#this is to against the flatten error
#from IGRSSdata import create_train_data,create_test_data,load_train_data,load_test_data
from keras.models import model_from_json
import sys
import os
import numpy as np
from libtiff import TIFF
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import time
import math
import random
from sklearn.metrics import confusion_matrix

import scipy.misc
import matplotlib.pyplot as plt
import argparse
#nb_classes = 10
#batch_size = 17#von 15 bis zu 32
start = time.clock()

batch_size = 64
nb_epoch =1

img_rows,img_cols = 28,28
img_channels = 48

def samele_wise_normalization(data):
    return 255.0 * (data - np.min(data)) / (np.max(data) - np.min(data))

def channal_normalization():
    for i in range (48):
        train[i,:,:] = samele_wise_normalization(train[i,:,:])
    return train

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))

def sample_wise_standardization(data):
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0   # overall number of pixels in 3 dimensions
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)   # why use this max function?

def compute_Kappa(confusion_matrix):
    """
    TODO =_=
    """
    N = np.sum(confusion_matrix) # ALL points
    N_observed = np.trace(confusion_matrix)
    Po = 1.0 * N_observed / N
    h_sum = np.sum(confusion_matrix, axis=0)
    v_sum = np.sum(confusion_matrix, axis=1)
    Pe = np.sum(np.multiply(1.0 * h_sum / N, 1.0 * v_sum / N))
    kappa = (Po - Pe) / (1.0 - Pe)
    return kappa

def randdom(x,y):
    X = []
    Y = []
    index = [i for i in range(len(y))]
    random.shuffle(index)
    for i in index:
        print i
        X.append(x[i])
        Y.append(y[i])
    X = np.array(X,dtype='float32')
    Y = np.array(Y,dtype='float32')
    print X.shape,Y.shape
    return X,Y

def readimg(x,p):
    imgs_train_img = np.load('dataset50/train/img/imgs_train{0}.npy'.format(x))
    print imgs_train_img.shape
    s = int(p * imgs_train_img.shape[0])
    print s
    imgs_train_img = imgs_train_img[0:s, :, :]
    print imgs_train_img.shape
    # imgs_train_img/=imgs_train_img.max()-imgs_train_img.min()
    print 'imgs_train{0} has been loaded'.format(x)
    return imgs_train_img

def readlabel(x,p):
    imgs_train_img = np.load('dataset50/train/label/train_label{0}.npy'.format(x))
    print imgs_train_img.shape
    s = int(p * imgs_train_img.shape[0])
    print s
    imgs_train_img = imgs_train_img[0:s]
    print imgs_train_img.shape
    print 'train_label{0} has been loaded'.format(x)
    return imgs_train_img

def testimg(x,p):
    imgs_test_img = np.load('dataset50/test/img/imgs_test{0}.npy'.format(x))
    print imgs_test_img.shape
    s = int(p * imgs_test_img.shape[0])
    print s
    imgs_test_img = imgs_test_img[0:s, :, :]
    print imgs_test_img.shape
    # imgs_test_img/=imgs_test_img.max()-imgs_test_img.min()
    print 'imgs_test{0} has been loaded'.format(x)
    return imgs_test_img

def testlabel(x,p):
    imgs_test_img = np.load('dataset50/test/label/test_label{0}.npy'.format(x))
    print imgs_test_img.shape
    s = int(p * imgs_test_img.shape[0])
    print s
    imgs_test_img = imgs_test_img[0:s]
    print imgs_test_img.shape
    print 'test_label{0} has been loaded'.format(x)
    return imgs_test_img

def make_network():
    model = Sequential()
  #  model.add(Permute((3,1,2), input_shape=(img_rows, img_cols, img_channels)))
    model.add(Convolution2D(32, (1, 1), activation='relu',input_shape=(img_channels,img_rows, img_cols)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    return model

def train_model(model,X_train,Y_train,X_test,Y_test):
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])#do not forget parameter metrics=['accuracy'], or in the end of evaluation can't output accuracy
#    for x_train,y_train in zip(X_train,Y_train):
#        x_train = np.reshape(x_train,(1,10,10,3))
#        y_train = np.reshape(y_train,(100,21))
    model.fit(X_train, Y_train, shuffle=True,epochs=nb_epoch,  batch_size=batch_size)
    print('Testing...')
    acc = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
#    classes = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
#    print(classes)
    print('Test loss and accuracy: ')
    print(acc)


if __name__ == "__main__":
    K.set_image_dim_ordering('tf')#this is to against the flatten error

    print 'start'
    tiff_image_name = '2018_IEEE_GRSS_DFC_HSI_TR.tif'
    imagetif = TIFF.open(tiff_image_name, mode="r")
    img = imagetif.read_image()
    train = img[0:48,:,:]
    print train.shape

    up_img = train[:,0:14,:]
    down_img = train[:,587:601,:]
    print up_img.shape
    print down_img.shape

    up = FZ(up_img)
    down = FZ(down_img)

    train = np.concatenate((up_img, train), axis=1)
    train = np.concatenate((train,down),axis=1)
    print train.shape

    left_img = train[:,:,0:14]
    right_img = train[:,:,2370:2384]
    print left_img.shape
    print right_img.shape

    left = (left_img)
    right = (right_img)
    print left.shape
    print right.shape

    train = np.concatenate((left,train),axis=2)
    train = np.concatenate((train,right),axis=2)
    print train.shape,train.shape[1],train.shape[2]

    train = channal_normalization()

    # ------------------------------------------------------------------------#
    tiff_image_name = 'gt.tif'
    tif = TIFF.open(tiff_image_name, mode="r")
    inputMap = tif.read_image()
    print inputMap.shape

    # ------------------------------------------------------------------------#

    windows = 14
    ro = []
    co = []
    imgdatas = []
    row = 14
    while ((row - windows > -1) & (row + windows < train.shape[1])):
        col = 14
        while ((col - windows > -1) & (col + windows < train.shape[2])):
            x_t_crop = train[:, row - windows:row + windows, col - windows:col + windows]
            if (inputMap[row-14][col-14]>0):
                imgdatas.append(x_t_crop)
            col+=1
        row+=1
    print len(imgdatas)#20_classes：521761   0_class:911023    all_calsses:1432784
    imgdatas = np.array(imgdatas)

    print 'model loading'
    model = make_network()
    model.load_weights('./model_weights/weight-101-1.00.hdf5')

    preds = []
    for i in range(len(imgdatas)):
        # front = 1432784/16*i
        # back = (1432784/16)*(i+1)
        # print front,back
        # X_test = imgdatas[front:back]
        # X_test = sample_wise_standardization(X_test)
        X_test = np.resize(imgdatas[i],(1,48,28,28))
        X_test = np.array(X_test)
        X_test = X_test.astype('float32')
        print X_test.shape

        pred = model.predict(X_test,batch_size =1, verbose=1)
        pred = np.argmax(pred,axis=1)
        preds.append(pred)
    print len(preds)
    preds = np.asarray(preds,dtype=np.int8)+1
    np.savetxt('result/preds.txt', preds)

    print('Model Test Process has alread finished.')
    end = time.clock()
    t = end - start
    print 'all time:{0}'.format(t)
        #
        # X_test1 = imgdatas[89549:179018]
        # X_test1 = sample_wise_standardization(X_test1)
        # X_test1 = X_test1.astype('float32')
        # print X_test1.shape

        # print pred
    # C = confusion_matrix(Y_test,pred)
    # kappa = compute_Kappa(C)
    # k = np.zeros((2,2))
    # k[:] = kappa
    # print k.shape,k.dtype,k

    # np.savetxt('result/kapa_20_class_100_epoch.txt',k)
    # np.savetxt('result/Y_test_20_class_100_epoch.txt',Y_test)
    # np.savetxt('result/pred_20_class_100_epoch.txt',pred)
    # np.savetxt('result/confusion_matrix_20_class_100_epoch.txt',C)
    # print C

    #train_model(model,X_train,Y_train,X_test,Y_test)
    #model.save('./save/model_2class.h5')
    #model_json = model.to_json()
    #open('cifar10_architecture_fy.json', 'w').write(model_json)
    #model.save_weights('cifar10_weights_fy.h5')
