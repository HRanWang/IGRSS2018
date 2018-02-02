# -*- coding: utf-8 -*-
import numpy as np
import cv2
from libtiff import TIFF
import random
from sklearn.metrics import confusion_matrix
import collections
from compiler.ast import flatten

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

height, width = 601, 2384
num_class = 20

rgb_value = [[0, 0, 0], [124, 252, 0], [107, 142, 35], [34, 139, 34], [0, 255, 0],               # 0~4
            [48, 128, 20], [115, 74, 18], [0, 0, 255], [188, 143, 143], [214, 189, 217],         # 5~9
            [160, 32, 240], [189, 252, 201], [240, 230, 140], [21, 6, 162], [255, 69, 0],     # 10~14
            [255, 192, 203], [255, 255, 255], [0, 199, 140], [169, 2, 178], [61, 89, 171], [0, 255, 127]] # 15~20

classes = ['健康草','被压草','人造草坪','常绿乔木','落叶乔木','裸土','水','住宅楼宇','非住宅楼宇','道路','人行道','人行横道',
           '主要街道','公路','铁路','铺好的停车场','铺砌的停车场','车','列车','体育场席位']
#0---------------------------find road from building---------------------------------------#
DSM_image_name = 'DSM.tif'
dsm = TIFF.open(DSM_image_name, mode="r")
DSM = dsm.read_image()
print DSM.shape
in_row, in_col = np.shape(DSM)
poolStride =poolSize = 2
out_row, out_col = int(np.floor(in_row / poolStride)), int(np.floor(in_col / poolStride))
outputMap = np.zeros((out_row, out_col))

for r_idx in range(0, out_row):
    for c_idx in range(0, out_col):
        poolField = DSM[r_idx*2:r_idx*2+poolStride, c_idx*2:c_idx*2 + poolSize]
        xx = flatten(poolField.tolist())
        dic=collections.Counter(xx)
        value = max(dic.values())
        for k, v in dic.items():
            if v == value:
                poolOut = k
        outputMap[r_idx, c_idx] = poolOut
print outputMap.shape
d_s_m = np.resize(outputMap,(1432784))


#1----------------- load predict label from saved txt---------------#
pred = np.loadtxt("./rgb/preds_BN.txt")
pred = pred.astype("uint8")
print pred.shape
#1----------------- load predict label from saved txt---------------#

#2----------------- load ground true label from gt.tif and resize gt to flatten array---------------#
tiff_image_name = 'gt.tif'
tif = TIFF.open(tiff_image_name, mode="r")
inputMap = tif.read_image()
print inputMap.shape
gt = np.resize(inputMap,(1432784))
print gt.shape
#2----------------- load ground true label from gt.tif and resize gt to flatten array---------------#

#3----------------- n is number of zero:911023,m is number of 20 classes:521761---------------#
n = 0
m = 0
road = 0
solid = 0
street = 0
sport = 0
preds = np.zeros((1432784),dtype='uint8')
for y in range(1432784):
    if gt[y] == 0:
        preds[y] = 0
        n+=1
    else:
        preds[y] = pred[m]
        m+=1
print preds.shape,n,m

# for z in range(1432784):
#     if preds[z] == 9:
#         if d_s_m[z] <-14:
#             if d_s_m[z] < -16:
#                 if d_s_m[z] <-17:
#                     preds[z] = 6
#                     solid+=1
#                 else:
#                     preds[z] = 10
#                     road += 1
#             else:
#                 preds[z] = 13
#                 street+=1
# print 'road from building:',road
#3----------------- n is number of zero:911023,m is number of 20 classes:521761---------------#

#----------------- AA(average accuracy)---------------#
num_true = np.zeros((21))
num_pred = np.zeros((21))
every_preds = []
for w in range(1432784):
    if gt[w] != 0:
        num_true[gt[w]]+=1
        if preds[w] == gt[w]:
            num_pred[preds[w]]+=1
# print num_pred[0],num_pred[1],num_pred[2],num_pred[3],num_pred[4]
# print num_true[0],num_true[1],num_true[2],num_true[3],num_true[4]
print num_pred.shape
mean_all = 0
for m in range(20):
    mean_single = (num_pred[m+1])/(num_true[m+1])
    mean_all =mean_all + mean_single
    every_preds.append(mean_single)
    print classes[m],':',mean_single
mean = mean_all/20
print 'AA:',mean
#----------------- AA(average accuracy)---------------#

#4--------------------------------OA (overall accuracy)----------------------------------#
acc = 0
for x in range(1432784):
    if (preds[x] == gt[x]):
        acc+=1
acc = acc - n
print 'number of all right classification:',acc
oa = 100.0*acc/(1432784-n)
OA = np.zeros((2, 2))
OA[:] = oa
print 'OA:',100.0*acc/(1432784-n)
#4--------------------------------OA (overall accuracy)----------------------------------#

#8--------------------------------true label without 0 class----------------------------------#
Y = np.zeros((521761))
P = np.zeros((521761))
l = 0
for y in range(1432784):
    if gt[y] != 0:
        Y[l] = gt[y]
        P[l] = preds[y]
        l+=1
print Y.shape
#8--------------------------------true label without 0 class----------------------------------#

#9-------------------------------- confusion matrix ,Kappa----------------------------------#

C = confusion_matrix(Y, P)     #  confusion matrix
kappa = compute_Kappa(C)          #  Kappa
k = np.zeros((2, 2))
k[:] = kappa
print 'kappa:',kappa
#9-------------------------------- confusion matrix ,Kappa----------------------------------#

#5--------------------------------predict class in 2D array----------------------------------#
preds = np.resize(preds,(601,2384))
#5--------------------------------predict class in 2D array----------------------------------#

#6-------------------------------- draw predict image----------------------------------#
img_preds = np.ndarray((height,width,3),dtype=int)
for i in range(601):
    for j in range(2383):
        img_preds[i][j] = rgb_value[preds[i][j]]
cv2.imwrite('rgb_nor.jpg',img_preds)
#6-------------------------------- draw predict image----------------------------------#

#7-------------------------------- draw ground true image----------------------------------#
img_gt = np.ndarray((height,width,3),dtype=int)
for i in range(601):
    for j in range(2383):
        img_gt[i][j] = rgb_value[inputMap[i][j]]
cv2.imwrite('gt_nor.jpg',img_gt)
#7-------------------------------- draw ground true image----------------------------------#

#10-------------------------------- save kappa,confusion matrix,overall accuracy----------------------------------#
# every_preds = np.array(every_preds,dtype=int)

np.savetxt('result100/kapa_20_class_100_epoch_nor.txt', k)
np.savetxt('result100/confusion_matrix_20_class_100_epoch_nor.txt', C)
np.savetxt('result100/Overall_accuracy_20_class_100_epoch_nor.txt', OA)
np.savetxt('result100/every_accuracy_20_class_100_epoch_nor.txt', every_preds)
#10-------------------------------- save kappa,confusion matrix,overall accuracy----------------------------------#
