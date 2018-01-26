# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import copy
class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, C=1.2500000000000000e+001,
                      gamma=3.3750000000000002e-002, term_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                                    100000, 1e-1))
        self.model.train(samples, responses, params=params)

    def train_auto(self, samples, responses, varldx, sampleldx):
        params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC,
                      k_fold=10)
        self.model.train_auto(samples, responses, varldx, sampleldx, params=params)

    def predict(self, samples):
        return np.float32([self.model.predict(samples)])

def projectedhistogram(img_in, string):
    sz = 0
    if string == "Horizontal":
        sz = img_in.shape[0]

    else:
        sz = img_in.shape[1]
    nonezeorimg = []
    img_in = cv2.extractChannel(img_in, 0)
    for j in range(sz):
        data = getrow(img_in, j) if (string == "Horizontal") else getcol(img_in, j)
        count = cv2.countNonZero(np.array(data))
        nonezeorimg.append(count)
    maxnum = 0.0
    for j in range(len(nonezeorimg)):
        maxnum = max(maxnum, nonezeorimg[j])
    if maxnum > 0:
        for j in range(len(nonezeorimg)):
            nonezeorimg[j] = nonezeorimg[j] / float(maxnum)
    return nonezeorimg

def getrow(img_in, j):
    return img_in[j]
def getcol(img_in, j):
    col = []
    for i in range(img_in.shape[0]):
        col.append(img_in[i][j])
    return col

def getfeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img_in = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    vhist = projectedhistogram(img_in, "Vertical")
    hhist = projectedhistogram(img_in, "Horizontal")
    numcols = len(vhist) + len(hhist)
    out = np.zeros((1, numcols), np.float32)
    j = 0
    for i in range(len(vhist)):
        out[0][j] = vhist[i]
        j = j + 1
    for i in range(len(hhist)):
        out[0][j] = hhist[i]
        j = j + 1
    return out


#
# samples = np.array(np.random.random((4, 2)), dtype=np.float32)
# y_train = np.array([1., 0., 0., 1.], dtype=np.float32)
#
# clf = SVM()
# clf.load("svm.xml")
# y_val = clf.predict(samples)
# print y_val