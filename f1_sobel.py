# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import copy
from judge import *
def platelocate(img_path):
    resultRects = []
    img = cv2.imread(img_path)
    if debug < 3:
        cv2.namedWindow("img", 0)
        cv2.imshow("img", img)
        cv2.waitKey(0)
    # 高斯模糊：车牌识别中利用高斯模糊将图片平滑化，去除干扰的噪声对后续图像处理的影响
    gaussian = cv2.GaussianBlur(img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    if debug == 1:
        cv2.imshow("gaussian", gaussian)
        cv2.waitKey(0)
    # 灰度化
    gray = cv2.cvtColor(gaussian, cv2.COLOR_RGB2GRAY)
    if debug == 1:
        cv2.imshow("img", gray)
        cv2.waitKey(0)
    # equal = cv2.equalizeHist(gray)
    # if debug == 1:
    #     cv2.imshow("equal", equal)
    #     cv2.waitKey(0)
    # sobel算子：车牌定位的核心算法，水平方向上的边缘检测，检测出车牌区域
    sobelx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3))
    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3))
    sobely1 = cv2.convertScaleAbs(sobely)
    sobel = cv2.addWeighted(sobelx1, 0.9, sobely1, 0.1, 0)
    if debug == 1:
        cv2.imshow("img", sobel)
        cv2.waitKey(0)
    # 进一步对图像进行处理，强化目标区域，弱化背景。
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if debug == 1:
        cv2.imshow("img", binary)
        cv2.waitKey(0)
    # 进行开操作，去除细小噪点
    # eroded = cv2.erode(binary, None, iterations=1)
    # dilation = cv2.dilate(binary, None, iterations=1)
    # if debug == 1:
    #     cv2.imshow("dilation", dilation)
    #     cv2.waitKey(0)

    # 进行闭操作，闭操作可以将目标区域连成一个整体，便于后续轮廓的提取
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    if debug == 1 or debug == 2:
        cv2.imshow("img", closed)
        cv2.waitKey(0)

    #寻找轮廓
    rects = []
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = copy.copy(img)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 0)

    if debug == 1 or debug == 2:
        cv2.imshow("img", result)
        cv2.waitKey(0)
    for contour in contours:
        minRect = cv2.minAreaRect(contour)
        if verifysizes(minRect):
            rects.append(minRect)
    for rect in rects:
        if verifysizes(rect):
            # 根据矩形转成box类型，并int化
            width = rect[1][0]
            height = rect[1][1]
            dsize = (int(width), int(height))
            # 正常情况车牌长高比在2.7-5之间,那种两行的有可能小于2.5，这里不考虑
            ratio = float(width) / float(height)
            angle = rect[2]

            if ratio < 1:
                angle = angle + 90
                dsize = (int(height), int(width))
            if (angle < 30) and (angle > -30):
                rotmat = cv2.getRotationMatrix2D(rect[0], angle, 1)
                img_rotated = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0]))
                resultRect = showresultrect(img_rotated, dsize, rect[0])
                resultRects.append(resultRect)
                # cv2.imshow("test", resultRect)
                # cv2.waitKey(0)
    return resultRects

def verifysizes(rect):
    area = int(rect[1][0]*rect[1][1])
    try:
        ratio = max(rect[1])/min(rect[1])
    except:
        return False
    return (area >= 800) and (ratio >= 1.0) and (ratio <= 8.5)

def showresultrect(img_rotated, dsize, center):
    img_crop = cv2.getRectSubPix(img_rotated, dsize, center)
    img_test = cv2.resize(img_crop, (136, 36))

    return img_test

if __name__ == '__main__':
    debug = 1
    path = "e:/second_dataset/"
    error = 0
    fileList = os.listdir(path)
    for file in fileList:
        resultrects = platelocate(path + file)
        print path+file
        resultlist = platejudge(resultrects)
        for result in resultlist:
            # cv2.imwrite("e:/180122/"+file, result)
            # break
            cv2.imshow("123", result)
            cv2.waitKey(0)
    # resultrects = platelocate(path)
    # print len(resultrects)
    # resultlist = platejudge(resultrects)
    # print "result is", len(resultlist)
    # for result in resultlist:
    #     cv2.imshow("hehe", result)
    #     cv2.waitKey(0)













