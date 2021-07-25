# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import pandas as pd

theshold1=58
theshold2=72
theshold0=43

def getmax(img):
    max_of_grey_level=img[0][0]
    max_x=0
    max_y=0
    #classic method of find the max
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=max_of_grey_level:
                max_of_grey_level=img[i][j]
                max_x=j
                max_y=i
    return(max_of_grey_level,max_x,max_y,len(img),len(img[0]))

def getfeature1(img):
    sum=0
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=theshold2:
                sum+=1
    return(sum)

def getfeature2(img):
    sum=0
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=theshold1:
                sum+=1
    return(sum)

def getfeature3(img):#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=theshold2:
                tmp.append(img[i][j])
    if len(tmp) <= 1:
        return (0)
    else:
        return (np.var(tmp, ddof=1))  # 计算样本方差(/(N-1))

def getfeature4(img):
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=theshold2:
                tmp.append(img[i][j])
    if len(tmp)<= 1:
        return(0)
    else:
        return(np.mean(tmp))


def getfeature5(img):
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=theshold1:
                tmp.append(img[i][j])
    if len(tmp)<= 1:
        return(0)
    else:
        return(np.var(tmp,ddof = 1))  # 计算样本方差(/(N-1))


def getfeature6(img):
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=theshold1:
                tmp.append(img[i][j])
    if len(tmp)<= 1:
        return(0)
    else:
        return(np.mean(tmp))


def getfeature7(img):
    sum=0
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]<=theshold0:
                sum+=1
    return(sum)


def getfeature8(img):
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            tmp.append(img[i][j])
    return(np.mean(tmp))


def getfeature9(img):
    tmp=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            tmp.append(img[i][j])
    return(np.var(tmp, ddof = 1))

def getfeature10(img):
    tmp = []
    for i in range(len(img)):
        for j in range(len(img[1])):
            tmp.append(img[i][j])
    return (np.median(tmp))

def getfeature11(img):
    tmp = []
    for i in range(len(img)):
        for j in range(len(img[1])):
            tmp.append(img[i][j])
    middle=np.median(tmp)
    tmp1=[]
    max_all=getmax(img)
    max_value = max_all[0]
    height=(middle+max_value*2)/3
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=height:
                tmp1.append(img[i][j])
    return(np.mean(tmp1))

def getfeature12(img):
    tmp = []
    for i in range(len(img)):
        for j in range(len(img[1])):
            tmp.append(img[i][j])
    middle=np.median(tmp)
    tmp1=[]
    for i in range(len(img)):
        for j in range(len(img[1])):
            if img[i][j]>=middle:
                tmp1.append(img[i][j])
    return (np.var(tmp1, ddof = 1))




def getfeatures(img):
    sum=len(img)*len(img[0])
    result = []
    #0--最大亮度的灰度值
    tmp1=getmax(img)
    result.append(tmp1[0])

    #1--灰度值大于 theshold2 的像素点的数目
    result.append(getfeature1(img)/sum)

    #2--灰度值大于 theshold1 的像素点的数目
    result.append(getfeature2(img)/sum)

    #3--灰度值大于 theshold2 的像素点的灰度值的均方差
    result.append(getfeature3(img))

    #4--灰度值大于 theshold2 的像素点的灰度值的均值
    result.append(getfeature4(img))

    #5--灰度值大于 theshold1 的像素点的灰度值的均方差
    result.append(getfeature5(img))

    #6--灰度值大于 theshold1 的像素点的灰度值的均值
    result.append(getfeature6(img))

    #7--灰度值小于 theshold0 的像素点的数目
    result.append(getfeature7(img)/sum)

    #8--整体灰度值的均值
    result.append(getfeature8(img))

    #9--整体灰度值的均方差
    result.append(getfeature9(img))

    #10--中位数
    result.append(getfeature10(img))

    #11--峰值平均
    result.append(getfeature11(img))

    #12--峰值方差
    result.append(getfeature12(img))

    #13--max-middle
    result.append(tmp1[0]-getfeature10(img))

    return(result)

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


if __name__ == '__main__':
    list_of_csv=[]
    for filepath, dirnames, filenames in os.walk(r"D:/test/images-0712/positive"):
        for filename in filenames:
            tmp=os.path.join(filepath,filename)
            pre_img = cv2.imread(tmp,0)
            img=add_gaussian_noise(pre_img)
            tmp_result=getfeatures(img)
            tmp_result.append(1)
            list_of_csv.append(tmp_result)
    for filepath, dirnames, filenames in os.walk(r"D:/test/images-0712/negtive"):
        for filename in filenames:
            tmp=os.path.join(filepath,filename)
            pre_img = cv2.imread(tmp, 0)
            img = add_gaussian_noise(pre_img)
            tmp_result=getfeatures(img)
            tmp_result.append(0)
            list_of_csv.append(tmp_result)
    name=["max",">=2--sum",">=1--sum",">=2--var",">=2--mean",">=1--var",">=1--mean","<=0--sum","all--mean","all--var","all-middle","high-mean","above--var","max-middle","type"]
    csv = pd.DataFrame(columns=name, data=list_of_csv)
    csv.drop([len(csv) - 1], inplace=True)
    csv.to_csv("D:/test/images-0712/feature_matrix_wrong.csv",encoding='gbk')
