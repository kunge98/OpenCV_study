import numpy as np
import cv2 as cv

if __name__ == '__main__':

    img = cv.imread('test.png')

    # 图像分割算法

    # 分水岭算法
    # 根据灰度值判断山丘凹地，灰度大的为山丘，小的为凹地

    # GrabCut算法
