import numpy as np
import cv2 as cv

if __name__ == '__main__':

    # 从这个py文件开始，逐渐开始偏离正题，逐渐听不懂了

    img = cv.imread('test.png')
    img2 = cv.imread('test.png')

    # 帧差法---方法简单，但是会引入噪音和空洞问题（了解）
    # 算法对时间上连续的两帧图像进行差分运算，不同帧对应的像素点相减，判断灰度差的绝对值，
    # 当绝对值超过一定阈值时，可判定为运动目标，从而实现目标的检测功能

    # 混合高斯模型---（听天书）目标检测

