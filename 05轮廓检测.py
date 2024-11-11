import cv2 as cv
import numpy as np

if __name__ == '__main__':

    img = cv.imread('test.png')
    # 边缘检测和轮廓检测区别：
    # 边缘检测线条是不连续的，只有有梯度，就把有梯度的像素点提取出来，这就是边缘检测
    # 轮廓检测时要一个完整的整体，就是将边缘连接起来成一个整体，才叫轮廓
    # 可以说边缘包括轮廓。
    # 边缘主要是作为图像的特征使用，比如可以用边缘特征区分脸和手，
    # 轮廓则是一个很好的图像目标的外部特征，这种特征对于我们进行图像分析，目标识别和理解等更深层次的处理。

    # 五、轮廓检测

    # 函数使用前提推荐，对原数据进行灰度图处理转为二值
    # mode轮廓检索模式，常用cv.RETR_TREE，检测所有的轮廓，并重构嵌套轮廓的整个层次，mode不同产生的轮廓也会不同
    # method轮廓逼近方法
    # 只可以处理二值图像，彩色图需要转为灰度图，再转为二值图
    binary, contours, hierarchy = cv.findContours(image=img.copy(), mode=cv.RETR_TREE,
                                                  method=cv.CHAIN_APPROX_NONE)
    # 画轮廓
    # contourIdx=-1默认输出全部轮廓，0或者正整数代表绘制的索引
    res = cv.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 0, 255))

    # 利用轮廓检测与逻辑与运算，mask，可以实现抠图
    # 彩色图 -> 灰度图 -> 二值图 -> 轮廓检测 -> 轮廓绘制（掩膜mask） -> 掩膜与原始图像与运算 -> 完成抠图

    # 轮廓特征
    # 计算第1个轮廓的面积
    cv.contourArea(contours[0])
    # 计算第2个轮廓的周长
    cv.arcLength(curve=contours[1], closed=True)

    # 轮廓近似
    # epsilon设置越大，轮廓变化程度越大
    epsilon = 0.1 * cv.arcLength(curve=contours[1], closed=True)
    cv.approxPolyDP(curve=contours[0], epsilon=epsilon, closed=True)

