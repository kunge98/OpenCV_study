import cv2 as cv
import numpy as np


if __name__ == '__main__':

    # 先将彩色图转为灰度图
    # 如果非要处理彩图，那么必须将彩图的通道分别处理
    # 直方图

    # 0表示灰度图读取
    img = cv.imread('test.png', 0)
    # 也可以这样书写
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 直方图：不同图像的直方图不均衡，像素点差异大
    # 横坐标：0-255；纵坐标：数量
    # 参数的形式需要使用列表，channels统计哪一个维度，mask指定统计的区域
    cv.calcHist(images=[img], channels=[0], mask=None, histSize=[256], ranges=[0,256])

    # 均衡化：一种分布变换到另一种分布
    # 如果一幅图像的灰度值都集中在一个小的范围内，那么这幅图看起来就是一个色调，图像画面也没有什么对比度
    # 由原先的“瘦高”变为“矮胖”
    # 但是有时候会将局部的纹理特征全部分散掉
    cv.equalizeHist(src=img)

    # 自适应直方图均衡化
    # 直方图均衡化是从整个图像的所有像素点进行均衡处理的。
    # 在很多情况下，这样做效果并不好，因为对比度改变了实际上我们会丢失很多信息，
    # 比如一些细小的边缘信息，实际上是会丢失的
    # 对图中不同区域使用不同的均衡化处理，局部细节会保留
    cv.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))

