import cv2 as cv
import numpy as np


if __name__ == '__main__':

    img = cv.imread('test.png')

    # 1.图像阈值
    # type=cv.THRESH_XXXX  主要是依靠这个变量来对图像进行不同的操作
    cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_BINARY)
    # 反二值化阈值处理：大于127->0; 小于127->255
    cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_BINARY_INV)
    # 截断二值化阈值处理： 大于127->127；小于等于127维持不变
    cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_TRUNC)
    # 超阈值零处理：大于127的设置为0，小于等于127的不变
    cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_TOZERO_INV)
    # 低阈值零处理：大于127的不变，小于等于127的设置为0
    cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_TOZERO_INV)

    # 自适应预支分割
    cv.adaptiveThreshold(src=img, adaptiveMethod=cv.BORDER_REPLICATE, thresholdType=cv.THRESH_BINARY, blockSize=3,C=2)

    # Otsu阈值
    # 在type中使用“+”将两种方法相加，cv.THRESH_OTSU可实现自动阈值
    # 该方法只能处理二维图像，所以可以将bgr转换成gray，处理完成后再gray转换成rgb
    cv.threshold(src=img, thresh=127, maxval=255, type=cv.THRESH_BINARY+cv.THRESH_OTSU)

    # 2.图像平滑
    # 均值滤波,相当与简单平均的卷积操作
    # 用一个只有尺寸没有数值的卷积核去卷积原图像，卷积核越大，参与到运算的像素点越多
    # 更接近均值，去噪效果越好，但是图像会变的更模糊
    cv.blur(src=img, ksize=(3, 3))

    # 方框滤波，可以选择归一化，如果设置为False发生越界会当做255处理
    # 用一个有尺寸有数值的卷积核去卷积
    cv.boxFilter(src=img, ksize=(3, 3), normalize=True)

    # 高斯滤波
    # 卷积核尺寸必须是奇数，卷积核里的数值要经过归一化处理
    # 用一个有尺寸有数值的卷积核去卷积，但是卷积核的数值权重大小不一样，距离越近权重越高
    cv.GaussianBlur(src=img, ksize=(5, 5), sigmaX=1, sigmaY=1)

    # 中值滤波,数值进行排序，选择中间值，这个api效果较好
    # 用一个只有尺寸没有数值的卷积核去卷积原图像，
    # 被卷积核套住的区域中的中间值作为当前像素点的像素值
    cv.medianBlur(src=img, ksize=(3, 3))

    # 双边滤波
    # 同时考虑了距离和像素的色彩信息
    # 距离越远权重越小，色彩差别越大权重越小
    # d取正整数，sigmaSpace毫无意义，两个差不多都表示空间核大小
    # sigmaColor表示卷积核套住的区域中像素差要大于255就不计算了，
    # 显然设置255毫无意义
    cv.bilateralFilter(src=img, d=3, sigmaColor=255, sigmaSpace=(3, 3))


    # 3.形态学：图像形态学就指以形态为基础对图像进行分析的一种方法或技术
    # 图像形态学操作主要是对二值图像进行操作的，来连接相邻的元素或分离成独立的元素。
    # 其次是灰度图像，但处理彩色图像几乎没有意义
    # 应用领域：消除噪声、提取边界、填充区域、提取连通分量、凸壳、细化、粗化
    # 割出独立的图像元素，或者图像中相邻的元素；求取图像中明显的极大值区域和极小值区域；求取图像梯度
    # 视觉检测、图像理解、文字识别、医学图像处理、图像压缩编码等领域有非常重要的应用
    # 图像形态学操作主要有：
    # 膨胀、腐蚀、开运算、闭运算、梯度运算、礼帽运算、黑帽运算，击中与击不中变换
    # 其中膨胀和腐蚀是基本操作，后面的操作都是在膨胀和腐蚀的基础上延伸而来的。

    # 腐蚀操作（白色变少）
    kernel = np.ones((5, 5), np.uint8)
    # 腐蚀次数，核越大腐蚀越快
    cv.erode(src=img, kernel=kernel, iterations=1)
    # 膨胀操作（白色变多）,反操作腐蚀操作
    cv.dilate(src=img, kernel=kernel, iterations=1)

    # 开运算,先腐蚀，后膨胀
    # 开运算可以去噪、计数等。
    # 比如识别一张图像中有几个人，要先把人和人重叠的部分分开，然后再计数
    cv.morphologyEx(src=img, op=cv.MORPH_OPEN, kernel=kernel)

    # 闭运算,先膨胀，后腐蚀
    # 闭运算可以去除前景物内部的黑点，还可以将不同前景图像进行连接，就是实现前景图像的连接。
    cv.morphologyEx(src=img, op=cv.MORPH_CLOSE, kernel=kernel)

    # 梯度运算
    # 运算方式： 膨胀的图像（白色占多数，扩张亮度） - 腐蚀的图像（白色占少数，收缩亮度）
    # 就可以获取原始图像中的前景图像的边缘
    cv.morphologyEx(src=img, op=cv.MORPH_GRADIENT, kernel=kernel)

    # 礼帽(顶帽) ： 原始输入 - 开运算（先腐蚀，后膨胀）
    # 获得图像的噪声信息，或者得到比原始图的边缘更亮的边缘信息
    cv.morphologyEx(src=img, op=cv.MORPH_TOPHAT, kernel=kernel)

    # 黑帽 ： 闭运算（先膨胀，后腐蚀） - 原始输入
    # 以获得图像内部的噪音，或者得到比原始图的边缘更暗的边缘信息。
    cv.morphologyEx(src=img, op=cv.MORPH_BLACKHAT, kernel=kernel)
