import numpy as np
import cv2 as cv

if __name__ == '__main__':

    img = cv.imread('test.png')

    # 傅里叶变换（看懵逼了）
    # 高频：变化剧烈的灰度分量，图像的边界
    # 低频：变化缓慢的灰度分量，图像中间

    # 滤波
    # 高通滤波器：只保留高频，边界信息会更加清晰，图像会得到细节上的增强
    # 低通滤波器：只保留低频，边界信息得不到保留，图像会变模糊

    # 霍夫变换
    # 是一种在图像中寻找直线、圆、椭圆等简单的几何形状的方法
    # 把笛卡尔坐标系下的点或者直线映射到霍夫空间
    # 极坐标系内的一个点映射到霍夫空间中是一条直线（曲线）
    # 极坐标系内的一条直线映射到霍夫空间中是一个点




