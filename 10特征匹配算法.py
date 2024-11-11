import numpy as np
import cv2 as cv

if __name__ == '__main__':

    img = cv.imread('test.png')
    img2 = cv.imread('test.png')

    # Brute-Force 暴力匹配
    sift = cv.SIFT_create()
    # 检测并计算特征点与向量
    kp, des = sift.detectAndCompute(img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 一对一匹配
    # 进行匹配，normType使用归一化处理，crossCheck表示两个特征点要匹配
    bf = cv.BFMatcher(normType=True, crossCheck=True)
    matches = bf.match(des, des2)
    # 将matches中匹配的相关特征点根据距离大小进行排序匹配
    matches = sorted(matches, key=lambda x : x.distance)
    # 取出img的关键点和img2的关键点，两两相匹配，取出前10个结果进行画
    img3 = cv.drawMatches(img1=img, keypoints1=kp,
                          img2=img2,  keypoints2=kp2,
                          matches1to2=matches[:10], outImg=None, flags=2)
    img3 = cv.imshow('img3', img3)

    # 一对多匹配
    bf = cv.BFMatcher()
    # 一对多相当于找多个点进行匹配，knn算法的思想，所以api也带着knn
    matches = bf.knnMatch(des, des2, k=3)

    # 过滤操作,m的距离如果小于一定的阈值，就添加到结果列表中
    res = []
    for m, n in matches:
        # 0.75超参数
        if m.distance < n.distance * 0.75:
            res.append([m])

    # 取出img的关键点和img2的关键点，两两相匹配，取出前10个结果进行画
    img3 = cv.drawMatches(img1=img, keypoints1=kp,
                          img2=img2,  keypoints2=kp2,
                          matches1to2=res, outImg=None, flags=2)
    img3 = cv.imshow('img3', img3)

    # 一对一，一对多的区别：
    # 在匹配算法上，都是使用cv.BFMatcher()创建对象，
    # 但是在实例化对象上，一个用match，一个是knnMatch
    # 特征选择上两者几乎一样

    # 如果匹配的特征点出错了怎么办？
    # RANSAC 随机抽样一致算法，类似于最小二乘
    # 它是一种不确定的算法——它有一定的概率得出一个合理的结果；
    # 优点是它能鲁棒的估计模型参数
    # 计算参数的迭代次数没有上限；如果设置迭代次数的上限，
    # 得到的结果可能不是最优的结果，甚至可能得到错误的结果


