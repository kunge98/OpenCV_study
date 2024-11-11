import cv2 as cv
import numpy as np


# 展示图片
def show_img():
    img = cv.imread('test.png')
    # 显示图像
    cv.imshow('input', img)
    # 保存图像
    cv.imwrite('./xxx.jpg', img)
    # 0表示任意键终止，其他的数值表示图片展示的时间，单位是毫秒
    cv.waitKey(0)
    cv.destroyAllWindows()


# 转变为灰度图和hsv图
def color_space():
    img = cv.imread('test.png')
    # 可以来回转换
    gray = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    cv.imshow('hsv', hsv)
    cv.waitKey(0)

    cv.destroyAllWindows()


# 对象创建赋值,cv2读取的图片都是numpu对象
def mat_demo():
    img = cv.imread('test.png')
    cv.imshow('input', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print(img.shape)
    print(type(img))
    print(img)


# 图像读写操作
def pixel_demo():
    img = cv.imread('test.png')
    cv.imshow('input', img)

    h, w, c = img.shape
    for row in range(h):
        for columns in range(w):
            b, g, r = img[row, columns]
            # 取反,灵活操作
            img[row, columns] = (255-b, 255-g, 255-r)
    cv.imshow('result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 图像像素算术操作
def math_demo():
    img = cv.imread('test2.png')
    cv.imshow('input', img)
    # 创建空白图像,像素点均为50
    blank = np.zeros_like(img)
    blank[:,:] = (50, 50, 50)
    cv.imshow('blank', blank)
    # 两张图的像素值进行相加
    result1 = cv.di(img, blank)
    cv.imshow('result1', result1)
    # 两张图的像素值进行相减
    result2 = cv.subtract(img, blank)
    cv.imshow('result2', result2)
    # 相乘 cv2.multiply
    # 相除 cv2.divide
    cv.waitKey(0)
    cv.destroyAllWindows()


# TrackBar滚动条,目的就是利用按钮滑动窗口，调整图像的像素值对比度，实际开发中没有什么用处
def nothing(x):
    print(x)
def TrackBar():
    img = cv.imread('test2.png')
    cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('lightness', 'input', 0, 100, nothing)
    cv.imshow('input', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 键盘响应函数,利用键盘上的数字，对原图进行相应的设置
def keys_demo():
    img = cv.imread('test2.png')
    cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
    cv.imshow('input', img)
    while True:
        c = cv.waitKey(1)
        if c == '49':
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imshow('result', gray)
        if c == '50':
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            cv.imshow('result', hsv)
        if c == '27':
            break
    cv.destroyAllWindows()


# 自带颜色查找表操作
def color_table_demo():
    img = cv.imread('test2.png')
    cv.namedWindow('input', cv.WINDOW_AUTOSIZE)
    cv.imshow('input', img)
    while True:
        c = cv.waitKey(1)
        if c == '49':
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            cv.imshow('result', gray)
        if c == '50':
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            cv.imshow('result', hsv)
        if c == '27':
            break
    cv.destroyAllWindows()
# show_img()
# color_space()
# mat_demo()
# pixel_demo()
# math_demo()
# keys_demo()
# color_table_demo()


if __name__ == '__main__':

    img = cv.imread('test.png')
    # 读取视频
    video= cv.VideoCapture()

    # 图像的通道不是RGB，而是BGR！,切割每个通道
    b, g, r = cv.split(img)
    # 合并
    cv.merge()

    # ROI截取特定区域,切片操作
    img = img[:,:]

    # 边界填充
    # 根据不同的  borderType=cv.BORDER_xxxxx 得到不同的图像填充
    cv.copyMakeBorder(src=img, top=50, bottom=50, left=50, right=50, borderType=cv.BORDER_CONSTANT)

    # 一、数值计算
    # 加减乘除，维度需要一致，数值如果超过255,一律按照255，
    # 如果是在numpy中，超出255的数值，会进行取余运算，numpy计算方法为“img1 + img2”
    # 使用cv的add方法时，图像会变亮，超过255的都当做255处理，255为白色
    # 使用numpy的“+”时，图像会变暗，超过255的会进行取模运算，0为黑色
    cv.add()
    # 使用add这一类的方法时都会有mask参数变量
    # 掩膜参数意思是提供这个掩膜，那么add就会在该区域的非空像素点上进行，
    # 将mask以外的像素值设置为0

    cv.subtract()
    cv.multiply()
    cv.divide()

    # 图像融合，前提形状一致
    cv.resize()
    img2 = cv.imread('./xxx.png')
    # addWeighted将两者权重加载一起，也就意味着融合
    # 数字所代表的的参数a，b，c
    cv.addWeighted(img, 0.5, img2, 0.5, 0)

    # 与运算
    # 任何数a与0进行与运算得到的是0
    # 任何数a与255进行与运算得到的是255
    # 任何数a与自身a进行与运算得到的是a
    # 可以使用该特点将某一部分图像设置为255和别的图像进行与运算，保留这部分区域
    # 可以实现抠图的操作
    cv.bitwise_and()

    # 或运算
    cv.bitwise_or()

    # 非运算
    cv.bitwise_not()

    # 异或运算
    # 两个数值相同为0。。。。
    # 用途：可用于图像的加密解密
    # 需要定义一幅秘钥图像，加密者需要使用二进制的原始图和秘钥图进行异或运算
    # 把结果转化为十进制得到加密图
    # 解密者需要使用二进制的加密图和秘钥图进行异或运算，把结果转化为十进制得到原始图
    cv.bitwise_xor()

    # 图像旋转
    cv.getRotationMatrix2D()

    # 仿射变换，即：线性变换+平移
    # 线性变换：旋转、位移
    # 变换后的图像仍能够保持
    # 平直性(直线经过放射变换后还是直线)
    # 和平行性(平行线经过放射变换后还是平行线)
    # 用途：图像旋转、平移等
    cv.getAffineTransform()

    # 透射变换
    # 把一个图像投影到一个新的视平面的过程，
    # 该过程包括：把一个二维坐标系转换为三维坐标系，
    # 然后把三维坐标系投影到新的二维坐标系。该过程是一个非线性变换过程，
    # 因此，一个平行四边形经过透视变换后只得到四边形，但不平行。
    # 用途较大：视角纠正、全景拼接
    cv.getPerspectiveTransform()

    # 重映射
    # 我们希望通过自定义的方式来指定如何映射，此时我们就需要进行重映射操作。
    # 把一张图像内的像素点放置到另外一幅图像内指定的位置，这个操作就是重映射。
    # 可以实现图像的复制、图像绕x、y轴翻转、互换、缩小图像
    cv.remap()















