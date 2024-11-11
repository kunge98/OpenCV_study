import cv2 as cv
import numpy as np

if __name__ == '__main__':

    img = cv.imread('test.png')
    # 需要模板图片
    template = cv.imread('test2.png')

    # 模板匹配
    # 与卷积类似，也是滑动窗口进行匹配，计算模板与（图像被模板覆盖地方）的差别程度
    # 将模板在图片上滑动(从左向右，从上向下)，遍历所有滑窗，计算匹配度，将所有计算结果保存在一个矩阵种，并将矩阵中匹配度最高的值作为匹配结果。
    # method=cv.TM_xxxx_NORMED,结尾尽量使用带归一化的API
    # res为匹配后的形状大小为（原图长-模板长+1，原图宽-模板宽+1）
    res = cv.matchTemplate(image=img, templ=template, method=cv.TM_SQDIFF_NORMED)
    # 根据匹配的结果确定模板在原图中的位置
    # min_loc代表模板匹配在原图中左上角的位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # 如果需要匹配多个对象，需要设置阈值threshold，这样就将res进一步筛选

    # 模板匹配不使用尺度变换、视角变换后的图像
    # 如果非要匹配，要是用关键点匹配算法（SIFT、SURF）
    # 通过关键点检测算法获取模板和检测图像中的关键点，然后使用关键带你匹配算法处理
    # 因为这些关键点可以很好的处理尺度变换、市交变换、旋转变化、光照变化等







