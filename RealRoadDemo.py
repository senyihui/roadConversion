import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
import glob
import pickle

image = cv2.imread('road.jpg')
# cv2.imshow('img', image)

img_cvtGrey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转化grey空间

ret, img_binary0 = cv2.threshold(img_cvtGrey, 125, 255, cv2.THRESH_BINARY)  # 采用固定阈值100进行二值化
cv2.imshow("cvtGRAYO", img_binary0)  # 示固定阈值二值化结果


# th2 = cv2.adaptiveThreshold(img_binary0, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
# cv2.imshow("cvtGRAY1", th2) # 自适应阈值
#
# th3 = cv2.adaptiveThreshold(img_binary0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# cv2.imshow("cvtGRAY2", th3)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字结构
opening = cv2.morphologyEx(img_binary0, cv2.MORPH_OPEN, kernel)   # 开运算
cv2.imshow('canny1', opening)

dst = cv2.dilate(opening, None, iterations=2)
cv2.imshow("cvtGRAY3", dst)     # 扩大白色区域


cv2.waitKey(0)
