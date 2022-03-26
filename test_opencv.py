import cv2 as cv

scr = cv.imread("C:/Users/DELL/Desktop/baozan.jpg")  # 获取图片资源
cv.namedWindow("Picture Output Box", 0)
cv.resizeWindow("Picture Output Box", 500, 500)  # 设置窗口大小为500*500
cv.imshow("Picture Output Box", scr)
cv.waitKey(0)
cv.destroyAllWindows()  # 关闭窗口
