import cv2
from contoursDetection import stackImages

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	# 告诉OpenCV使用人脸识别分类器
	data_path = r"haarcascades\haarcascade_frontalface_default.xml"
	classfier = cv2.CascadeClassifier(data_path)
	# 识别出人脸后要画的边框的颜色，RGB格式
	color = (0, 255, 0)
	num = 0
	while cap.isOpened():
		ok, frame = cap.read()  # 读取一帧数据
		if not ok:
			break
		imgGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
		# 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
		faceRects = classfier.detectMultiScale(imgGrey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
		if len(faceRects) > 0:  # 大于0则检测到人脸
			for faceRect in faceRects:  # 单独框出每一张人脸
				x, y, w, h = faceRect
				# 画出矩形框
				cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
				# 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
				# font = cv2.FONT_HERSHEY_SIMPLEX
				# cv2.putText(img, 'num:%d' % num, (x + 30, y + 30), font, 1, (255, 0, 255), 4)
		# 显示图像
		cv2.imshow("Face", frame)
		c = cv2.waitKey(10)
		if c & 0xFF == ord('q'):
			break
	# 释放摄像头并销毁所有窗口
	cap.release()
	cv2.destroyAllWindows()
