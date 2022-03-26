import cv2
import time
import numpy as np

protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

threshold = 0.2

# 读取内置摄像头或者usb摄像头
cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)

# 处理结果保存成视频
# vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

# 加载模型权重
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
k = 0
while True:
	k+=1
	t = time.time()
	# 读取每一帧的数据
	hasFrame, frame = cap.read()
	frameCopy = np.copy(frame)
	if not hasFrame:
		cv2.waitKey()
		break
	
	# blobFromImage将图像转为blob
	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
							  (0, 0, 0), swapRB=False, crop=False)

	net.setInput(inpBlob)

	# forward实现网络推断
	# 模型可生成22个关键点，其中21个点是人手部的，第22个点代表着背景
	output = net.forward()

	print("forward = {}".format(time.time() - t))

	# Empty list to store the detected keypoints
	points = []

	for i in range(nPoints):
		probMap = output[0, i, :, :]
		probMap = cv2.resize(probMap, (frameWidth, frameHeight))

		# 找到精确位置
		minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

		if prob > threshold :
			cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

			points.append((int(point[0]), int(point[1])))
		else :
			points.append(None)

	# 画出关键点
	for pair in POSE_PAIRS:
		partA = pair[0]
		partB = pair[1]

		if points[partA] and points[partB]:
			cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
			cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

	print("Time Taken for frame = {}".format(time.time() - t))

	cv2.imshow('webcam', frame)
	# 监听键盘事件
	key = cv2.waitKey(1)
	if key == 27:
		break

	print("total = {}".format(time.time() - t))

	# vid_writer.write(frame)

# vid_writer.release()