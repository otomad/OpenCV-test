from typing import *

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np

# ^ 设置几张图片拼接
def stackImages(scale, imgArray):
	rows = len(imgArray)
	cols = len(imgArray[0])
	# & 输出一个 rows * cols 的矩阵（imgArray）
	print(rows, cols)
	# & 判断imgArray[0] 是不是一个list
	rowsAvailable = isinstance(imgArray[0], list)
	# & imgArray[][] 是什么意思呢？
	# & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
	# & 而shape[1]就是width，shape[0]是height，shape[2]是
	width = imgArray[0][0].shape[1]
	height = imgArray[0][0].shape[0]
	
	# & 例如，我们可以展示一下是什么含义
	# cv2.imshow("img", imgArray[0][1])
	
	if rowsAvailable:
		for x in range(rows):
			for y in range(cols):
				# & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
				if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
					imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
				else:
					imgArray[x][y] = cv2.resize(
						imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale
					)
				# & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
				if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
		# & 设置零矩阵
		imageBlank = np.zeros((height, width, 3), np.uint8)
		hor = [imageBlank] * rows
		# hor_con = [imageBlank] * rows
		for x in range(rows):
			hor[x] = np.hstack(imgArray[x])
		ver = np.vstack(hor)
	# & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
	else:
		for x in range(rows):
			if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
				imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
			else:
				imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
			if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
		hor = np.hstack(imgArray)
		ver = hor
	return ver

def perror(text: str):
	print(f"\033[1;31m{text}\033[0m")

def getDictByIndex(_dict: dict, index: int, value: bool = False) -> Optional[any]:  # 根据序号查找字典的键或值
	i: int = 0
	for key in _dict:
		if i == index:
			return key if not value else _dict[key]
		i += 1  # Python 居然没有 i++ ？
	else:
		return None

def getColorName(color: tuple[int, int, int]) -> Union[str, bool]:  # 参数为元组 (r, g, b)
	preset: dict[str, tuple[int, int, int]] = {
		"白色": (255, 255, 255),
		"黑色": (0, 0, 0),
		"红色": (255, 0, 0),
		"绿色": (0, 255, 0),
		"蓝色": (0, 0, 255),
		"黄色": (255, 255, 0),
		"青色": (0, 255, 255),
		"洋红色": (255, 0, 255),
		"橙色": (255, 128, 0),
		"灰色": (128, 128, 128),
		"紫色": (128, 0, 255),
	}
	D: int = len(getDictByIndex(preset, 0, True))  # D - dimensional
	if len(color) != D:
		perror("参数长度不匹配！")
		return False
	variance = { }
	for name in preset:
		value: tuple[int, int, int] = preset[name]
		s2: int = 0
		for c in range(D):
			s2 += (color[c] - value[c]) ** 2  # s2 就是方差，因为这个计算的公式就是类比方差的计算公式
		variance[name] = s2
	sort = sorted(variance.items(), key = lambda kv: (kv[1], kv[0]))
	return sort[0][0]

def bgr2rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
	return color[2], color[1], color[0]

def putTextZh(
	img, text, org, fontFace, fontScale, color, thickness = None, lineType = None, bottomLeftOrigin = None,
	fontFamily = r"C:\Windows\Fonts\msyh.ttc"
):
	if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
		img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	# 创建一个可以在给定图像上绘图的对象
	draw = PIL.ImageDraw.Draw(img)
	# 字体的格式
	textSize = int(fontScale * 10)
	fontStyle = PIL.ImageFont.truetype(fontFamily, textSize, encoding = "utf-8")
	# 转换字体颜色
	textColor = bgr2rgb(color)
	# 绘制文本
	draw.text(org, text, textColor, font = fontStyle)
	# 转换回OpenCV格式
	return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

class ShapeAnalysis:  # 定义形状分析的类
	def __init__(self):
		self.threshold: float = 0.5
		self.shapesName: list[str] = "triangle,rectangle,polygons,circles,pentagram,square".split(',')
		shapesNameChinese: list[str] = "三角形、矩形、多边形、圆形、五角星、正方形".split('、')
		self.shapesNameChinese: dict[str, str] = { }
		self.shapes: dict[str, int] = { }
		for i in range(len(self.shapesName)):
			shapeName = self.shapesName[i]
			self.shapesNameChinese[shapeName] = shapesNameChinese[i]
			self.shapes[shapeName] = 0
	
	def draw_text_info(self, image: np.ndarray) -> np.ndarray:
		text = ""
		for i in range(len(self.shapesName)):
			shapeName = self.shapesName[i]
			shapeNameChinese = self.shapesNameChinese[shapeName]
			# cv2.putText(
			'''
			putText(
				image,
				f"{shapeNameChinese}: {str(self.shapes[shapeName])}",
				(10, (i + 1) * 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 1
			)
			'''
			text += f"{shapeNameChinese}: {str(self.shapes[shapeName])}\n"
		image = putTextZh(image, text, (10, 20), None, 2, (255, 255, 0))
		return image
	
	def analysis(
		self,
		frame: np.ndarray,
		threshold: float = 0.5,
		scale: float = 0.7,
		blur: int = 7,
		minArea: int = 0,
	):
		self.threshold = threshold
		h, w, ch = frame.shape
		imgResult = np.zeros((h, w, ch), dtype = np.uint8)
		# 二值化图像
		print("开始检测边缘...")
		imgGrey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 要二值化图像，要先进行灰度化处理
		imgGrey = cv2.medianBlur(imgGrey, blur)
		# imgTh = cv2.adaptiveThreshold(imgGrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)  # 局部阈值
		imgBin = cv2.threshold(imgGrey, int(self.threshold * 255), 255, cv2.THRESH_BINARY_INV)[
			1]  # cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
		# cv2.imshow("input image", frame)
		contours = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		# contours = cv2.findContours(imgBin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		imgContours = frame.copy()
		cv2.drawContours(imgContours, contours, -1, (0, 0, 255), 2)
		for contour in range(len(contours)):
			# 提取与绘制轮廓
			cv2.drawContours(imgResult, contours, contour, (0, 0, 255), 2)
			# 轮廓逼近
			epsilon = 0.01 * cv2.arcLength(contours[contour], True)
			approx = cv2.approxPolyDP(contours[contour], epsilon, True)
			# 计算面积与周长
			p = cv2.arcLength(contours[contour], True)
			area = cv2.contourArea(contours[contour])
			if area == 0: continue
			if area < minArea: continue
			# 分析几何形状
			corners = len(approx)
			situation = [
				corners == 3,
				corners == 4,
				4 < corners < 10,
				corners > 10,
				corners == 10,
				False
			]
			shape_type = ""
			chineseNumber = "〇一二三四五六七八九十"
			for i in range(len(situation)):
				if situation[i]:
					# 正方形修正
					square = False
					if len(approx) == 4:
						(x, y, w, h) = cv2.boundingRect(approx)
						if abs((float(w) / h) - 1) < 0.1:  # 规定一定的精度
							square = True
					# 常规
					shapeName = self.shapesName[i] if not square else "square"
					self.shapes[shapeName] += 1
					shape_type = self.shapesNameChinese[shapeName]
					# 多边形修正
					if shape_type == "多边形": shape_type = f"{chineseNumber[corners]}边形"
					break
			else:
				continue
			# 求解中心位置
			mm = cv2.moments(contours[contour])
			center = [0, 0]
			if mm['m00']:
				center[0] = int(mm['m10'] / mm['m00'])
				center[1] = int(mm['m01'] / mm['m00'])
			cv2.circle(imgResult, center, 3, (0, 0, 255), -1)
			# 颜色分析和提取
			color = bgr2rgb(frame[center[1]][center[0]])
			colorName = getColorName(color)
			# 图形标注
			imgResult = putTextZh(imgResult, colorName + shape_type, center, None, 1.5, (0, 0, 255))
			# 输出“总结”
			print("周长: %.3f, 面积: %.0f 颜色: %s %s 形状: %s " % (p, area, colorName, str(color), shape_type))
		imgWithText = self.draw_text_info(imgResult)
		imgStack = stackImages(scale, ([frame, imgGrey, imgBin], [imgContours, imgResult, imgWithText]))
		cv2.imshow("Analysis Result", imgStack)
		return self.shapes

if __name__ == "__main__":
	# path = r"C:\Users\DELL\Pictures\shape.png"
	# src = cv2.imread(path, 1)
	cap: cv2.VideoCapture = cv2.VideoCapture(0)
	while cap.isOpened():
		ok: bool
		src: np.ndarray
		ok, src = cap.read()
		if not ok:
			break
		ld: ShapeAnalysis = ShapeAnalysis()
		ld.analysis(
			src,
			threshold = 0.5,
			scale = 0.7,
			blur = 5,
		)
		if cv2.waitKey(1) & 0xFF == ord('q'):  # 按键q后break
			break
	cap.release()
	cv2.destroyAllWindows()
