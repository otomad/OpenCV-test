import math
import tkinter.filedialog
from typing import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import filter

def _open_pic(arg):
	path = tkinter.filedialog.askopenfilename()
	if path != "": _show_pic(path)

def _show_pic(path):
	frame = cv2.imread(path)  # 读取图片
	frame = _resize_pic(frame)  # 缩小图片
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	images: list[tuple[any, str, Optional[str]]] = [
		(rgb, "RGB"),  # RGB
		(gray, "Gray", "gray"),  # Gray
		(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), "HSV"),  # HSV
		(cv2.cvtColor(frame, cv2.COLOR_BGR2HLS), "HLS"),  # HLS
		(frame, "BGR"),  # BGR
		# (cv2.cvtColor(frame, cv2.COLOR_BGR2LUV), "LUV"),  # LUV
		# (cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), "LAB"),  # LAB
		# (cv2.cvtColor(frame, cv2.COLOR_BGR2YUV), "YUV"),  # YUV
		(cv2.applyColorMap(gray, 2), "Thermal"),
		(filter.negative(rgb), "Negative"),
		(filter.canny(frame), "Canny"),
	]
	
	for i in range(len(images)):
		img = images[i]
		src = img[0]
		title: str = img[1]
		cmap: Optional[str] = img[2] if len(img) >= 3 else None
		plt.subplot(math.ceil(len(images) / COLUMN), COLUMN, i + 1)
		plt.imshow(src, cmap)
		plt.title(title)


def _resize_pic(image: np.array, thumb_size: int=400):
	width = image.shape[1]
	height = image.shape[0]
	if (vmax := max(width, height)) <= thumb_size:
		return image
	image = image.copy()
	new_width, new_height = thumb_size, thumb_size
	if vmax == width:
		new_height = int(height / width * new_width)
	else:
		new_width = int(width / height * new_height)
	print("from: %s; to: %s" % ((width, height), (new_width, new_height)))
	return cv2.resize(image, (new_width, new_height))


if __name__ == "__main__":
	COLUMN = 4
	# path = r"C:\Users\DELL\Pictures\810a19d8bc3eb135a3390086480a06dbfd1f442a.jpg"  # path
	plt.figure(figsize=(12, 7))  # 调整窗口大小的语句必须在前面执行
	
	window_name = 'Image'  # 图片展示窗口名称
	buttonaxe = plt.axes([0.90, 0.03, 0.06, 0.04])
	button1 = matplotlib.widgets.Button(buttonaxe, "Open", color = "khaki", hovercolor = "yellow")
	button1.on_clicked(_open_pic)
	
	plt.show()
	plt.title(window_name)  # 没鸟用
