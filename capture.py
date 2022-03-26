import cv2 as cv
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as msgbox  # 弹窗库
import tkinter.filedialog
from PIL import Image, ImageTk
import numpy as np
import sys
import time
from typing import *

def window_quit(window: tk.Tk) -> None:
	window.quit()
	window.destroy()
	sys.exit(0)

def set_camera_is_open(isOpen: bool) -> None:
	global camera_is_open
	camera_is_open = isOpen
	state = (tk.DISABLED, tk.NORMAL)
	btn_on["state"] = state[not isOpen]
	btn_off["state"] = state[isOpen]
	btn_shot["state"] = state[isOpen]

def get_win_size(win: tk.Tk) -> tuple[int, int]:
	return win.winfo_width(), win.winfo_height()

def get_camara_size(win: tk.Tk) -> tuple[int, int]:
	sideWidth = 200
	padding = 2  # 显然不是我乐意所为
	width: int
	height: int
	width, height = get_win_size(win)
	winWidth = width
	restrict = lambda num: max(int(num), 1)
	if width >= sideWidth * 3:
		width -= sideWidth
	else:
		width = restrict(width * 2 / 3)
	# 设置按钮宽度
	fun = lambda y: (y - padding - 10) / 7  # 反函数：y = 7 * x + 10  # 为什么要有这种SB公式？别问我，测出来就是这样的
	for btn in [btn_on, btn_off, btn_shot]:
		btn["width"] = restrict(fun(winWidth - width))
	width, height = restrict(width - padding), restrict(height - padding)
	return width, height

def show_camera_pic() -> None:
	global cap_nd_array
	black = np.asarray(Image.new("RGB", (1, 1), (0, 0, 0)))
	while cap.isOpened():
		ret: bool
		ret, frame = cap.read()  # 读取
		if not ret:
			msgbox.showwarning("警告", "摄像头已关闭或断开！")
			sys.exit()
			break
		cap_nd_array = frame if camera_is_open else black
		img = cv.cvtColor(cap_nd_array, cv.COLOR_BGR2RGBA)  # 重新排列颜色通道，转换颜色使播放时保持原有色彩
		img = Image.fromarray(img)  # 将图像转换成Image对象
		img = img.resize(get_camara_size(root))
		img = ImageTk.PhotoImage(image = img)  # 转换图像对象为 TkPhoto 对象
		# img = img if cameraIsOpen else ""
		movie_label.imgtk = img
		movie_label["image"] = img
		movie_label.update()  # 每执行一次只显示一张图片，需要更新窗口实现视频播放
	root.quit()

def savePhoto() -> None:
	if not camera_is_open: return
	# path = txtPath.get() + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + ".jpg"
	# print("Save picture to " + path)
	# cv.imwrite(path, capNdArray)
	fileName = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + ".jpg"
	path = tkinter.filedialog.asksaveasfilename(
		title = "保存图片为",
		filetypes = [
			('JPEG 图片', '*.jpg'),
			('所有文件', '*')
		],
		defaultextension = ".jpg",
		initialfile = fileName
	)
	cv.imwrite(path, cap_nd_array)
	
if __name__ == "__main__":
	cap_nd_array: Optional[np.ndarray] = None
	camera_is_open: bool = True
	root: tk.Tk = tk.Tk()  # 显示窗体
	root.protocol("WM_DELETE_WINDOW", lambda: window_quit(root))
	root.title("Capture")
	root.geometry("960x540")  # 乘号是小写的 X
	movie_label = ttk.Label(root)  # 创建一个用于播放视频的label容器
	movie_label.grid(row = 0, column = 0, rowspan = 3, sticky = 'w')
	btn_on = ttk.Button(root, text = "开启", command = lambda: set_camera_is_open(True), state = tk.DISABLED)
	btn_off = ttk.Button(root, text = "关闭", command = lambda: set_camera_is_open(False))
	btn_shot = ttk.Button(root, text = "拍照", command = lambda: savePhoto())
	btn_on.grid(column = 1, row = 0)
	btn_off.grid(column = 1, row = 1)
	btn_shot.grid(column = 1, row = 2)
	
	for i in range(10):
		cap: cv.VideoCapture = cv.VideoCapture(i)  # 调整参数实现读取视频或调用摄像头
		if cap.read()[0]: break
	else:
		msgbox.showerror("错误", "未连接摄像头！")
		sys.exit()
	
	show_camera_pic()  # 播放视频
	root.mainloop()  # 启动 GUI
	# 	if cv.waitKey(100) & 0xff == ord('q'):  # 按q退出
	# 		break
	cap.release()  # 释放摄像头对象和窗口
	cv.waitKey(0)
	cv.destroyAllWindows()
