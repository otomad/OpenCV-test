import cv2
import numpy as np

def negative(img: np.array):
	img = img.copy()
	for height in range(len(img)):
		for width in range(len(img[height])):
			for color in range(len(img[height, width])):
				img[height, width, color] = 255 - img[height, width, color]
	return img


def canny(img: np.array):
	return cv2.Canny(img, 50, 50)

