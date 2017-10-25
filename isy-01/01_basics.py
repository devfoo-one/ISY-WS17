import numpy as np
import cv2

######################################################################
# IMPORTANT: Please make yourself comfortable with numpy and python:
# e.g. https://www.stavros.io/tutorials/python/
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Note: data types are important for numpy and opencv
# most of the time we'll use np.float32 as arrays
# e.g. np.float32([0.1,0.1]) equal np.array([1, 2, 3], dtype='f')

######################################################################
# A2. OpenCV and Transformation and Computer Vision Basic

# (1) read in the image Lenna.png using opencv in gray scale and in color
# and display it NEXT to each other (see result image)
# Note here: the final image displayed must have 3 color channels
#            So you need to copy the gray image values in the color channels
#            of a new image. You can get the size (shape) of an image with rows, cols = img.shape[:2]

# why Lenna? https://de.wikipedia.org/wiki/Lena_(Testbild)

def bgr2grayscale(img):
    _img = np.copy(img)
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            b, g, r, = _img[y, x]
            bw = np.uint8(0.72 * b + 0.07 * g + 0.21 * r)
            _img[y, x] = [bw, bw, bw]
    return _img

def bgr2grayscale_fast(img):
    _img = np.copy(img)
    _img = _img * np.array([[[0.72, 0.07, 0.21]]])
    bw = np.sum(_img, axis=2)[...,np.newaxis] # equiv. to [:,:,None]
    _img = np.concatenate((bw,bw,bw), axis=2).astype(np.uint8)
    return _img

M_translation = np.float32([[1,0,10],[0,1,0]])
angle = -0.5
M_rotation = np.float32([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0]])

img = cv2.imread('images/Lenna.png')
img_bw = bgr2grayscale_fast(img)
display = np.concatenate((img_bw, img), axis=1)
while True:
    cv2.imshow('ISY Aufgabe 1', display)
    key = cv2.waitKey(0)
    if key == ord('q') or key == ord('Q'):
        break
    if key == ord('r') or key == ord('R'):
        display = cv2.warpAffine(display, M_rotation, (display.shape[1], display.shape[0]))
    if key == ord('t') or key == ord('T'):
        display = cv2.warpAffine(display, M_translation, (display.shape[1], display.shape[0]))

cv2.destroyAllWindows()
# (2) Now shift both images by half (translation in x) it rotate the colored image by 30 degrees using OpenCV transformation functions
# + do one of the operations on keypress (t - translate, r - rotate, 'q' - quit using cv::warpAffine
# http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
# Tip: you need to define a transformation Matrix M
# see result image
