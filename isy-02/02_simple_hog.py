import numpy as np
import cv2
import math
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################

def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def normalize(arr):
    return arr / np.max(arr)


def compute_simple_hog(imgcolor, keypoints):
    # convert color to gray image and extract feature in gray

    imggray = cv2.cvtColor(imgcolor, cv2.COLOR_BGRA2GRAY)
    # compute x and y gradients
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    sobel_x = cv2.Sobel(imggray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(imggray, cv2.CV_64F, 0, 1)
    # compute magnitude and angle of the gradients
    mog = cv2.magnitude(sobel_x, sobel_y)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    for count, kp in enumerate(keypoints):
        kp_y, kp_x = kp.pt
        y1 = int(kp_y - int(kp.size / 2))
        y2 = int(kp_y + int(kp.size / 2))
        x1 = int(kp_x - int(kp.size / 2))
        x2 = int(kp_x + int(kp.size / 2))
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        # (hist, bins) = np.histogram(...)
        # Answer: if mog = 0, cv2.phase returns a 100% phase of 0 rad which means full vertical angle at 0deg (not 0)

        mog_window = mog[y1:y2, x1:x2]
        sobel_x_window = sobel_x[y1:y2, x1:x2]
        sobel_y_window = sobel_y[y1:y2, x1:x2]
        sobel_x_window_nonzero = sobel_x_window[np.nonzero(mog_window)]
        sobel_y_window_nonzero = sobel_y_window[np.nonzero(mog_window)]
        angle = cv2.phase(sobel_x_window_nonzero, sobel_y_window_nonzero)

        (hist, bins) = np.histogram(angle, bins=8, range=(0.0, 2 * np.pi))
        hist = normalize(hist)
        plot_histogram(hist, bins)
        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
testImages = []
testImages.append(cv2.imread('./images/hog_test/diag.jpg'))
testImages.append(cv2.imread('./images/hog_test/horiz.jpg'))
testImages.append(cv2.imread('./images/hog_test/vert.jpg'))
testImages.append(cv2.imread('./images/hog_test/circle.jpg'))
for test in testImages:
    descriptor = compute_simple_hog(test, keypoints)
