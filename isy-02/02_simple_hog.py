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
    mog = np.uint8(mog)

    # test images are noisy jpegs.. so thresholding leads to better results
    # _, mog = cv2.threshold(mog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    for count, kp in enumerate(keypoints):
        kp_y, kp_x = kp.pt
        y1 = int(kp_y - int(kp.size / 2))
        y2 = int(kp_y + int(kp.size / 2)) + 1
        x1 = int(kp_x - int(kp.size / 2))
        x2 = int(kp_x + int(kp.size / 2)) + 1
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow
        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        # (hist, bins) = np.histogram(...)
        # Answer: Where there is no gradient, angle returns 0 (instead of NaN or None) which means an angle of 0 or 360 degrees.

        mog_window = mog[y1:y2, x1:x2]
        sobel_x_window = sobel_x[y1:y2, x1:x2]
        sobel_y_window = sobel_y[y1:y2, x1:x2]
        angle = cv2.phase(sobel_x_window, sobel_y_window)
        angle = angle[mog_window != 0]
        (hist, bins) = np.histogram(angle, bins=8, range=(0.0, 2 * np.pi), density=True)
        plot_histogram(hist, bins)
        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 20)]

# test for all test images
testImages = []
testImages.append(cv2.imread('./images/hog_test/diag.jpg'))
testImages.append(cv2.imread('./images/hog_test/horiz.jpg'))
testImages.append(cv2.imread('./images/hog_test/vert.jpg'))
testImages.append(cv2.imread('./images/hog_test/vert.bmp'))
testImages.append(cv2.imread('./images/hog_test/circle.jpg'))
for test in testImages:
    descriptor = compute_simple_hog(test, keypoints)
