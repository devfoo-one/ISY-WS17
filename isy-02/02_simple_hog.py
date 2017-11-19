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

def imshow(img, title='DEBUG'):  # TODO: RMD
    cv2.imshow(title, img)
    cv2.waitKey(0)


def plot_histogram(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints):
    # convert color to gray image and extract feature in gray
    imggray = cv2.cvtColor(imgcolor, cv2.COLOR_BGRA2GRAY)
    # compute x and y gradients
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
    sobel_x = cv2.Sobel(imggray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(imggray, cv2.CV_64F, 0, 1)
    # compute magnitude and angle of the gradients
    mog = cv2.magnitude(sobel_x, sobel_y)
    imshow(np.uint8(mog)) # TODO RMD

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        print(kp.pt, kp.size)
        # extract angle in keypoint sub window
        # extract gradient magnitude in keypoint subwindow

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        # (hist, bins) = np.histogram(...)

        plot_histogram(hist, bins)

        descr[count] = hist

    return descr


keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
test = cv2.imread('./images/hog_test/circle.jpg')
descriptor = compute_simple_hog(test, keypoints)
