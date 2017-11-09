import numpy as np
import cv2
import math
import sys
from ImageStitcher import *


############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
pano6 = cv2.imread('images/pano6.jpg')
pano5 = cv2.imread('images/pano5.jpg')
pano4 = cv2.imread('images/pano4.jpg')
pano3 = cv2.imread('images/pano3.jpg')
pano2 = cv2.imread('images/pano2.jpg')
pano1 = cv2.imread('images/pano1.jpg')

# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher([pano6, pano5, pano4]) # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    None
    # YOUR CODE HERE
    # output all matching images
    # output result
    # Note: if necessary resize the image

