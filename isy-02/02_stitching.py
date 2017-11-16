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

stitchers = []
# order of input images is important is important (from right to left)
# imageStitcher = ImageStitcher([pano4, pano5, pano6]) # list of images


stitchers.append(ImageStitcher([pano6, pano5, pano4]))
stitchers.append(ImageStitcher([pano3, pano2, pano1]))

for stitcher_i, stitcher in enumerate(stitchers):
    (matchlist, result) = stitcher.stitch_to_panorama()
    if not matchlist:
        print("We have not enough matching keypoints to create a panorama")
    else:
        # YOUR CODE HERE
        for i, match in enumerate(matchlist):
            cv2.imshow('stitcher {} match {}'.format(stitcher_i, i), match)
        # output all matching images
        # output result
        cv2.imshow('Result stitcher {}'.format(stitcher_i), result)

cv2.waitKey(0)
# Note: if necessary resize the image
