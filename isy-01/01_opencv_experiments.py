import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    if ch == ord('2'):
        print('MODE 2 - HSV')
        mode = 2
    if ch == ord('3'):
        print('MODE 3 - LAB')
        mode = 3
    if ch == ord('4'):
        print('MODE 4 - YUV')
        mode = 4
    if ch == ord('5'):
        print('MODE 5 - Adaptive Thresholding (Gausian)')
        mode = 5
    if ch == ord('6'):
        print('MODE 6 - Adaptive Thresholding (Otsu)')
        mode = 6
    if ch == ord('7'):
        print('MODE 7 - Adaptive Thresholding (Otsu) with Gaussian Blur')
        mode = 7
    if ch == ord('8'):
        print('MODE 8 - Canny Edge Detection')
        mode = 8

    if ch == ord('q'):
        break

    if mode == 1:
        # just example code
        # your code should implement
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if mode == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if mode == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    if mode == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    if mode == 5:
        # https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if mode == 6:
        # https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mode == 7:
        # https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_REPLICATE)
        ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mode == 8:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        """
        https://en.wikipedia.org/w/index.php?title=Canny_edge_detector&oldid=805883077
        Robust method to determine the dual-threshold value
        In order to resolve the challenges where it is hard to determine the dual-threshold value empirically,
        Otsu's method [11] can be used on the non-maximum suppressed gradient magnitude image to generate the
        high threshold. The low threshold is typically set to 1/2 of the high threshold in this case.
        """
        otsu_threshold, _ = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame = cv2.Canny(frame, otsu_threshold / 2, otsu_threshold)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
