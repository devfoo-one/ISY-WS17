import cv2

cap = cv2.VideoCapture(0)
cv2.namedWindow('Interactive Systems: Towards AR Tracking')
while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # YOUR CODE HERE

    ret, frame = cap.read()
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()  # code taken from https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    kp = sift.detect(frame_bw)
    frame_bw = cv2.cvtColor(frame_bw, cv2.COLOR_GRAY2BGR)  # needed to get color markers onto it
    cv2.drawKeypoints(frame_bw, kp, frame_bw, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Interactive Systems: Towards AR Tracking', frame_bw)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
