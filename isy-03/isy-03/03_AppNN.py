import numpy as np
import cv2

def draw_matches(img1, img2, kp1, kp2, matches):
    """For each pair of points we draw a line between both images and circles,
    then connect a line between them.
    Returns a new image containing the visualized matches
    """
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2
    for (idx2, idx1) in matches:
        # only process the match if the keypoint was successfully matched
        # x - columns
        # y - rows
        (x1, y1) = kp1[idx1].pt
        (x2, y2) = kp2[idx2].pt
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(vis, (int(x1), int(y1)), 4, (255, 255, 0), 1)
        cv2.circle(vis, (int(x2) + w1, int(y2)), 4, (255, 255, 0), 1)
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(vis, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (255, 0, 0), 1)
    return vis


cap = cv2.VideoCapture(0)
refImg = cv2.imread('images/marker.jpg')
_, frame = cap.read()
# refImg = cv2.resize(refImg, (frame.shape[1], frame.shape[0]))
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

while True:
    _, frame = cap.read()
    refImg_kp, refImg_descriptors = sift.detectAndCompute(refImg, None)
    frame_kp, frame_descriptors = sift.detectAndCompute(frame, None)
    if refImg_descriptors is None or frame_descriptors == None  :
        continue
    raw_matches = bf.knnMatch(refImg_descriptors, frame_descriptors, k=2)
    matches = []
    distanceRatio = 0.75 # taken from ImageStitcher
    for m in raw_matches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. David Lowe's ratio = 0.75)
        # in other words - panorama images need some useful overlap
        if len(m) == 2 and m[0].distance < m[1].distance * distanceRatio:
            # filter matches that are to far away
            matches.append((m[0].trainIdx, m[0].queryIdx))
    status = np.ones((len(matches)))
    foo = draw_matches(refImg, frame, refImg_kp, frame_kp, matches)
    cv2.imshow("match", foo)
    key = cv2.waitKey(25)
    if key == ord('q'):
        break
