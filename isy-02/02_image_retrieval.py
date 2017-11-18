from queue import PriorityQueue

import cv2
import glob
import numpy as np

############################################################
#
#              Simple Image Retrieval
#
############################################################

sift = cv2.xfeatures2d.SIFT_create()


# implement distance function
def distance(a, b):
    # YOUR CODE HERE
    dist = 0.0
    for vec_a, vec_b in zip(a, b):
        for elem_a, elem_b in zip(vec_a, vec_b):
            dist += (elem_a - elem_b) ** 2
    return int(np.sqrt(dist))


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    # YOUR CODE HERE
    stepsN = 15  # how many grid elements should the uniform grid have?
    stepSize = 1.0 / stepsN
    stepFactors = np.arange(0.0, 1.0, stepSize)
    for f_w in stepFactors:
        for f_h in stepFactors:
            k = cv2.KeyPoint(int(w * f_w), int(h * f_h), _size=keypointSize)
            keypoints.append(k)
    return keypoints


def getDescriptor(img):
    img_h, img_w = img.shape[:2]
    # 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
    keypoints = create_keypoints(img_w, img_h)
    # 3. use the keypoints for each image and compute SIFT descriptors
    #    for each keypoint. this compute one descriptor for each image.
    _, descriptors = sift.compute(img, keypoints)
    return descriptors


def getFancyResultImage(queryImage, rankedCandidates):
    # assume that images are squares
    destinationHeight = 200
    fancyImage = cv2.resize(queryImage, (destinationHeight, destinationHeight))
    rankedList = []
    while not rankedCandidates.empty():
        _, _, img = rankedCandidates.get()
        rankedList.append(cv2.resize(img, (int(destinationHeight / 2), int(destinationHeight / 2))))
    # add a black image to avoid IndexError
    rankedList.append(np.zeros((int(destinationHeight / 2), int(destinationHeight / 2), 3), dtype=np.uint8))
    results = np.zeros((destinationHeight, 0, 3), dtype=np.uint8)
    for i in range(0, int(len(rankedList) / 2)):
        topImg = rankedList[i]
        bottomImg = rankedList[i + int(len(rankedList) / 2)]
        block = np.concatenate((topImg, bottomImg), axis=0)
        results = np.concatenate((results, block), axis=1)
    fancyImage = np.concatenate((fancyImage, results), axis=1)
    return fancyImage


# 1. preprocessing and load
images_paths = glob.glob('./images/db/*/*.jpg')
images = []

for path in images_paths:
    img = cv2.imread(path)
    descriptors = getDescriptor(img)
    images.append((img, getDescriptor(img)))

queries = []
for query_path in glob.glob('./images/db/*.jpg'):
    img = cv2.imread(query_path)
    queries.append((img, getDescriptor(img)))

for queryImage, queryDescriptor in queries:
    rankedCandidates = PriorityQueue()
    # PriorityQueue seems to crash if prio values are not distinct. So i intruduced a counter.
    # https://stackoverflow.com/questions/9289614/how-to-put-items-into-priority-queues
    for i, (image, descriptor) in enumerate(images):
        # 4. use one of the query input image to query the 'image database' that
        #    now compress to a single area. Therefore extract the descriptor and
        #    compare the descriptor to each image in the database using the L2-norm
        #    and save the result into a priority queue (q = PriorityQueue())
        dist = distance(queryDescriptor, descriptor)
        rankedCandidates.put((dist, i, image))
    # 5. output (save and/or display) the query results in the order of smallest distance
    cv2.imshow('Fancy Result', getFancyResultImage(queryImage, rankedCandidates))
    cv2.waitKey(0)