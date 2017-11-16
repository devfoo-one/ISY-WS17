import cv2
import glob
import numpy as np
from asyncio import PriorityQueue


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
        for elem_a, elem_b in zip(vec_a,vec_b):
            dist += (elem_a - elem_b)**2
    return np.sqrt(dist)

def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11

    # YOUR CODE HERE
    stepsN = 11
    stepSize = 1.0 / stepsN
    stepFactors = np.arange(stepSize, 1.0, stepSize)
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

# 1. preprocessing and load
images_paths = glob.glob('./images/db/*/*.jpg')
images = []

for path in images_paths:
    img = cv2.imread(path)
    descriptors = getDescriptor(img)
    # images.append((img, descriptors))
    images.append((img, getDescriptor(img)))

queries = []
for query_path in glob.glob('./images/db/*.jpg'):
    img = cv2.imread(query_path)
    queries.append((img, getDescriptor(img)))


for queryImage, queryDescriptor in queries: # TODO: DEBUG FILTER
    q = PriorityQueue()
    cv2.imshow('query', queryImage)  # TODO: RMD
    for image, descriptor in images:
        dist = distance(queryDescriptor, descriptor)
        q.put((image, dist))  #TODO: BUG, this adds nothing!

    while not q.empty():
        cv2.imshow('candidate', q.get())
        cv2.waitKey(0)


# 4. use one of the query input image to query the 'image database' that
#    now compress to a single area. Therefore extract the descriptor and
#    compare the descriptor to each image in the database using the L2-norm
#    and save the result into a priority queue (q = PriorityQueue())

# YOUR CODE HERE

# 5. output (save and/or display) the query results in the order of smallest distance

# YOUR CODE HERE
