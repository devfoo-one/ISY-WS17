import numpy as np
import cv2
import glob
from sklearn import svm
import threading

############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px

sift = cv2.xfeatures2d.SIFT_create()
labels = {1: 'cars', 2: 'faces', 3: 'flowers'}

class FeatureExtractionThread(threading.Thread):
    def __init__(self, labelInt, path, datastore):
        threading.Thread.__init__(self)
        self.labelInt = labelInt
        self.path = path
        self.datastore = datastore
    def run(self):
        print('Start extracting SIFT features of {}'.format(self.path))
        self.img = cv2.imread(self.path)
        self.h, self.w = self.img.shape[:2]
        self.keypoints = []
        self.keypointSize = 15
        self.stepsN = 256  # how many grid elements should the uniform grid have?
        self.stepSize = 1.0 / self.stepsN
        self.stepFactors = np.arange(0.0, 1.0, self.stepSize)
        for self.f_w in self.stepFactors:
            for self.f_h in self.stepFactors:
                self.k = cv2.KeyPoint(int(self.w * self.f_w), int(self.h * self.f_h), _size=self.keypointSize)
                self.keypoints.append(self.k)
        _, self.descriptor = sift.compute(self.img, self.keypoints)
        threadLock.acquire(blocking=True)
        self.datastore.append((self.labelInt, self.descriptor.ravel(), self.path))
        print('Finished extracting SIFT features of {}'.format(self.path))
        threadLock.release()


threadLock = threading.Lock()
threads = []
train = []
print('Start extracting SIFT features for training data')
for labelInt, labelString in labels.items():
    train_paths = glob.glob('./images/db/train/{}/*.jpg'.format(labelString))
    for path in train_paths:
        thread = FeatureExtractionThread(labelInt, path, train)
        thread.start()
        threads.append(thread)
for t in threads:
    t.join()
threads.clear()

print('Finished extracting SIFT features for training data')

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
if len(train) is 0:
    quit(1)

y_train = train[0][0]
X_train = np.asmatrix(train[0][1])
for y, X, _ in train[1:]:
    X_train = np.vstack((X_train, X))
    y_train = np.vstack((y_train, y))

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)
print('Training SVM')
clf = svm.LinearSVC()
clf.fit(X_train, y_train.ravel())
print('Done training SVM')

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image

print('Generating features for test images.')
test = []
for test_path in glob.glob('./images/db/test/*.jpg'):
    thread = FeatureExtractionThread(999, test_path, test)
    thread.start()
    threads.append(thread)
for t in threads:
    t.join()
threads.clear()

print('Finished generating features for test images.')

# 5. output the class + corresponding name
for _, features, name in test:
    y = clf.predict([features])
    print('SVM says that {} seems to be category {} ({}).'.format(name, y[0], labels.get(int(y[0]))))

