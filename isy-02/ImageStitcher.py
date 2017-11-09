import numpy as np
import cv2


# complete the class ImageStitcher

class ImageStitcher:
    """A simple class for stitching two images. The class expects two color images"""

    def __init__(self, imagelist):

        self.imagelist = imagelist

        # match ratio to clean feature area from non-important ones
        self.distanceRatio = 0.75
        # theshold for homography
        self.reprojectionThreshold = 4.0

    def match_keypoints(self, kpsPano1, kpsPano2, descriptors1, descriptors2):
        """This function computes the matching of image features between two different
        images and a transformation matrix (aka homography) that we will use to unwarp the images
        and place them correctly next to each other. There is no need for modifying this, we will
        cover what is happening here later in the course.
        """
        # compute the raw matches using a Bruteforce matcher that
        # compares image descriptors/feature vectors in high-dimensional space
        # by employing K-Nearest-Neighbor match (more next course)
        bf = cv2.BFMatcher()
        rawmatches = bf.knnMatch(descriptors1, descriptors2, 2)
        matches = []

        # loop over the raw matches and filter them
        for m in rawmatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. David Lowe's ratio = 0.75)
            # in other words - panorama images need some useful overlap
            if len(m) == 2 and m[0].distance < m[1].distance * self.distanceRatio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        print("matches:", len(matches))
        # we need to compute a homography - more next course
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsPano1 = np.float32([kpsPano1[i].pt for (_, i) in matches])
            ptsPano2 = np.float32([kpsPano2[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsPano1, ptsPano2, cv2.RANSAC, self.reprojectionThreshold)

            # we return the corresponding perspective transform and some
            # necessary status object + the used matches
            return (H, status, matches)

        # otherwise, no homograpy could be computed
        return None

    def draw_matches(self, img1, img2, kp1, kp2, matches, status):
        """For each pair of points we draw a line between both images and circles,
        then connect a line between them.
        Returns a new image containing the visualized matches
        """

        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
        vis[0:h1, 0:w1] = img1
        vis[0:h2, w1:] = img2

        for ((idx2, idx1), s) in zip(matches, status):
            # only process the match if the keypoint was successfully matched
            if s == 1:
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

    def crop_black(self, img):
        # code taken from https://stackoverflow.com/questions/39465812/how-to-crop-zero-edges-of-a-numpy-array
        # crops zero values from image
        # argwhere will give you the coordinates of every non-zero point
        true_points = np.argwhere(img)
        # take the smallest points and use them as the top left of your crop
        top_left = true_points.min(axis=0)
        # take the largest points and use them as the bottom right of your crop
        bottom_right = true_points.max(axis=0)
        out = img[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
              top_left[1]:bottom_right[1] + 1]  # inclusive
        return out

    def scale_to_width(self, img, width):
        # proportional scale of an image to a given width
        h, w = img.shape[:2]
        h_ = int(h / w * width)
        return cv2.resize(img, (width, h_))

    def stitch_to_panorama(self):

        # YOUR CODE HERE
        matchList = []
        baseImage = self.imagelist[0]
        panoramaImg = np.zeros((3000, baseImage.shape[1] * 2, 3), dtype=np.uint8)  ## TODO: adjust size
        panoramaImg[0:baseImage.shape[0], 0:baseImage.shape[1]] = baseImage

        # 1. create feature extraction
        sift = cv2.xfeatures2d.SIFT_create()

        # 2. detect and compute keypoints and descriptors for the first image
        kp, desc = sift.detectAndCompute(baseImage, None)

        # 3. loop through the remaining images and detect and compute keypoints + descriptors
        for img in self.imagelist[1:]:
            kp2, desc2 = sift.detectAndCompute(img, None)
            # 4. match features between the two images consecutive images and check if the result might be None.
            H, status, matches = self.match_keypoints(kp, kp2, desc, desc2)
            if not matches:
                # if not enough matches were found we can't stitch and we break here
                break
            else:
                matchImg = self.draw_matches(baseImage, img, kp, kp2, matches, status)
                matchList.append(self.scale_to_width(matchImg, 1000))
                warpedImg = cv2.warpPerspective(img, H, (2000, 2000), flags=cv2.WARP_INVERSE_MAP)
                panoramaImg[0:warpedImg.shape[0], 0:warpedImg.shape[1]] += warpedImg


                # The result contains matches and a status object that can be used to draw the matches.
                # Additionally (and more importantly it contains the transformation matrix (homography matrix)
                # commonly refered to as H. That can and should be used with cv2.warpPerspective to transform
                # consecutive images such that they fit together.
                # make sure the size of the new (warped) image is large enough to support the overlap

                # the resulting image might be too wide (lot of black areas on the right) because there is a
                # substantial overlap

                # 5. create a new image using draw_matches containing the visualized matches

        # 6. return the resulting stitched image

        panoramaImg = self.crop_black(panoramaImg)
        panoramaImg = self.scale_to_width(panoramaImg, 1000)
        return (matchList, panoramaImg)
