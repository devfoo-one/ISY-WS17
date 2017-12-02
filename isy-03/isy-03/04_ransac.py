import random

import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RansacPointGenerator:
    """generates a set points - linear distributed + a set of outliers"""
    def __init__(self, numpointsInlier, numpointsOutlier):
        self.numpointsInlier = numpointsInlier
        self.numpointsOutlier = numpointsOutlier
        self.points = []

        pure_x = np.linspace(0, 1, numpointsInlier)
        pure_y = np.linspace(0, 1, numpointsInlier)
        noise_x = np.random.normal(0, 0.025, numpointsInlier)
        noise_y = np.random.normal(0, 0.025, numpointsInlier)

        outlier_x = np.random.random_sample((numpointsOutlier,))
        outlier_y = np.random.random_sample((numpointsOutlier,))

        points_x = pure_x + noise_x
        points_y = pure_y + noise_y
        points_x = np.append(points_x, outlier_x)
        points_y = np.append(points_y, outlier_y)

        self.points = np.array([points_x, points_y])

class Line:
    """helper class"""
    def __init__(self, a, b):
        # y = mx + b
        self.m = a
        self.b = b


class Ransac:
    """RANSAC class. """
    def __init__(self, points, threshold):
        self.points = points
        self.threshold = threshold
        self.best_model = Line(1, 0)
        self.best_inliers = []
        self.best_score   = 1000000000
        self.current_inliers = []
        self.current_model   = Line(1, 0)
        self.num_iterations  = int(self.estimate_num_iterations(0.99, 0.5, 2))
        self.iteration_counter = 0


    def estimate_num_iterations(self, ransacProbability, outlierRatio, sampleSize):
        """
        Helper function to generate a number of generations that depends on the probability
        to pick a certain set of inliers vs. outliers.
        See https://de.wikipedia.org/wiki/RANSAC-Algorithmus for more information

        :param ransacProbability: std value would be 0.99 [0..1]
        :param outlierRatio: how many outliers are allowed, 0.3-0.5 [0..1]
        :param sampleSize: 2 points for a line
        :return:
        """
        return math.ceil(math.log(1-ransacProbability) / math.log(1-math.pow(1-outlierRatio, sampleSize)))

    def estimate_error(self, p, line):
        """
        Compute the distance of a point p to a line y=mx+b
        :param p: Point
        :param line: Line y=mx+b
        :return:
        """
        return math.fabs(line.m * p[0] - p[1] + line.b) / math.sqrt(1 + line.m * line.m)


    def step(self, iter):
        """
        Run the ith step in the algorithm. Collects self.currentInlier for each step.
        Sets if score < self.bestScore
        self.bestModel = line
        self.bestInliers = self.currentInlier
        self.bestScore = score

        :param iter: i-th number of iteration
        :return:
        """
        self.current_inliers = []
        score = 0
        idx = 0
        p0 = np.zeros((2,1))
        p1 = np.zeros((2,1))
        # sample two random points from point set
        while p0[0] == p1[0] and p0[1] == p1[1]:  # make sure that p0 != p1
            i0 = np.random.randint(0, self.points.shape[1])
            i1 = np.random.randint(0, self.points.shape[1])
            p0 = self.points[0:,i0]
            p1 = self.points[0:,i1]


        # compute line parameters m / b and create new line

        m = (p1[1] - p0[1]) / (p1[0] - p0[0]) # m = delta_y / delta_x
        b = p0[1] - m * p0[0] # b = y - mx
        line = Line(m, b)

        # loop over all points
        for index in range(0, self.points.shape[1]):
        # for index, p in enumerate(self.points):  # TODO: WRONG ITERATION
            if index == i0 or index == i1:  # make sure that p0 or p1 will not be used
                continue
            p = self.points[0:,index]
            error = self.estimate_error(p, line)
            if error < self.threshold:
                self.current_inliers.append(p)
                score -= error  # TODO: FIX SCORING
                print(error)
            else:
                score += error / self.threshold
            # TODO: WHAT SCORE? WHY NOT ONLY LEN(INLIERS)?
            # compute error of all points and add to inliers of # TODO: ???? WHAT?
            # err smaller than threshold update score, otherwise add error/threshold to score

        # if score < self.bestScore: update the best model/inliers/score
        if score < self.best_score:
            self.best_score = score
            self.best_model = Line(m, b)
            self.best_inliers = self.current_inliers
        # please do look at resources in the internet :)

        print(iter, "  :::::::::: bestscore: ", self.best_score, " bestModel: ", self.best_model.m, self.best_model.b)

    def run(self):
        """
        run RANSAC for a number of iterations
        :return:
        """
        for i in range(0, self.num_iterations):
            self.step(i)


rpg = RansacPointGenerator(100,200)
# rpg = RansacPointGenerator(100,45)
print(rpg.points)

ransac = Ransac(rpg.points, 0.05)
ransac.run()

# print rpg.points.shape[1]
plt.plot(rpg.points[0,:], rpg.points[1,:], 'ro')
m = ransac.best_model.m
b = ransac.best_model.b
plt.plot([0, 1], [m*0 + b, m*1+b], color='k', linestyle='-', linewidth=2)
# #
plt.axis([0, 1, 0, 1])
plt.show()

