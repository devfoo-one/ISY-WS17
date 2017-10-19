import numpy as np
import cv2
import math
import time


######################################################################
# IMPORTANT: Please make yourself comfortable with numpy and python:
# e.g. https://www.stavros.io/tutorials/python/
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Note: data types are important for numpy and opencv
# most of the time we'll use np.float32 as arrays
# e.g. np.float32([0.1,0.1]) equal np.array([1, 2, 3], dtype='f')

print("a) (*) Erzeugen Sie einen Vektor mit Nullen der Länge 10 (10 Elemente) und setzen den Wert des 5.Elementes auf eine 1.")

vec_a = np.zeros(10)
vec_a[4] = 1
print(vec_a)

print("b) (*) Erzeugen Sie einen Vektor mit Ganzahl-Werten von 10 bis 49 (geht in einer Zeile).")

vec_b = np.arange(10,50)
print(vec_b)

print("c) (*) Drehen Sie die Werte des Vektors um (geht in einer Zeile).")

print(vec_b[::-1])

print("d) (*) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 bis 15 (links oben rechts unten).")
print("e) (*) Erzeuge eine 8x8 Matrix mit Zufallswerte und finde deren Maximum und Minimum und normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen - ein Wert wird 1 (max) sein und einer 0 (min)).")
print("f) (*) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix")
print("g) (*) Erzeugen Sie ein 1D Array mit den Werte von 0 bis 20 und negieren Sie Werte zwischen 8 und 16 nachträglich.")
print("h) (*) Summieren Sie alle Werte in einem Array.")
print("i) (** ) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile aus.")
print("j) (** ) Erzeugen Sie eine Matrix M der Größe 4x3 und einen Vektor v mit Länge 3. Multiplizieren Sie jeden Spalteneintrag aus v mit der kompletten Spalte aus M. Schauen Sie sich dafür an, was Broadcasting in Numpy bedeutet.")
print("k) (** ) Erzeugen Sie einen Zufallsmatrix der Größe 10x2, die Sie als Kartesische Koordinaten interpretieren können ([[x0, y0],[x1, y1],[x2, y2]]). Konvertieren Sie diese in Polarkoordinaten https://de.wikipedia.org/wiki/Polarkoordinaten.")
print("l) (***) Implementieren Sie zwei Funktionen, die das Skalarprodukt und die Vektorlänge für Vek- toren beliebiger Länge berechnen. Nutzen Sie dabei NICHT die gegebenen Funktionen von NumPy. Testen Sie Ihre Funktionen mit den gegebenen Vektoren:")
print("m) (***) Berechnen Sie (v0T v1)Mv0 unter der Nutzung von NumPy Operationen. Achten Sie darauf, dass hier v0,v1 Spaltenvektoren gegeben sind. v0T ist also ein Zeilenvektor.")


print(np.float32([0.1,0.1]))

######################################################################
# A2. OpenCV and Transformation and Computer Vision Basic

# (1) read in the image Lenna.png using opencv in gray scale and in color
# and display it NEXT to each other (see result image)
# Note here: the final image displayed must have 3 color channels
#            So you need to copy the gray image values in the color channels
#            of a new image. You can get the size (shape) of an image with rows, cols = img.shape[:2]

# why Lenna? https://de.wikipedia.org/wiki/Lena_(Testbild)

# (2) Now shift both images by half (translation in x) it rotate the colored image by 30 degrees using OpenCV transformation functions
# + do one of the operations on keypress (t - translate, r - rotate, 'q' - quit using cv::warpAffine
# http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
# Tip: you need to define a transformation Matrix M
# see result image




