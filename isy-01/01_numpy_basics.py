import numpy as np


# A1. Numpy and Linear Algebra

print(
    "a) (*) Erzeugen Sie einen Vektor mit Nullen der Länge 10 (10 Elemente) und setzen den Wert des 5.Elementes auf eine 1.")

vec_a = np.zeros(10)
vec_a[4] = 1
print(vec_a)

print("b) (*) Erzeugen Sie einen Vektor mit Ganzahl-Werten von 10 bis 49 (geht in einer Zeile).")

vec_b = np.arange(10, 50)
print(vec_b)

print("c) (*) Drehen Sie die Werte des Vektors um (geht in einer Zeile).")

print(vec_b[::-1])

print("d) (*) Erzeugen Sie eine 4x4 Matrix mit den Werte 0 bis 15 (links oben rechts unten).")

print(np.arange(0, 16).reshape(4, 4))

print(
    "e) (*) Erzeuge eine 8x8 Matrix mit Zufallswerte und finde deren Maximum und Minimum und normalisieren Sie die Werte (sodass alle Werte zwischen 0 und 1 liegen - ein Wert wird 1 (max) sein und einer 0 (min)).")

mat_e = np.random.randint(0, 255, size=(8, 8))
print('Zufallsmatrix:\n')
print(mat_e)
print('\nmin:', mat_e.min(), 'max:', mat_e.max())
print("\nNormalisiert:\n")
mat_e_norm = (mat_e - mat_e.min()) / mat_e.max()
print(mat_e_norm)
print('\nmin:', mat_e_norm.min(), 'max:', mat_e_norm.max())

print("f) (*) Multiplizieren Sie eine 4x3 Matrix mit einer 3x2 Matrix")

mat4x3 = np.arange(0, 12).reshape((4, 3))
mat3x2 = np.arange(0, 6).reshape((3, 2))
print('4x3 Matrix:\n', mat4x3)
print('\n3x2 Matrix:\n', mat3x2)
print('\n', np.dot(mat4x3, mat3x2))

print(
    "g) (*) Erzeugen Sie ein 1D Array mit den Werte von 0 bis 20 und negieren Sie Werte zwischen 8 und 16 nachträglich.")

vec_g = np.arange(0, 21)
for n, val in enumerate(vec_g):
    if val >= 8 and val <= 16:
        vec_g[n] = -val
print(vec_g)

print("h) (*) Summieren Sie alle Werte in einem Array.")

vec_h = np.arange(1, 6)
print('sum of', vec_h, 'is', vec_h.sum())

print("i) (** ) Erzeugen Sie eine 5x5 Matrix und geben Sie jeweils die geraden und die ungeraden Zeile aus.")

mat_i = np.arange(0, 25).reshape((5, 5))
print(mat_i)
print('\nGerade Zeilen\n')
print(mat_i[0::2])
print('\nUngerade Zeilen\n')
print(mat_i[1::2])

print(
    "j) (** ) Erzeugen Sie eine Matrix M der Größe 4x3 und einen Vektor v mit Länge 3. Multiplizieren Sie jeden Spalteneintrag aus v mit der kompletten Spalte aus M. Schauen Sie sich dafür an, was Broadcasting in Numpy bedeutet.")

M = np.arange(0, 12).reshape(4, 3)
v = np.arange(2, 5)
print(M, '\n------\n')
print(v, '\n------\n')
print(M * v, '\n------\n')

print(
    "k) (** ) Erzeugen Sie einen Zufallsmatrix der Größe 10x2, die Sie als Kartesische Koordinaten interpretieren können ([[x0, y0],[x1, y1],[x2, y2]]). Konvertieren Sie diese in Polarkoordinaten https://de.wikipedia.org/wiki/Polarkoordinaten.")


def cart2pol(x, y):
    # https://www.w3resource.com/python-exercises/numpy/python-numpy-random-exercise-14.php
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)  # arctan2 automatically chooses the right quadrant
    return np.array([r, theta])


mat_k_cart = np.random.randint(0, 255, size=(10, 2))
mat_k_pol = np.empty_like(mat_k_cart, dtype=np.float32)

for n, (x, y) in enumerate(mat_k_cart[:]):
    mat_k_pol[n] = cart2pol(x, y)

print(mat_k_cart)
print(mat_k_pol)

print(
    "l) (***) Implementieren Sie zwei Funktionen, die das Skalarprodukt und die Vektorlänge für Vek- toren beliebiger Länge berechnen. Nutzen Sie dabei NICHT die gegebenen Funktionen von NumPy. Testen Sie Ihre Funktionen mit den gegebenen Vektoren:")

# TODO: ist es relevant dass das Spaltenvektoren sind?

v1 = np.array([1, 2, 3, 4, 5])
v2 = np.array([-1, 9, 5, 3, 1])


def dotproduct(vec1, vec2):
    product = 0.
    if vec1.shape != vec2.shape:
        return None
    for x1, x2 in zip(vec1, vec2):
        product += x1 * x2
    return product


def magnitude(vec):
    return np.sqrt(dotproduct(vec, vec))


print('Skalarprodukt von', v1, v2, '=', dotproduct(v1, v2))
print('Länge von', v1, magnitude(v1))
print('Länge von', v2, magnitude(v2))

print(
    "m) (***) Berechnen Sie (v0T v1)Mv0 unter der Nutzung von NumPy Operationen. Achten Sie darauf, dass hier v0,v1 Spaltenvektoren gegeben sind. v0T ist also ein Zeilenvektor.")

# Should result in [3,9,15,2]T

M_m = np.asmatrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 2, 2]]))
v0_m = np.array([[1], [1], [0]])
v1_m = np.array([[-1], [2], [5]])

# (np.dot(v0_m.T, v1_m)) leads to a single element two-dimensional array.
#  Broadcasting does not help there. It needs to get casted to a scalar because mp.dot or other matrix
#  multiplication featues do not recognize an single element matrix as a scalar.
print(np.asscalar(np.dot(v0_m.T, v1_m)) * M_m * v0_m)