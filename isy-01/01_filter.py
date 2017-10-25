import numpy as np
import cv2


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
    return k / np.sum(k)


def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbritrary sized convolution here
    # Hint: you need the kernelsize
    offset_x = int(kernel.shape[1] / 2)
    offset_y = int(kernel.shape[0] / 2)
    newimg = np.zeros(img.shape)

    for x in range(offset_x, img.shape[1] - offset_x):
        for y in range(offset_y, img.shape[0] - offset_y):
            area = img[y - offset_y:y + offset_y + 1, x - offset_x: x + offset_x + 1]
            intensity = sum((area * kernel).ravel())
            newimg.itemset(y, x, intensity)

    return newimg


if __name__ == "__main__":
    # 1. load image in grayscale
    img = cv2.imread('images/Lenna.png', cv2.IMREAD_GRAYSCALE)

    # 2. convert image to 0-1 image (see im2double)
    img_norm = im2double(img)

    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 3 .use image kernels on normalized image

    sobel_x = convolution_2d(img_norm, sobelmask_x)
    sobel_y = convolution_2d(img_norm, sobelmask_y)

    # 4. compute magnitude of gradients

    # Show resulting images
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    # cv2.imshow("mog", mog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
