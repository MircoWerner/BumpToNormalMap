##################################
# BUMP to NORMAL MAP
##################################
# python bumptonormalmap.py <path to bump map> <strength>
#
# <path to bump map> : string -> path to the input image (the bump map, i.e. the height map)
#
# <strength> : float > 0 -> "strength" of the normal map
#                           results in smoother (strength -> 0) or sharper (strength -> \infty) features
#                           strength = 1 (recommended to start with)
#                           strength = 2 (more defined features)
#                           strength = 10 (really strong normal mapping effect...)
#                           just experiment a little bit :)
##################################
# Uses horizontal and vertical sobel filters (https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)
# to detect edges to determine the gradients in horizontal and vertical direction respectively.
#
# GX = [ -1 0 1] = [1]
#      [ -2 0 2]   [2] * [-1 0 1]
#      [ -1 0 1]   [1]
#
# GY = [ -1 -2 -1] = [-1]
#      [  0  0  0]   [ 0] * [1 2 1]
#      [  1  2  1]   [ 1]
#
# The sobel filter is separable which allows to compute two one-dimensional convolutions instead of one two-dimensional.
#
# The normals are computed from the gradients in horizontal (dx) and vertical direction (dy) as follows:
# normal = normalize(vec3(dx, dy, 1.0 / strength))
#
# Note that the normals are transformed from [-1,1] space to [0,1] space (normal * vec3(0.5) + vec3(0.5)).
# Remember to undo the transformation (vec3(2.0) * normal - vec3(1.0)) when reading the normals from the normal map.

import cv2
import numpy as np
import sys
import time


#################
# UTILITY
#################
def normalize(vec: np.array) -> np.array:
    # length = np.sqrt(np.dot(vec, vec))
    length = np.expand_dims(np.linalg.norm(vec, axis=-1), axis=-1)
    return vec / length


def normalToColor(vec: np.array) -> np.array:
    return np.array(
        [np.uint8((vec[2] * 0.5 + 0.5) * 255), np.uint8((vec[1] * 0.5 + 0.5) * 255),
         np.uint8((vec[0] * 0.5 + 0.5) * 255)])  # BGR


def pixel_value(img, dim_x: int, dim_y: int, x: int, y: int) -> float:
    x_clamped = min(max(x, 0), dim_x - 1)
    y_clamped = min(max(y, 0), dim_y - 1)
    return img[y_clamped][x_clamped][0] / 255.0


def storage_value(storage: np.array, dim_x: int, dim_y: int, x: int, y: int) -> float:
    x_clamped = min(max(x, 0), dim_x - 1)
    y_clamped = min(max(y, 0), dim_y - 1)
    return storage[y_clamped][x_clamped]


# https://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python/20929881#20929881
def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


##################################
# SOBEL (slow)
##################################
# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("arguments: <path to image> <strength>")
#         sys.exit()
#
#     if not is_float(sys.argv[2]):
#         print(sys.argv[2] + " is not a float")
#         sys.exit()
#
#     img = cv2.imread(sys.argv[1])
#     dim = img.shape
#     height = dim[0]
#     width = dim[1]
#
#     strength = float(sys.argv[2])
#     if strength <= 0.0:
#         print("strength has to be >0")
#         sys.exit()
#
#     targetImg = np.zeros((height, width, 3), np.uint8)
#
#     start = time.time()
#
#     for y in range(height):
#         for x in range(width):
#             if (y * width + x) % 10000 == 0:
#                 print("Converting... " + "{:.{}f}".format((y * width + x) / (width * height) * 100.0, 2) + "%")
#
#             mXmY = pixel_value(img, width, height, x - 1, y - 1)
#             mXoY = pixel_value(img, width, height, x - 1, y)
#             mXpY = pixel_value(img, width, height, x - 1, y + 1)
#             oXpY = pixel_value(img, width, height, x, y + 1)
#             pXpY = pixel_value(img, width, height, x + 1, y + 1)
#             pXoY = pixel_value(img, width, height, x + 1, y)
#             pXmY = pixel_value(img, width, height, x + 1, y - 1)
#             oXmY = pixel_value(img, width, height, x, y - 1)
#
#             dx = -mXmY - 2 * mXoY - mXpY + pXmY + 2 * pXoY + pXpY
#             dy = mXpY + 2 * oXpY + pXpY - mXmY - 2 * oXmY - pXmY
#
#             normal = normalize(np.array([dx, dy, 1.0 / strength]))
#
#             targetImg[y][x] = normalToColor(normal)
#
#     end = time.time()
#
#     cv2.imwrite("normal_map.png", targetImg)
#
#     print("Finished. Conversion took " + str(end - start) + "s.")

##################################
# SEPARABLE SOBEL (faster)
##################################
# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("arguments: <path to image> <strength>")
#         sys.exit()
#
#     if not is_float(sys.argv[2]):
#         print(sys.argv[2] + " is not a float")
#         sys.exit()
#
#     img = cv2.imread(sys.argv[1])
#     dim = img.shape
#     height = dim[0]
#     width = dim[1]
#
#     strength = float(sys.argv[2])
#     if strength <= 0.0:
#         print("strength has to be >0")
#         sys.exit()
#
#     targetImg = np.zeros((height, width, 3), np.uint8)
#
#     start = time.time()
#
#     tempHorizontalGX = np.zeros((height, width), float)
#     tempHorizontalGY = np.zeros((height, width), float)
#     for y in range(height):
#         for x in range(width):
#             if (y * width + x) % 10000 == 0:
#                 print(
#                     "Converting horizontal... " + "{:.{}f}".format((y * width + x) / (width * height) * 100.0, 2) + "%")
#
#             mXoY = pixel_value(img, width, height, x - 1, y)
#             oXoY = pixel_value(img, width, height, x, y)
#             pXoY = pixel_value(img, width, height, x + 1, y)
#
#             dx = -mXoY + pXoY
#             dy = mXoY + 2.0 * oXoY + pXoY
#
#             tempHorizontalGX[y][x] = dx
#             tempHorizontalGY[y][x] = dy
#
#     for y in range(height):
#         for x in range(width):
#             if (y * width + x) % 10000 == 0:
#                 print("Converting vertical... " + "{:.{}f}".format((y * width + x) / (width * height) * 100.0, 2) + "%")
#
#             oXpYGX = storage_value(tempHorizontalGX, width, height, x, y + 1)
#             oXoYGX = storage_value(tempHorizontalGX, width, height, x, y)
#             oXmYGX = storage_value(tempHorizontalGX, width, height, x, y - 1)
#
#             oXpYGY = storage_value(tempHorizontalGY, width, height, x, y + 1)
#             oXmYGY = storage_value(tempHorizontalGY, width, height, x, y - 1)
#
#             dx = oXmYGX + 2.0 * oXoYGX + oXpYGX
#             dy = -oXmYGY + oXpYGY
#
#             normal = normalize(np.array([dx, dy, 1.0 / strength]))
#
#             targetImg[y][x] = normalToColor(normal)
#
#     end = time.time()
#
#     cv2.imwrite("normal_map.png", targetImg)
#
#     print("Finished. Conversion took " + str(end - start) + "s.")

##################################
# OPENCV's SOBEL (fastest)
##################################
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("arguments: <path to image> <strength> <mode>")
        sys.exit()

    if not is_float(sys.argv[2]):
        print(sys.argv[2] + " is not a float")
        sys.exit()

    if sys.argv[3] not in ["png", "exr"]:
        print("mode has to be either 'png' or 'exr'")
        sys.exit()

    img = cv2.imread(sys.argv[1])
    dim = img.shape
    height = dim[0]
    width = dim[1]

    strength = float(sys.argv[2])
    if strength <= 0.0:
        print("strength has to be >0")
        sys.exit()

    start = time.time()

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_REPLICATE)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_REPLICATE)
    print("Sobel filters applied...")

    dx = grad_x[:, :, 0] / 255.0
    dy = grad_y[:, :, 0] / 255.0
    inv_strength = np.full_like(dx, 1.0 / strength)
    normals = normalize(np.stack([inv_strength, dy, dx], axis=-1)) # BGR format
    colors = normals * 0.5 + 0.5

    if sys.argv[3] == "png":
        colors = np.uint8(colors * 255)
        cv2.imwrite("normal_map.png", colors)
    else:
        cv2.imwrite("normal_map.exr", colors.astype(np.float32))

    end = time.time()
    print("Finished. Conversion took " + str(end - start) + "s.")
