##################################
# BUMP to NORMAL MAP
##################################
# python bumptonormalmap.py <path to bump map> <strength> <output format>
#
# <path to bump map> : string -> path to the input image (the bump map, i.e. the height map)
#
# <strength> : float > 0 -> "strength" of the normal map
#                           results in smoother (strength -> 0) or sharper (strength -> \infty) features
#                           strength = 1 (recommended to start with)
#                           strength = 2 (more defined features)
#                           strength = 10 (really strong normal mapping effect...)
#                           just experiment a little bit :)
#
# <output format> : string -> "png" or "exr"; output image format. Use "exr" for higher precision.
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
import argparse
import platform
import sys
import time

missing_packages = {}
try:
    import cv2
except ImportError:
    missing_packages["cv2"] = "python3-opencv"
try:
    import numpy as np
except ImportError:
    missing_packages["numpy"] = "python3-numpy"
if missing_packages:
    sys.stderr.write(
        "ImportError: You must install the missing dependencies(s) {}"
        " such as from pip/conda"
        .format(list(missing_packages.keys())))
    if platform.system() != "Windows":
        sys.stderr.write(
            " or precompiled versions of compilable modules"
            " if your distro provides them"
            " (recommended in the case of such modules)")
    print(":", file=sys.stderr)
    sys.stderr.write("   ")
    for value in missing_packages.values():
        sys.stderr.write(" {}".format(value))
    print("", file=sys.stderr)
    sys.exit(1)


def normalize(vec: np.array) -> np.array:
    length = np.expand_dims(np.linalg.norm(vec, axis=-1), axis=-1)
    return vec / length


def main():
    parser = argparse.ArgumentParser(description='Convert bump/height map to normal map')
    parser.add_argument('path', type=str, help='path to the input image (the bump map, i.e. the height map)')
    parser.add_argument('strength', type=float, help='strength of the normal map. results in smoother (strength -> 0) or sharper (strength -> \infty) features')
    parser.add_argument('output_format', type=str, choices=["png", "exr"], help='output image format')
    args = parser.parse_args()

    bump_to_normal(args.path, strength=args.strength,
                   output_format=args.output_format)
    return 0


def bump_to_normal(path, strength=1.0, output_format="png"):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        print("Could not read image")
        sys.exit()

    if img.dtype == np.uint8:
        # convert integer images to floating point
        img = img.astype(np.float32) / 255.0

    if len(img.shape) == 2:
        # convert grayscale to rgb
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # dim = img.shape
    # height = dim[0]
    # width = dim[1]

    if strength <= 0.0:
        print("strength has to be >0")
        sys.exit()

    start = time.time()

    scale = 1
    delta = 0
    ddepth = cv2.CV_64F

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta,
                       borderType=cv2.BORDER_REPLICATE)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta,
                       borderType=cv2.BORDER_REPLICATE)
    print("Sobel filters applied")

    dx = grad_x[:, :, 0]
    dy = grad_y[:, :, 0]
    inv_strength = np.full_like(dx, 1.0 / strength)
    normals = normalize(np.stack([inv_strength, dy, dx], axis=-1)) # BGR format
    colors = normals * 0.5 + 0.5

    if output_format == "png":
        colors = np.uint8(colors * 255)
        cv2.imwrite("normal_map.png", colors)
    else:
        cv2.imwrite("normal_map.exr", colors.astype(np.float32))

    end = time.time()
    print("Finished. Conversion took " + str(end - start) + "s.")


if __name__ == '__main__':
    sys.exit(main())