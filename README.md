# Bump to Normal Map

Simple Python script to convert a bump map to a normal map.
*Not optimized for performance.*

## Requirements
- Python 3
- numpy
- opencv-python 

## Usage / Run
`python bumptonormalmap.py <path to bump map> <strength> <output format>`

```python
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
```

## How does it work?
```python
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
# The sobel filter is separable which allows to compute two one-dimensional convolutions instead of one two-dimensional 
# (accelerates computation).
#
# The normals are computed from the gradients in horizontal (dx) and vertical direction (dy) as follows:
# normal = normalize(vec3(dx, dy, 1.0 / strength))
#
# Note that the normals are transformed from [-1,1] space to [0,1] space (normal * vec3(0.5) + vec3(0.5)).
# Remember to undo the transformation (vec3(2.0) * normal - vec3(1.0)) when reading the normals from the normal map.
```

## Example
*The bump map is taken from the `LPS Head` model. The complete model including the bump map can be downloaded from https://casual-effects.com/data/.*

`python bumptonormalmap.py example/bump_map.png 2 png`

![img bump map](https://github.com/MircoWerner/BumpToNormalMap/blob/main/example/bump_map.png?raw=true)

![img normal map](https://github.com/MircoWerner/BumpToNormalMap/blob/main/example/normal_map.png?raw=true)