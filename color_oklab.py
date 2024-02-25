# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 17:33:26 2022

Oklab, Oklch color space conversion

Oklab: https://bottosson.github.io/posts/oklab/

@author: dof
"""

import numpy as np


RGB_TO_LMS = np.asarray([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], dtype=np.float32)

LMS_TO_OKLAB = np.asarray([
    [0.2104542553, +0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, +0.4505937099],
    [0.0259040371, +0.7827717662, -0.8086757660],
], dtype=np.float32)

OKLAB_TO_LMS = np.asarray([
    [1.0, +0.3963377774, +0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
], dtype=np.float32)

LMS_TO_RGB = np.asarray([
    [+4.0767416621, -3.3077115913, +0.2309699292],
    [-1.2684380046, +2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, +1.7076147010],
], dtype=np.float32)


def linear_srgb_to_oklab(c):
    c = np.asarray(c, dtype=np.float32)

    lms = np.inner(c, RGB_TO_LMS)
    lms_ = np.cbrt(lms)
    return np.inner(lms_, LMS_TO_OKLAB)


def oklab_to_linear_srgb(c):
    c = np.asarray(c, dtype=np.float32)

    lms_ = np.inner(c, OKLAB_TO_LMS)
    lms = lms_ * lms_ * lms_
    return np.inner(lms, LMS_TO_RGB)


def linear_srgb_to_oklch(rgb):
    lab = linear_srgb_to_oklab(rgb)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360

    return np.stack([L, C, h], axis=-1)


def oklch_to_linear_srgb(lch):
    lch = np.asarray(lch, dtype=np.float32)

    L = lch[..., 0]
    C = lch[..., 1]
    h = lch[..., 2]

    h_ = np.radians(h)
    Lab = np.stack([L, C * np.cos(h_), C * np.sin(h_)], axis=-1)

    return oklab_to_linear_srgb(Lab)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from color_srgb import srgb_to_lin_srgb, lin_srgb_to_srgb

    srgb = [[0.2, 0.1, 0], [0.1, 0.2, 0.3], [0.8, 1, 1], [0.98, 0.35, 0.12]]
    plt.imshow([srgb])

    # sRGB to Oklch
    lrgb = srgb_to_lin_srgb(srgb)
    oklch = linear_srgb_to_oklch(lrgb)
    print(oklch)

    # Oklch to srgb
    lrgb2 = oklch_to_linear_srgb(oklch)
    srgb2 = lin_srgb_to_srgb(lrgb2)
    plt.imshow([srgb2])
