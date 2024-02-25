# -*- coding: utf-8 -*-
"""
XYB color space.

https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf
"""

import numpy as np


BIAS = 0.00379307325527544933
BIAS_CBRT = np.cbrt(BIAS)

# constans from opsin_params.h
LRGB_TO_LMS = np.asarray([
    [0.3,  0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.24342268924547819, 0.20476744424496821, 0.55180986650955360]
], dtype=np.float32)

LMS_TO_LRGB = np.asarray([
    [11.031566901960783, -9.866943921568629, -0.16462299647058826],
    [-3.254147380392157,  4.418770392156863, -0.16462299647058826],
    [-3.6588512862745097, 2.7129230470588235, 1.9459282392156863 ]
], dtype=np.float32)

# https://twitter.com/jonsneyers/status/1605321352143331328
# @jonsneyers Feb 22
# Yes, the default is to just subtract Y from B. In general there are locally
# signaled float multipliers to subtract some multiple of Y from X and some
# other multiple from B. But this is the baseline, making X=B=0 grayscale.
# ----
# We adjust the matrix to subtract Y from B match this statement.
XYB_LMS_TO_XYB = np.asarray([
    [0.5, -0.5, 0.0],
    [0.5,  0.5, 0.0],
    [0.0, -1.0, 1.0],
], dtype=np.float32)

XYB_TO_XYB_LMS = np.asarray([
    [ 1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [-1.0, 1.0, 1.0]
], dtype=np.float32)


def rgb_to_xyb(rgb):
    """Linear sRGB to XYB."""
    rgb = np.asarray(rgb, dtype=np.float32)

    lms = np.inner(rgb, LRGB_TO_LMS)
    lms_gamma = np.cbrt(lms + BIAS) - BIAS_CBRT
    return np.inner(lms_gamma, XYB_LMS_TO_XYB)


def xyb_to_rgb(xyb):
    """XYB to linear sRGB."""

    # # This cleans up the round trip on black.
    # if not any(xyb):
    #     return [0.0] * 3
    xyb = np.asarray(xyb, dtype=np.float32)

    xyb_lms = np.inner(xyb,  XYB_TO_XYB_LMS)
    lms_mix = (xyb_lms + BIAS_CBRT) ** 3 - BIAS
    print(lms_mix)
    return np.inner(lms_mix, LMS_TO_LRGB)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    rgb = [
        [[0, 0, 0], [0.12, 0, 0.5]],
        [[1, 1, 1], [0.11, 0.32, 0.9]]
    ]
    print(rgb)

    xyb = rgb_to_xyb(rgb)
    print(xyb)

    rgb_conv = xyb_to_rgb(xyb)
    print(rgb_conv)
