"""
Reversible YCoCg-R color space transform.
Integer-only arithmetic (adds + shifts), perfectly lossless.
"""

import numpy as np


def rgb_to_ycocg_r(r: np.ndarray, g: np.ndarray, b: np.ndarray):
    """Forward transform: RGB -> Y, Co, Cg (all int32)."""
    r = r.astype(np.int32)
    g = g.astype(np.int32)
    b = b.astype(np.int32)

    co = r - b
    t  = b + (co >> 1)
    cg = g - t
    y  = t + (cg >> 1)
    return y, co, cg


def ycocg_r_to_rgb(y: np.ndarray, co: np.ndarray, cg: np.ndarray):
    """Inverse transform: Y, Co, Cg -> RGB (all int32)."""
    y  = y.astype(np.int32)
    co = co.astype(np.int32)
    cg = cg.astype(np.int32)

    t = y  - (cg >> 1)
    g = cg + t
    b = t  - (co >> 1)
    r = co + b
    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)
