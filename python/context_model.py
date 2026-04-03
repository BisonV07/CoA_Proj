"""
Context modeler for wavelet coefficients.
Estimates the Golomb-Rice parameter k from causal (already-encoded) neighbors.
"""

import numpy as np


def estimate_k_map(subband: np.ndarray) -> np.ndarray:
    """
    For every coefficient in `subband`, estimate the Golomb-Rice parameter k
    using the mean absolute value of causal neighbors (top, left, top-left, top-right).
    Returns an int32 array of k values with the same shape as `subband`.
    """
    h, w = subband.shape
    k_map = np.zeros((h, w), dtype=np.int32)

    for r in range(h):
        for c in range(w):
            neighbors = []
            if r > 0:
                neighbors.append(abs(int(subband[r - 1, c])))       # top
            if c > 0:
                neighbors.append(abs(int(subband[r, c - 1])))       # left
            if r > 0 and c > 0:
                neighbors.append(abs(int(subband[r - 1, c - 1])))   # top-left
            if r > 0 and c < w - 1:
                neighbors.append(abs(int(subband[r - 1, c + 1])))   # top-right

            if neighbors:
                sigma = sum(neighbors) / len(neighbors)
                k_map[r, c] = max(0, int(np.floor(np.log2(sigma + 1))))
            else:
                k_map[r, c] = 0

    return k_map
