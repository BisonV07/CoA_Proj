"""
CDF 5/3 integer wavelet transform using the lifting scheme.
Fully vectorized with NumPy — operates entirely in int32, perfectly reversible.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1-D forward / inverse  (vectorized)
# ---------------------------------------------------------------------------

def _cdf53_forward_1d(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n < 2:
        return signal.copy(), np.array([], dtype=np.int32)

    even = signal[0::2].copy()
    odd  = signal[1::2].copy()
    n_even = len(even)
    n_odd  = len(odd)

    # Predict: d[i] = odd[i] - floor((even[i] + even[min(i+1, n_even-1)]) / 2)
    even_right = np.empty(n_odd, dtype=np.int32)
    even_right[:n_odd - 1] = even[1:n_odd]
    even_right[n_odd - 1]  = even[min(n_odd, n_even - 1)]
    odd -= (even[:n_odd] + even_right) >> 1

    # Update: s[i] = even[i] + floor((d[max(i-1,0)] + d[min(i, n_odd-1)] + 2) / 4)
    odd_left = np.empty(n_even, dtype=np.int32)
    odd_left[0]  = odd[0]
    odd_left[1:] = odd[:n_even - 1]
    odd_right = np.empty(n_even, dtype=np.int32)
    odd_right[:n_even] = odd[np.minimum(np.arange(n_even), n_odd - 1)]
    even += (odd_left + odd_right + 2) >> 2

    return even, odd


def _cdf53_inverse_1d(even: np.ndarray, odd: np.ndarray) -> np.ndarray:
    n_even = len(even)
    n_odd  = len(odd)
    if n_odd == 0:
        return even.copy()

    even = even.copy()
    odd  = odd.copy()

    # Undo update
    odd_left = np.empty(n_even, dtype=np.int32)
    odd_left[0]  = odd[0]
    odd_left[1:] = odd[:n_even - 1]
    odd_right = np.empty(n_even, dtype=np.int32)
    odd_right[:n_even] = odd[np.minimum(np.arange(n_even), n_odd - 1)]
    even -= (odd_left + odd_right + 2) >> 2

    # Undo predict
    even_right = np.empty(n_odd, dtype=np.int32)
    even_right[:n_odd - 1] = even[1:n_odd]
    even_right[n_odd - 1]  = even[min(n_odd, n_even - 1)]
    odd += (even[:n_odd] + even_right) >> 1

    n = n_even + n_odd
    out = np.empty(n, dtype=np.int32)
    out[0::2] = even
    out[1::2] = odd
    return out


# ---------------------------------------------------------------------------
# 2-D separable forward / inverse  (rows then columns, batch-vectorized)
# ---------------------------------------------------------------------------

def _forward_rows(img: np.ndarray):
    """Apply 1D forward wavelet to every row. Returns (low, high) column-split."""
    h, w = img.shape
    wl = (w + 1) // 2
    wh = w // 2
    low  = np.empty((h, wl), dtype=np.int32)
    high = np.empty((h, wh), dtype=np.int32)
    for r in range(h):
        low[r], high[r] = _cdf53_forward_1d(img[r])
    return low, high


def _forward_cols(arr: np.ndarray):
    """Apply 1D forward wavelet to every column. Returns (low, high) row-split."""
    h, w = arr.shape
    hl = (h + 1) // 2
    hh = h // 2
    low  = np.empty((hl, w), dtype=np.int32)
    high = np.empty((hh, w), dtype=np.int32)
    for c in range(w):
        low[:, c], high[:, c] = _cdf53_forward_1d(arr[:, c])
    return low, high


def _inverse_rows(low: np.ndarray, high: np.ndarray):
    h = low.shape[0]
    w = low.shape[1] + high.shape[1]
    out = np.empty((h, w), dtype=np.int32)
    for r in range(h):
        out[r] = _cdf53_inverse_1d(low[r], high[r])
    return out


def _inverse_cols(low: np.ndarray, high: np.ndarray):
    h = low.shape[0] + high.shape[0]
    w = low.shape[1]
    out = np.empty((h, w), dtype=np.int32)
    for c in range(w):
        out[:, c] = _cdf53_inverse_1d(low[:, c], high[:, c])
    return out


def cdf53_forward_2d(image: np.ndarray):
    """One level of 2-D CDF 5/3. Returns (LL, LH, HL, HH)."""
    img = image.astype(np.int32)
    row_low, row_high = _forward_rows(img)
    LL, LH = _forward_cols(row_low)
    HL, HH = _forward_cols(row_high)
    return LL, LH, HL, HH


def cdf53_inverse_2d(LL, LH, HL, HH):
    """Invert one level of 2-D CDF 5/3."""
    row_low  = _inverse_cols(LL, LH)
    row_high = _inverse_cols(HL, HH)
    return _inverse_rows(row_low, row_high)


# ---------------------------------------------------------------------------
# Multi-level decomposition
# ---------------------------------------------------------------------------

def multilevel_forward(image: np.ndarray, levels: int = 3):
    """
    Multi-level 2-D CDF 5/3.
    Returns (final_LL, [(LH, HL, HH), ...]) — finest level first.
    """
    subbands = []
    current = image.astype(np.int32)
    for _ in range(levels):
        LL, LH, HL, HH = cdf53_forward_2d(current)
        subbands.append((LH, HL, HH))
        current = LL
    return current, subbands


def multilevel_inverse(final_LL, subbands):
    """Reconstruct from multi-level decomposition."""
    current = final_LL
    for LH, HL, HH in reversed(subbands):
        current = cdf53_inverse_2d(current, LH, HL, HH)
    return current
