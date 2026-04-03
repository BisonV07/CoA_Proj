"""
Golomb-Rice entropy encoder / decoder with enhanced adaptive context.

Enhancements over baseline:
  1. MED (Median Edge Detector) based DPCM for LL subbands
  2. 6-neighbor spatial context with gradient weighting
  3. Inter-scale parent DETAIL context (wavelet-tree based)
  4. Escape coding for large values

Speed optimizations (bit-identical output):
  - Encoder: vectorized NumPy pre-computation of k map
  - Decoder: fully inlined context + k + value decoding
  - BitWriter/BitReader: multi-bit batch operations
"""

import numpy as np


ESCAPE_QUOTIENT = 12
FIXED_CODE_BITS = 16


# ---------------------------------------------------------------------------
# Signed <-> unsigned mapping
# ---------------------------------------------------------------------------

def _signed_to_unsigned(val):
    if isinstance(val, np.ndarray):
        return np.where(val >= 0, 2 * val, -2 * val - 1).astype(np.int64)
    return 2 * val if val >= 0 else -2 * val - 1


def _unsigned_to_signed(uval):
    if isinstance(uval, np.ndarray):
        return np.where(uval % 2 == 0, uval // 2, -(uval + 1) // 2).astype(np.int32)
    return uval // 2 if uval % 2 == 0 else -(uval + 1) // 2


# ---------------------------------------------------------------------------
# MED predictor (from JPEG-LS)
# ---------------------------------------------------------------------------

def _med_predict(top: int, left: int, top_left: int) -> int:
    if top_left >= max(top, left):
        return min(top, left)
    elif top_left <= min(top, left):
        return max(top, left)
    else:
        return top + left - top_left


# ---------------------------------------------------------------------------
# DPCM forward / inverse using MED predictor
# ---------------------------------------------------------------------------

def dpcm_forward(subband: np.ndarray) -> np.ndarray:
    h, w = subband.shape
    residuals = np.zeros_like(subband, dtype=np.int32)
    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                pred = 0
            elif r == 0:
                pred = int(subband[r, c - 1])
            elif c == 0:
                pred = int(subband[r - 1, c])
            else:
                pred = _med_predict(
                    int(subband[r - 1, c]),
                    int(subband[r, c - 1]),
                    int(subband[r - 1, c - 1])
                )
            residuals[r, c] = int(subband[r, c]) - pred
    return residuals


def dpcm_inverse(residuals: np.ndarray) -> np.ndarray:
    h, w = residuals.shape
    subband = np.zeros_like(residuals, dtype=np.int32)
    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                pred = 0
            elif r == 0:
                pred = int(subband[r, c - 1])
            elif c == 0:
                pred = int(subband[r - 1, c])
            else:
                pred = _med_predict(
                    int(subband[r - 1, c]),
                    int(subband[r, c - 1]),
                    int(subband[r - 1, c - 1])
                )
            subband[r, c] = int(residuals[r, c]) + pred
    return subband


# ---------------------------------------------------------------------------
# Vectorized k-map pre-computation (encoder only)
# ---------------------------------------------------------------------------

def _precompute_spatial_context(arr: np.ndarray):
    h, w = arr.shape
    a = np.abs(arr.astype(np.int64))

    abs_sum = np.zeros((h, w), dtype=np.int64)
    count = np.zeros((h, w), dtype=np.int32)

    abs_sum[1:, :] += 2 * a[:-1, :]
    count[1:, :] += 2
    abs_sum[:, 1:] += 2 * a[:, :-1]
    count[:, 1:] += 2
    abs_sum[1:, 1:] += a[:-1, :-1]
    count[1:, 1:] += 1
    abs_sum[1:, :-1] += a[:-1, 1:]
    count[1:, :-1] += 1
    abs_sum[2:, :] += a[:-2, :]
    count[2:, :] += 1
    abs_sum[:, 2:] += a[:, :-2]
    count[:, 2:] += 1

    return abs_sum, count


def _k_from_context(abs_sum, count):
    k_map = np.zeros_like(count, dtype=np.int32)
    mask = count > 0
    sigma = np.zeros_like(abs_sum, dtype=np.float64)
    sigma[mask] = abs_sum[mask].astype(np.float64) / count[mask]
    ge1 = sigma >= 1.0
    if np.any(ge1):
        k_map[ge1] = np.floor(np.log2(sigma[ge1])).astype(np.int32)
    np.minimum(k_map, 14, out=k_map)
    return k_map


def _precompute_k_map(arr: np.ndarray) -> np.ndarray:
    return _k_from_context(*_precompute_spatial_context(arr))


def _precompute_k_map_with_parent(arr: np.ndarray,
                                   parent_detail: np.ndarray) -> np.ndarray:
    abs_sum, count = _precompute_spatial_context(arr)
    h, w = arr.shape
    ph, pw = parent_detail.shape
    rows = np.minimum(np.arange(h) // 2, ph - 1)
    cols = np.minimum(np.arange(w) // 2, pw - 1)
    abs_sum += np.abs(parent_detail.astype(np.int64))[np.ix_(rows, cols)]
    count += 1
    return _k_from_context(abs_sum, count)


# ---------------------------------------------------------------------------
# Bit-level writer / reader  (batch multi-bit operations)
# ---------------------------------------------------------------------------

class BitWriter:
    __slots__ = ['data', 'byte', 'bit_pos', 'total_bits']

    def __init__(self):
        self.data = bytearray()
        self.byte = 0
        self.bit_pos = 0
        self.total_bits = 0

    def write_bit(self, b):
        self.byte = (self.byte << 1) | b
        self.bit_pos += 1
        self.total_bits += 1
        if self.bit_pos == 8:
            self.data.append(self.byte)
            self.byte = 0
            self.bit_pos = 0

    def write(self, value, nbits):
        remaining = nbits
        while remaining > 0:
            space = 8 - self.bit_pos
            if remaining >= space:
                chunk = (value >> (remaining - space)) & ((1 << space) - 1)
                self.byte = (self.byte << space) | chunk
                self.total_bits += space
                self.data.append(self.byte)
                self.byte = 0
                self.bit_pos = 0
                remaining -= space
            else:
                chunk = value & ((1 << remaining) - 1)
                self.byte = (self.byte << remaining) | chunk
                self.bit_pos += remaining
                self.total_bits += remaining
                remaining = 0

    def write_unary(self, value):
        remaining = value
        while remaining > 0:
            space = 8 - self.bit_pos
            if remaining >= space:
                self.byte = (self.byte << space) | ((1 << space) - 1)
                self.total_bits += space
                self.data.append(self.byte)
                self.byte = 0
                self.bit_pos = 0
                remaining -= space
            else:
                self.byte = (self.byte << remaining) | ((1 << remaining) - 1)
                self.bit_pos += remaining
                self.total_bits += remaining
                remaining = 0
        self.byte <<= 1
        self.bit_pos += 1
        self.total_bits += 1
        if self.bit_pos == 8:
            self.data.append(self.byte)
            self.byte = 0
            self.bit_pos = 0

    def flush(self):
        if self.bit_pos > 0:
            self.byte <<= (8 - self.bit_pos)
            self.data.append(self.byte)
            self.byte = 0
            self.bit_pos = 0

    @property
    def length(self):
        return self.total_bits

    def to_bytes(self):
        self.flush()
        return bytes(self.data)


class BitReader:
    __slots__ = ['data', 'pos', 'total_bits']

    def __init__(self, data: bytes, total_bits: int):
        self.data = data
        self.pos = 0
        self.total_bits = total_bits

    def read_bit(self):
        byte_idx = self.pos >> 3
        bit_idx  = 7 - (self.pos & 7)
        self.pos += 1
        return (self.data[byte_idx] >> bit_idx) & 1

    def read(self, nbits):
        val = 0
        remaining = nbits
        while remaining > 0:
            bit_offset = self.pos & 7
            available = 8 - bit_offset
            take = available if remaining >= available else remaining
            byte_val = self.data[self.pos >> 3]
            shift = available - take
            val = (val << take) | ((byte_val >> shift) & ((1 << take) - 1))
            self.pos += take
            remaining -= take
        return val

    def read_unary(self):
        count = 0
        while self.pos < self.total_bits:
            byte_idx = self.pos >> 3
            bit_idx = 7 - (self.pos & 7)
            self.pos += 1
            if (self.data[byte_idx] >> bit_idx) & 1:
                count += 1
            else:
                return count
        return count


# ---------------------------------------------------------------------------
# Encode subband (spatial context only) -- vectorized k, inlined coding
# ---------------------------------------------------------------------------

def encode_subband(subband: np.ndarray) -> tuple[int, bytes]:
    k_map = _precompute_k_map(subband)
    writer = BitWriter()
    arr = subband.tolist()
    km = k_map.tolist()
    h = len(arr)

    _wr_unary = writer.write_unary
    _wr = writer.write

    for r in range(h):
        row_a = arr[r]
        row_k = km[r]
        for c in range(len(row_a)):
            value = row_a[c]
            k = row_k[c]
            mapped = 2 * value if value >= 0 else -2 * value - 1
            q = mapped >> k
            if q < ESCAPE_QUOTIENT:
                _wr_unary(q)
                if k > 0:
                    _wr(mapped & ((1 << k) - 1), k)
            else:
                _wr_unary(ESCAPE_QUOTIENT)
                _wr(mapped, FIXED_CODE_BITS)

    return writer.length, writer.to_bytes()


# ---------------------------------------------------------------------------
# Decode subband (spatial context only) -- fully inlined
# ---------------------------------------------------------------------------

def decode_subband(bitstream_bytes: bytes, total_bits: int,
                   shape: tuple[int, int]) -> np.ndarray:
    reader = BitReader(bitstream_bytes, total_bits)
    h, w = shape
    out = [[0] * w for _ in range(h)]

    _rd_unary = reader.read_unary
    _rd = reader.read

    for r in range(h):
        out_row = out[r]
        prev_row = out[r - 1] if r > 0 else None
        prev2_row = out[r - 2] if r > 1 else None
        for c in range(w):
            total = 0
            cnt = 0
            if prev_row is not None:
                v = prev_row[c]
                total += (v if v >= 0 else -v) << 1
                cnt += 2
            if c > 0:
                v = out_row[c - 1]
                total += (v if v >= 0 else -v) << 1
                cnt += 2
            if prev_row is not None and c > 0:
                v = prev_row[c - 1]
                total += v if v >= 0 else -v
                cnt += 1
            if prev_row is not None and c + 1 < w:
                v = prev_row[c + 1]
                total += v if v >= 0 else -v
                cnt += 1
            if prev2_row is not None:
                v = prev2_row[c]
                total += v if v >= 0 else -v
                cnt += 1
            if c > 1:
                v = out_row[c - 2]
                total += v if v >= 0 else -v
                cnt += 1

            if cnt == 0 or total == 0:
                k = 0
            else:
                si = total // cnt
                k = (si.bit_length() - 1) if si >= 1 else 0
                if k > 14:
                    k = 14

            q = _rd_unary()
            if q < ESCAPE_QUOTIENT:
                if k > 0:
                    mapped = (q << k) | _rd(k)
                else:
                    mapped = q
            else:
                mapped = _rd(FIXED_CODE_BITS)

            out_row[c] = -(mapped + 1) // 2 if mapped & 1 else mapped // 2

    return np.array(out, dtype=np.int32)


# ---------------------------------------------------------------------------
# Encode subband with parent detail -- vectorized k, inlined coding
# ---------------------------------------------------------------------------

def encode_subband_with_parent(subband: np.ndarray,
                               parent_detail: np.ndarray) -> tuple[int, bytes]:
    k_map = _precompute_k_map_with_parent(subband, parent_detail)
    writer = BitWriter()
    arr = subband.tolist()
    km = k_map.tolist()
    h = len(arr)

    _wr_unary = writer.write_unary
    _wr = writer.write

    for r in range(h):
        row_a = arr[r]
        row_k = km[r]
        for c in range(len(row_a)):
            value = row_a[c]
            k = row_k[c]
            mapped = 2 * value if value >= 0 else -2 * value - 1
            q = mapped >> k
            if q < ESCAPE_QUOTIENT:
                _wr_unary(q)
                if k > 0:
                    _wr(mapped & ((1 << k) - 1), k)
            else:
                _wr_unary(ESCAPE_QUOTIENT)
                _wr(mapped, FIXED_CODE_BITS)

    return writer.length, writer.to_bytes()


# ---------------------------------------------------------------------------
# Decode subband with parent detail -- fully inlined
# ---------------------------------------------------------------------------

def decode_subband_with_parent(bitstream_bytes: bytes, total_bits: int,
                               shape: tuple[int, int],
                               parent_detail: np.ndarray) -> np.ndarray:
    reader = BitReader(bitstream_bytes, total_bits)
    h, w = shape
    out = [[0] * w for _ in range(h)]
    ph, pw = parent_detail.shape
    par = np.abs(parent_detail.astype(np.int64)).tolist()

    _rd_unary = reader.read_unary
    _rd = reader.read

    for r in range(h):
        out_row = out[r]
        prev_row = out[r - 1] if r > 0 else None
        prev2_row = out[r - 2] if r > 1 else None
        pr = r >> 1
        if pr >= ph:
            pr = ph - 1
        par_row = par[pr]
        for c in range(w):
            total = 0
            cnt = 0
            if prev_row is not None:
                v = prev_row[c]
                total += (v if v >= 0 else -v) << 1
                cnt += 2
            if c > 0:
                v = out_row[c - 1]
                total += (v if v >= 0 else -v) << 1
                cnt += 2
            if prev_row is not None and c > 0:
                v = prev_row[c - 1]
                total += v if v >= 0 else -v
                cnt += 1
            if prev_row is not None and c + 1 < w:
                v = prev_row[c + 1]
                total += v if v >= 0 else -v
                cnt += 1
            if prev2_row is not None:
                v = prev2_row[c]
                total += v if v >= 0 else -v
                cnt += 1
            if c > 1:
                v = out_row[c - 2]
                total += v if v >= 0 else -v
                cnt += 1

            pc = c >> 1
            if pc >= pw:
                pc = pw - 1
            total += par_row[pc]
            cnt += 1

            if cnt == 0 or total == 0:
                k = 0
            else:
                si = total // cnt
                k = (si.bit_length() - 1) if si >= 1 else 0
                if k > 14:
                    k = 14

            q = _rd_unary()
            if q < ESCAPE_QUOTIENT:
                if k > 0:
                    mapped = (q << k) | _rd(k)
                else:
                    mapped = q
            else:
                mapped = _rd(FIXED_CODE_BITS)

            out_row[c] = -(mapped + 1) // 2 if mapped & 1 else mapped // 2

    return np.array(out, dtype=np.int32)
