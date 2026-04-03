#pragma once
#include "matrix.hpp"
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

static constexpr int ESCAPE_QUOTIENT = 12;
static constexpr int FIXED_CODE_BITS = 16;
static constexpr int MAX_K = 14;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

inline int floor_log2(int64_t v) {
    if (v <= 0) return 0;
    int k = 0;
    while (v >= 2) { v >>= 1; k++; }
    return k;
}

inline int32_t med_predict(int32_t top, int32_t left, int32_t top_left) {
    if (top_left >= std::max(top, left)) return std::min(top, left);
    if (top_left <= std::min(top, left)) return std::max(top, left);
    return top + left - top_left;
}

// -----------------------------------------------------------------------
// DPCM forward / inverse (MED predictor)
// -----------------------------------------------------------------------

inline Matrix dpcm_forward(const Matrix& sub) {
    int h = sub.rows, w = sub.cols;
    Matrix res(h, w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int32_t pred;
            if (r == 0 && c == 0)      pred = 0;
            else if (r == 0)            pred = sub.at(r, c - 1);
            else if (c == 0)            pred = sub.at(r - 1, c);
            else                        pred = med_predict(sub.at(r-1,c), sub.at(r,c-1), sub.at(r-1,c-1));
            res.at(r, c) = sub.at(r, c) - pred;
        }
    }
    return res;
}

inline Matrix dpcm_inverse(const Matrix& res) {
    int h = res.rows, w = res.cols;
    Matrix sub(h, w);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int32_t pred;
            if (r == 0 && c == 0)      pred = 0;
            else if (r == 0)            pred = sub.at(r, c - 1);
            else if (c == 0)            pred = sub.at(r - 1, c);
            else                        pred = med_predict(sub.at(r-1,c), sub.at(r,c-1), sub.at(r-1,c-1));
            sub.at(r, c) = res.at(r, c) + pred;
        }
    }
    return sub;
}

// -----------------------------------------------------------------------
// Spatial context -> k parameter
// -----------------------------------------------------------------------

inline int compute_k_spatial(const Matrix& arr, int r, int c) {
    int h = arr.rows, w = arr.cols;
    int64_t total = 0;
    int cnt = 0;

    if (r > 0) {
        total += (int64_t)std::abs(arr.at(r-1, c)) * 2;
        cnt += 2;
    }
    if (c > 0) {
        total += (int64_t)std::abs(arr.at(r, c-1)) * 2;
        cnt += 2;
    }
    if (r > 0 && c > 0) {
        total += std::abs(arr.at(r-1, c-1));
        cnt += 1;
    }
    if (r > 0 && c + 1 < w) {
        total += std::abs(arr.at(r-1, c+1));
        cnt += 1;
    }
    if (r > 1) {
        total += std::abs(arr.at(r-2, c));
        cnt += 1;
    }
    if (c > 1) {
        total += std::abs(arr.at(r, c-2));
        cnt += 1;
    }

    if (cnt == 0 || total == 0) return 0;
    int64_t si = total / cnt;
    if (si < 1) return 0;
    int k = floor_log2(si);
    return std::min(k, MAX_K);
}

inline int compute_k_with_parent(const Matrix& arr, int r, int c,
                                  const Matrix& parent) {
    int h = arr.rows, w = arr.cols;
    int64_t total = 0;
    int cnt = 0;

    if (r > 0) {
        total += (int64_t)std::abs(arr.at(r-1, c)) * 2;
        cnt += 2;
    }
    if (c > 0) {
        total += (int64_t)std::abs(arr.at(r, c-1)) * 2;
        cnt += 2;
    }
    if (r > 0 && c > 0) {
        total += std::abs(arr.at(r-1, c-1));
        cnt += 1;
    }
    if (r > 0 && c + 1 < w) {
        total += std::abs(arr.at(r-1, c+1));
        cnt += 1;
    }
    if (r > 1) {
        total += std::abs(arr.at(r-2, c));
        cnt += 1;
    }
    if (c > 1) {
        total += std::abs(arr.at(r, c-2));
        cnt += 1;
    }

    int pr = std::min(r / 2, parent.rows - 1);
    int pc = std::min(c / 2, parent.cols - 1);
    total += std::abs((int64_t)parent.at(pr, pc));
    cnt += 1;

    if (cnt == 0 || total == 0) return 0;
    int64_t si = total / cnt;
    if (si < 1) return 0;
    int k = floor_log2(si);
    return std::min(k, MAX_K);
}

// -----------------------------------------------------------------------
// BitWriter
// -----------------------------------------------------------------------

class BitWriter {
    std::vector<uint8_t> buf_;
    uint32_t byte_ = 0;
    int bit_pos_ = 0;

public:
    int total_bits = 0;

    void write_bit(int b) {
        byte_ = (byte_ << 1) | (b & 1);
        bit_pos_++;
        total_bits++;
        if (bit_pos_ == 8) {
            buf_.push_back((uint8_t)byte_);
            byte_ = 0;
            bit_pos_ = 0;
        }
    }

    void write(uint32_t value, int nbits) {
        int remaining = nbits;
        while (remaining > 0) {
            int space = 8 - bit_pos_;
            if (remaining >= space) {
                uint32_t chunk = (value >> (remaining - space)) & ((1u << space) - 1);
                byte_ = (byte_ << space) | chunk;
                total_bits += space;
                buf_.push_back((uint8_t)byte_);
                byte_ = 0;
                bit_pos_ = 0;
                remaining -= space;
            } else {
                uint32_t chunk = value & ((1u << remaining) - 1);
                byte_ = (byte_ << remaining) | chunk;
                bit_pos_ += remaining;
                total_bits += remaining;
                remaining = 0;
            }
        }
    }

    void write_unary(int value) {
        int remaining = value;
        while (remaining > 0) {
            int space = 8 - bit_pos_;
            if (remaining >= space) {
                byte_ = (byte_ << space) | ((1u << space) - 1);
                total_bits += space;
                buf_.push_back((uint8_t)byte_);
                byte_ = 0;
                bit_pos_ = 0;
                remaining -= space;
            } else {
                byte_ = (byte_ << remaining) | ((1u << remaining) - 1);
                bit_pos_ += remaining;
                total_bits += remaining;
                remaining = 0;
            }
        }
        // Write the 0 terminator
        byte_ <<= 1;
        bit_pos_++;
        total_bits++;
        if (bit_pos_ == 8) {
            buf_.push_back((uint8_t)byte_);
            byte_ = 0;
            bit_pos_ = 0;
        }
    }

    std::vector<uint8_t> flush() {
        if (bit_pos_ > 0) {
            byte_ <<= (8 - bit_pos_);
            buf_.push_back((uint8_t)byte_);
            byte_ = 0;
            bit_pos_ = 0;
        }
        return std::move(buf_);
    }
};

// -----------------------------------------------------------------------
// BitReader
// -----------------------------------------------------------------------

class BitReader {
    const uint8_t* data_;
    int pos_ = 0;
    int total_bits_;

public:
    BitReader(const uint8_t* d, int total) : data_(d), total_bits_(total) {}

    int read_bit() {
        int byte_idx = pos_ >> 3;
        int bit_idx  = 7 - (pos_ & 7);
        pos_++;
        return (data_[byte_idx] >> bit_idx) & 1;
    }

    uint32_t read(int nbits) {
        uint32_t val = 0;
        int remaining = nbits;
        while (remaining > 0) {
            int bit_offset = pos_ & 7;
            int available  = 8 - bit_offset;
            int take = (remaining >= available) ? available : remaining;
            uint8_t byte_val = data_[pos_ >> 3];
            int shift = available - take;
            val = (val << take) | ((byte_val >> shift) & ((1u << take) - 1));
            pos_ += take;
            remaining -= take;
        }
        return val;
    }

    int read_unary() {
        int count = 0;
        while (pos_ < total_bits_) {
            int byte_idx = pos_ >> 3;
            int bit_idx  = 7 - (pos_ & 7);
            pos_++;
            if ((data_[byte_idx] >> bit_idx) & 1)
                count++;
            else
                return count;
        }
        return count;
    }
};

// -----------------------------------------------------------------------
// Encoded subband data
// -----------------------------------------------------------------------

struct SubbandEncoded {
    int rows = 0, cols = 0;
    int total_bits = 0;
    std::vector<uint8_t> packed;
};

// -----------------------------------------------------------------------
// Encode / decode subband (spatial context only)
// -----------------------------------------------------------------------

inline SubbandEncoded encode_subband(const Matrix& sub) {
    BitWriter writer;
    int h = sub.rows, w = sub.cols;

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int k = compute_k_spatial(sub, r, c);
            int32_t value = sub.at(r, c);
            uint32_t mapped = (value >= 0) ? (uint32_t)(2 * value)
                                           : (uint32_t)(-2 * value - 1);
            uint32_t q = mapped >> k;

            if ((int)q < ESCAPE_QUOTIENT) {
                writer.write_unary((int)q);
                if (k > 0) writer.write(mapped & ((1u << k) - 1), k);
            } else {
                writer.write_unary(ESCAPE_QUOTIENT);
                writer.write(mapped, FIXED_CODE_BITS);
            }
        }
    }

    SubbandEncoded enc;
    enc.rows = h;
    enc.cols = w;
    enc.total_bits = writer.total_bits;
    enc.packed = writer.flush();
    return enc;
}

inline Matrix decode_subband(const SubbandEncoded& enc) {
    BitReader reader(enc.packed.data(), enc.total_bits);
    int h = enc.rows, w = enc.cols;
    Matrix out(h, w);

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int k = compute_k_spatial(out, r, c);

            int q = reader.read_unary();
            uint32_t mapped;
            if (q < ESCAPE_QUOTIENT) {
                mapped = (k > 0) ? ((uint32_t)(q << k) | reader.read(k))
                                 : (uint32_t)q;
            } else {
                mapped = reader.read(FIXED_CODE_BITS);
            }

            out.at(r, c) = (mapped & 1)
                ? -(int32_t)((mapped + 1) / 2)
                :  (int32_t)(mapped / 2);
        }
    }
    return out;
}

// -----------------------------------------------------------------------
// Encode / decode subband with parent detail context
// -----------------------------------------------------------------------

inline SubbandEncoded encode_subband_with_parent(const Matrix& sub,
                                                  const Matrix& parent) {
    BitWriter writer;
    int h = sub.rows, w = sub.cols;

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int k = compute_k_with_parent(sub, r, c, parent);
            int32_t value = sub.at(r, c);
            uint32_t mapped = (value >= 0) ? (uint32_t)(2 * value)
                                           : (uint32_t)(-2 * value - 1);
            uint32_t q = mapped >> k;

            if ((int)q < ESCAPE_QUOTIENT) {
                writer.write_unary((int)q);
                if (k > 0) writer.write(mapped & ((1u << k) - 1), k);
            } else {
                writer.write_unary(ESCAPE_QUOTIENT);
                writer.write(mapped, FIXED_CODE_BITS);
            }
        }
    }

    SubbandEncoded enc;
    enc.rows = h;
    enc.cols = w;
    enc.total_bits = writer.total_bits;
    enc.packed = writer.flush();
    return enc;
}

inline Matrix decode_subband_with_parent(const SubbandEncoded& enc,
                                          const Matrix& parent) {
    BitReader reader(enc.packed.data(), enc.total_bits);
    int h = enc.rows, w = enc.cols;
    Matrix out(h, w);

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int k = compute_k_with_parent(out, r, c, parent);

            int q = reader.read_unary();
            uint32_t mapped;
            if (q < ESCAPE_QUOTIENT) {
                mapped = (k > 0) ? ((uint32_t)(q << k) | reader.read(k))
                                 : (uint32_t)q;
            } else {
                mapped = reader.read(FIXED_CODE_BITS);
            }

            out.at(r, c) = (mapped & 1)
                ? -(int32_t)((mapped + 1) / 2)
                :  (int32_t)(mapped / 2);
        }
    }
    return out;
}
