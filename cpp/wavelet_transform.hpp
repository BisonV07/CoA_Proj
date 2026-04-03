#pragma once
#include "matrix.hpp"
#include <algorithm>
#include <vector>

// -----------------------------------------------------------------------
// 1-D CDF 5/3 forward / inverse (lifting scheme, int32 arithmetic)
// -----------------------------------------------------------------------

inline void cdf53_forward_1d(const int32_t* signal, int n,
                              int32_t* even, int32_t* odd) {
    int n_even = (n + 1) / 2;
    int n_odd  = n / 2;

    if (n < 2) {
        if (n == 1) even[0] = signal[0];
        return;
    }

    for (int i = 0; i < n_even; i++) even[i] = signal[2 * i];
    for (int i = 0; i < n_odd;  i++) odd[i]  = signal[2 * i + 1];

    // Predict
    for (int i = 0; i < n_odd; i++) {
        int32_t er = even[std::min(i + 1, n_even - 1)];
        odd[i] -= (even[i] + er) >> 1;
    }

    // Update
    for (int i = 0; i < n_even; i++) {
        int32_t ol = odd[std::max(i - 1, 0)];
        int32_t or_ = odd[std::min(i, n_odd - 1)];
        even[i] += (ol + or_ + 2) >> 2;
    }
}

inline void cdf53_inverse_1d(const int32_t* even_in, int n_even,
                              const int32_t* odd_in, int n_odd,
                              int32_t* output) {
    if (n_odd == 0) {
        if (n_even > 0) output[0] = even_in[0];
        return;
    }

    std::vector<int32_t> even(even_in, even_in + n_even);
    std::vector<int32_t> odd(odd_in, odd_in + n_odd);

    // Undo update
    for (int i = 0; i < n_even; i++) {
        int32_t ol = odd[std::max(i - 1, 0)];
        int32_t or_ = odd[std::min(i, n_odd - 1)];
        even[i] -= (ol + or_ + 2) >> 2;
    }

    // Undo predict
    for (int i = 0; i < n_odd; i++) {
        int32_t er = even[std::min(i + 1, n_even - 1)];
        odd[i] += (even[i] + er) >> 1;
    }

    for (int i = 0; i < n_even; i++) output[2 * i]     = even[i];
    for (int i = 0; i < n_odd;  i++) output[2 * i + 1] = odd[i];
}

// -----------------------------------------------------------------------
// 2-D separable transform (rows then columns)
// -----------------------------------------------------------------------

inline void forward_rows(const Matrix& in, Matrix& low, Matrix& high) {
    int h = in.rows, w = in.cols;
    int wl = (w + 1) / 2, wh = w / 2;
    low  = Matrix(h, wl);
    high = Matrix(h, wh);

    for (int r = 0; r < h; r++)
        cdf53_forward_1d(in.row_ptr(r), w, low.row_ptr(r), high.row_ptr(r));
}

inline void forward_cols(const Matrix& in, Matrix& low, Matrix& high) {
    int h = in.rows, w = in.cols;
    int hl = (h + 1) / 2, hh = h / 2;
    low  = Matrix(hl, w);
    high = Matrix(hh, w);

    std::vector<int32_t> col(h), ev(hl), od(hh);
    for (int c = 0; c < w; c++) {
        for (int r = 0; r < h; r++) col[r] = in.at(r, c);
        cdf53_forward_1d(col.data(), h, ev.data(), od.data());
        for (int r = 0; r < hl; r++) low.at(r, c)  = ev[r];
        for (int r = 0; r < hh; r++) high.at(r, c) = od[r];
    }
}

inline void inverse_rows(const Matrix& low, const Matrix& high, Matrix& out) {
    int h = low.rows;
    int w = low.cols + high.cols;
    out = Matrix(h, w);

    for (int r = 0; r < h; r++)
        cdf53_inverse_1d(low.row_ptr(r), low.cols,
                         high.row_ptr(r), high.cols,
                         out.row_ptr(r));
}

inline void inverse_cols(const Matrix& low, const Matrix& high, Matrix& out) {
    int h = low.rows + high.rows;
    int w = low.cols;
    out = Matrix(h, w);

    std::vector<int32_t> ev(low.rows), od(high.rows), col(h);
    for (int c = 0; c < w; c++) {
        for (int r = 0; r < low.rows;  r++) ev[r] = low.at(r, c);
        for (int r = 0; r < high.rows; r++) od[r] = high.at(r, c);
        cdf53_inverse_1d(ev.data(), low.rows, od.data(), high.rows, col.data());
        for (int r = 0; r < h; r++) out.at(r, c) = col[r];
    }
}

// One level 2-D forward: image -> (LL, LH, HL, HH)
inline void cdf53_forward_2d(const Matrix& image,
                              Matrix& LL, Matrix& LH, Matrix& HL, Matrix& HH) {
    Matrix row_low, row_high;
    forward_rows(image, row_low, row_high);
    forward_cols(row_low,  LL, LH);
    forward_cols(row_high, HL, HH);
}

// One level 2-D inverse: (LL, LH, HL, HH) -> image
inline void cdf53_inverse_2d(const Matrix& LL, const Matrix& LH,
                              const Matrix& HL, const Matrix& HH,
                              Matrix& image) {
    Matrix row_low, row_high;
    inverse_cols(LL, LH, row_low);
    inverse_cols(HL, HH, row_high);
    inverse_rows(row_low, row_high, image);
}

// -----------------------------------------------------------------------
// Multi-level decomposition
// -----------------------------------------------------------------------

struct WaveletLevel {
    Matrix lh, hl, hh;
};

struct WaveletDecomp {
    Matrix ll;
    std::vector<WaveletLevel> levels; // levels[0] = finest
};

inline WaveletDecomp multilevel_forward(const Matrix& image, int num_levels = 3) {
    WaveletDecomp decomp;
    decomp.levels.resize(num_levels);

    Matrix current = image;
    for (int i = 0; i < num_levels; i++) {
        Matrix LL;
        cdf53_forward_2d(current, LL,
                         decomp.levels[i].lh,
                         decomp.levels[i].hl,
                         decomp.levels[i].hh);
        current = std::move(LL);
    }
    decomp.ll = std::move(current);
    return decomp;
}

inline Matrix multilevel_inverse(const WaveletDecomp& decomp) {
    Matrix current = decomp.ll;
    for (int i = (int)decomp.levels.size() - 1; i >= 0; i--) {
        Matrix out;
        cdf53_inverse_2d(current, decomp.levels[i].lh,
                         decomp.levels[i].hl, decomp.levels[i].hh, out);
        current = std::move(out);
    }
    return current;
}
