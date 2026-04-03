#pragma once
#include "matrix.hpp"

inline void rgb_to_ycocg_r(const Matrix& r, const Matrix& g, const Matrix& b,
                            Matrix& y, Matrix& co, Matrix& cg) {
    int n = r.size();
    y = Matrix(r.rows, r.cols);
    co = Matrix(r.rows, r.cols);
    cg = Matrix(r.rows, r.cols);

    for (int i = 0; i < n; i++) {
        int32_t rv = r.data[i], gv = g.data[i], bv = b.data[i];
        int32_t co_v = rv - bv;
        int32_t t = bv + (co_v >> 1);
        int32_t cg_v = gv - t;
        int32_t y_v = t + (cg_v >> 1);
        y.data[i] = y_v;
        co.data[i] = co_v;
        cg.data[i] = cg_v;
    }
}

inline void ycocg_r_to_rgb(const Matrix& y, const Matrix& co, const Matrix& cg,
                            Matrix& r, Matrix& g, Matrix& b) {
    int n = y.size();
    r = Matrix(y.rows, y.cols);
    g = Matrix(y.rows, y.cols);
    b = Matrix(y.rows, y.cols);

    for (int i = 0; i < n; i++) {
        int32_t yv = y.data[i], cov = co.data[i], cgv = cg.data[i];
        int32_t t = yv - (cgv >> 1);
        int32_t gv = cgv + t;
        int32_t bv = t - (cov >> 1);
        int32_t rv = cov + bv;
        r.data[i] = rv;
        g.data[i] = gv;
        b.data[i] = bv;
    }
}
