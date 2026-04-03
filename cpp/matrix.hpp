#pragma once
#include <vector>
#include <cstdint>

struct Matrix {
    std::vector<int32_t> data;
    int rows = 0, cols = 0;

    Matrix() = default;
    Matrix(int r, int c) : data((size_t)r * c, 0), rows(r), cols(c) {}

    int32_t& at(int r, int c) { return data[(size_t)r * cols + c]; }
    const int32_t& at(int r, int c) const { return data[(size_t)r * cols + c]; }

    int32_t* row_ptr(int r) { return data.data() + (size_t)r * cols; }
    const int32_t* row_ptr(int r) const { return data.data() + (size_t)r * cols; }

    int size() const { return rows * cols; }
    bool empty() const { return data.empty(); }
};
