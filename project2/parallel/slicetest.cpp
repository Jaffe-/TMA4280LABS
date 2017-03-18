#include <iostream>
#include "slice.h"
#include <assert.h>

constexpr int n = 7;

double f(int y, int x, double) {
    return n * y + x;
}

int main() {
    Slice slices[] = {{n, 4, 0}, {n, 4, 1}, {n, 4, 2}, {n, 4, 3}};

    int r = 0;
    for (Slice& slice : slices) {
        const int bin_dim = n / 4;
        const int bins = 4;
        const int bin_size = std::pow(bin_dim, 2);

        slice.map(f);

        auto check = [&] (double* row) {
            for (int col = 0; col < n; col++) {
                if (row[col] != f(r, col, 0)) {
                    std::cout << "Failed: r=" << r
                              << ", col=" << col
                              << ", val=" << row[col]
                    << ", expected=" << f(r, col, 0)
                    << "\n";
                }
            }
            r++;
        };

        slice.forEachRow(check);
    }

    Slice A(n, 4, 0);

    A.map(f);

    auto printer = [&] (int i, int j, double val) {
        std::cout << "i=" << i
        << ", j=" << j
        << ", val=" << val << "\n";
        return val;
    };

    A.transpose();

    auto check = [&] (int i, int j, double val) {
        if (j < n) {
            std::cout << "i=" << i
            << ", j=" << j
            << ", val=" << val
            << ", expected=" << f(j, i, 0) << '\n';
            assert(val == f(i, j, 0));
        }
        return val;
    };

    A.map(check);
}
