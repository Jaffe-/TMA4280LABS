#include <iostream>
#include <vector>
#include "slice.h"
#include <assert.h>

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    int numslices = atoi(argv[2]);

    std::vector<Slice*> Bslices;
    std::vector<Slice*> BTslices;
    for (int i = 0; i < numslices; i++) {
        Bslices.push_back(new Slice(n, numslices, i));
        BTslices.push_back(new Slice(n, numslices, i));
    }

    auto f = [&n] (int y, int x, double) {
        return n * y + x;
    };

    int r = 0;
    for (Slice* slice : Bslices) {
        slice->map(f);

        auto check = [&] (double* row) {
            for (int col = 0; col < n; col++) {
                assert(row[col] == f(r, col, 0));
            }
            r++;
        };

        slice->forEachRow(check);
    }

    for (Slice* slice : BTslices) {
        
    }
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

    //    A.map(check);
}
