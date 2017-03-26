#pragma once
#include <cmath>
#include <memory>
#include <algorithm>
#include "submatrix.h"

/*
  Slice class

  Represents a collection of sub-matrices that make up a
  collection of consecutive row in the matrix.

  The slice allocates two buffers of size (n/p)^2 * p,
  where one holds the elements of the matrix and the other
  is used when transposing and calculating fst / inverse fst.
*/

class Slice {
    int subs;
    int row_size;
    std::unique_ptr<std::unique_ptr<SubMatrix>[]> submatrices;

public:
    std::unique_ptr<double[]> data;
    std::unique_ptr<double[]> temp_data;
    int rows;
    int offset;
    int sub_size;

    Slice(const Slice&) = delete;

    Slice(int n, int slices, int index)
        : subs(slices)
    {
        const int storage_dim = std::ceil((double)n / slices);
        row_size = storage_dim * slices;
        sub_size = std::pow(storage_dim, 2);
        offset = storage_dim * index;

        submatrices = std::make_unique<std::unique_ptr<SubMatrix>[]>(slices);
        data = std::make_unique<double[]>(sub_size * slices);
        temp_data = std::make_unique<double[]>(std::max(sub_size * slices, 5 * row_size));

        int last_sub_dim = n - (slices - 1) * storage_dim;
        if (index < slices - 1) {
            rows = storage_dim;
        } else {
            rows = last_sub_dim;
        }

        for (int i = 0; i < slices; i++) {
            int cols = (i == slices - 1) ? last_sub_dim : storage_dim;
            submatrices[i] =
                std::make_unique<SubMatrix>(offset, i * storage_dim,
                                            rows, cols,
                                            storage_dim,
                                            &data.get()[i * sub_size],
                                            &temp_data.get()[i * sub_size]);
        }
    }

    template <typename... Args>
    void eachSubMatrix(void (SubMatrix::*fn)(Args...), Args... args) {
        for (int i = 0; i < subs; i++) {
            (submatrices[i].get()->*fn)(args...);
        }
    }

    template <typename Op>
    void map(Op op) {
        eachSubMatrix(&SubMatrix::map, op);
    }

    template <typename Op>
    void forEachRows(int start, int num, Op op) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num; i++) {
            double* const buffer = &temp_data[i * 5 * row_size];
            eachSubMatrix(&SubMatrix::copyRowToBuffer, start + i, buffer);
            op(offset + start + i, buffer);
            eachSubMatrix(&SubMatrix::copyRowFromBuffer, start + i, buffer);
        }
    }

    template <typename Op>
    void forEachRow(Op op) {
        const int chunk_size = std::max(rows / 5, 1);
        const int chunks = rows / chunk_size;
        const int last_chunk_size = rows - chunks * chunk_size;

        for (int chunk = 0; chunk < chunks; chunk++) {
            forEachRows(chunk * chunk_size, chunk_size, op);
        }
        if (last_chunk_size > 0) {
            forEachRows(chunks * chunk_size, last_chunk_size, op);
        }
    }

    void transpose() {
        eachSubMatrix(&SubMatrix::transpose);
    }

    double* getSendBuffer() {
        return data.get();
    }

    double* getRecvBuffer() {
        return temp_data.get();
    }

};
