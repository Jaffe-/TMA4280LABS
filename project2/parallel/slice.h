#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <omp.h>


/*
  SubMatrix class

  Represents a partition of the matrix. It imposes its
  structure on a given data buffer.
*/

class SubMatrix {
    int row_offset;
    int col_offset;
    int rows;
    int cols;
    int storage_dim;
    double* data;
    double* temp_data;

    int index(int i, int j) {
        return storage_dim * i + j;
    }

public:
    SubMatrix(int row_offset, int col_offset, int rows, int cols,
              int storage_dim, double* data, double* temp_data)
        : row_offset(row_offset)
        , col_offset(col_offset)
        , rows(rows)
        , cols(cols)
        , storage_dim(storage_dim)
        , data(data)
        , temp_data(temp_data)
    {
    }

    template <typename Op>
    void map(Op op) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // Translate into corresponding indices in the matrix
                const int i = row_offset + row;
                const int j = col_offset + col;
                data[index(row, col)] = op(i, j, data[index(row, col)]);
            }
        }
    }

    void transpose() {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int row = 0; row < storage_dim; row++) {
            for (int col = 0; col < storage_dim; col++) {
                data[index(row, col)] = temp_data[index(col, row)];
            }
        }
    }

    void copyRowToBuffer(int row, double* buffer) {
        std::copy(&data[index(row, 0)], &data[index(row, 0)] + cols, &buffer[col_offset]);
    }

    void copyRowFromBuffer(int row, double* buffer) {
        std::copy(&buffer[col_offset], &buffer[col_offset + cols], &data[index(row, 0)]);
    }
};


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
        temp_data = std::make_unique<double[]>(sub_size * slices);

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

    template <typename Op>
    void map(Op op) {
        for (int i = 0; i < subs; i++) {
            submatrices[i]->map(op);
        }
    }

    template <typename Op>
    void forEachRows(int start, int num, Op op) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num; i++) {
            double* const buffer = &temp_data[i * 5 * row_size];
            for (int j = 0; j < subs; j++) {
                submatrices[j]->copyRowToBuffer(start + i, buffer);
            }
            op(offset + start + i, buffer);
            for (int j = 0; j < subs; j++) {
                submatrices[j]->copyRowFromBuffer(start + i, buffer);
            }
        }
    }

    template <typename Op>
    void forEachRow(Op op) {
        const int chunk_size = rows / 5;
        const int chunks = (chunk_size == 0) ? 0 : 5;
        const int leftover = (chunk_size == 0) ? rows : rows % chunk_size;

        for (int chunk = 0; chunk < chunks; chunk++) {
            forEachRows(chunk * chunk_size, chunk_size, op);
        }
        if (leftover > 0) {
            forEachRows(chunks * chunk_size, leftover, op);
        }
    }

    void transpose() {
        for (int i = 0; i < subs; i++) {
            submatrices[i]->transpose();
        }
    }

    double* getSendBuffer() {
        return data.get();
    }

    double* getRecvBuffer() {
        return temp_data.get();
    }

};
