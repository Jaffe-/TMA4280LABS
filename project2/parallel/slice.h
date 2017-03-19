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
    double** data;

public:
    SubMatrix(int row_offset, int col_offset, int rows, int cols, int storage_dim, double* storage)
        : row_offset(row_offset)
        , col_offset(col_offset)
        , rows(rows)
        , cols(cols)
        , storage_dim(storage_dim)
    {
        data = new double*[storage_dim];
        data[0] = storage;
        for (int i = 1; i < storage_dim; i++) {
            data[i] = data[i-1] + storage_dim;
        }
    }

    ~SubMatrix() {
        delete data;
    }

    template <typename Op>
    void map(Op op) {
        #pragma omp parallel for schedule(static) collapse(2)
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                // Translate into corresponding indices in the matrix
                const int i = row_offset + row;
                const int j = col_offset + col;
                data[row][col] = op(i, j, data[row][col]);
            }
        }
    }

    void transpose() {
        #pragma omp parallel for schedule(dynamic)
        for (int row = 0; row < storage_dim; row++) {
            for (int col = 0; col < row; col++) {
                std::swap(data[row][col], data[col][row]);
            }
        }
    }

    void copyRowToBuffer(int row, double* buffer) {
        std::copy(data[row], data[row] + cols, &buffer[col_offset]);
    }

    void copyRowFromBuffer(int row, double* buffer) {
        std::copy(&buffer[col_offset], &buffer[col_offset + cols], data[row]);
    }
};


/*
  Slice class

  Represents a collection of sub-matrices that make up a
  collection of consecutive row in the matrix.
*/

class Slice {
    //    int m_n;
    int subs;
    int sub_dim;
    int last_sub_dim;
    SubMatrix** submatrices;

public:
    double* data;
    int offset;
    int rows;

    ~Slice() {
        for (int i = 0; i < subs; i++) {
            delete submatrices[i];
        }
        delete data;
    }

    Slice(const Slice&) = delete;

    Slice(int n, int slices, int index)
        : subs(slices)
    {
        const int storage_dim = std::ceil((double)n / slices);
        const int storage_size = std::pow(storage_dim, 2);
        submatrices = new SubMatrix*[slices];
        data = new double[storage_size * slices];

        offset = storage_dim * index;
        sub_dim = storage_dim;
        last_sub_dim = n - (slices - 1) * storage_dim;
        if (index < slices - 1) {
            rows = sub_dim;
        } else {
            rows = last_sub_dim;
        }

        for (int i = 0; i < slices; i++) {
            int cols = (i == slices - 1) ? last_sub_dim : storage_dim;
            submatrices[i] = new SubMatrix(offset, i * storage_dim,
                                           rows, cols,
                                           storage_dim,
                                           &data[i * storage_size]);
        }
    }

    template <typename Op>
    void map(Op op) {
        #pragma omp parallel for
        for (int i = 0; i < subs; i++) {
            submatrices[i]->map(op);
        }
    }

    template <typename Op>
    void forEachRow(Op op) {
        static double* buffer = new double[subs * sub_dim];

        for (int row = 0; row < rows; row++) {
            #pragma omp parallel for
            for (int i = 0; i < subs; i++) {
                submatrices[i]->copyRowToBuffer(row, buffer);
            }

            op(buffer);

            #pragma omp parallel for
            for (int i = 0; i < subs; i++) {
                submatrices[i]->copyRowFromBuffer(row, buffer);
            }
        }
    }

    void transpose() {
        #pragma omp parallel for
        for (int i = 0; i < subs; i++) {
            submatrices[i]->transpose();
        }
    }

};
