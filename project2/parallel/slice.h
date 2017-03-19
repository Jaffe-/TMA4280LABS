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
*/

class Slice {
    //    int m_n;
    int subs;
    int sub_dim;
    int last_sub_dim;
    SubMatrix** submatrices;

public:
    double* data;
    double* temp_data;
    int offset;
    int rows;

    ~Slice() {
        for (int i = 0; i < subs; i++) {
            delete submatrices[i];
        }
        delete data;
        delete temp_data;
    }

    Slice(const Slice&) = delete;

    Slice(int n, int slices, int index)
        : subs(slices)
    {
        const int storage_dim = std::ceil((double)n / slices);
        const int storage_size = std::pow(storage_dim, 2);
        submatrices = new SubMatrix*[slices];
        data = new double[storage_size * slices];
        temp_data = new double[storage_size * slices];

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
                                           &data[i * storage_size],
                                           &temp_data[i * storage_size]);
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
