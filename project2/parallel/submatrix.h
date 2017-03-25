#pragma once
#include <algorithm>

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
