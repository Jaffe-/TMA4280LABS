#pragma once
#include <algorithm>
#include <cmath>

/*
  Slice class

  Represents a set of consecutive rows in an array.
  The slice is divided into a given number of bins which
  simplifies doing the multi-process transpose operation.
*/

class Slice {
    int m_n;
    int m_bins;
    int m_rows;
    int m_offset;
    int m_bin_dim;
    double** m_data;

public:
    Slice() = delete;

    Slice(int n, int slices, int index)
        : m_n(n)
        , m_bins(slices)
    {
        const int rows = std::ceil((double)n / slices);

        // All slices should have the same size except for the
        // last one, which should fill up the rest of the rows.
        if (index < slices - 1) {
            m_rows = rows;
        } else {
            m_rows = n - (slices - 1) * rows;
        }
        m_offset = rows * index;
        m_data = new double*[slices];
        m_bin_dim = rows;

        // All slices (including the last one) should use the same
        // memory layout for compatability between processes, so rows
        // and not m_rows is used here
        const int bin_size = std::pow(rows, 2);
        m_data[0] = new double[bin_size * slices] {};
        for (int i = 1; i < m_bins; i++) {
            m_data[i] = m_data[i-1] + bin_size;
        }
    }

    /*
      Go through each bin and collect the sub arrays that make up the
      requested row.
    */
    double* getRow(int row) {
        // Only allocate once and reuse that buffer
        static double* row_buffer = new double[m_bins * m_bin_dim];

        for (int i = 0; i < m_bins; i++) {
            const int columns = (i < m_bins - 1) ? m_bin_dim : m_rows;
            std::copy(&m_data[i][row * m_bin_dim], &m_data[i][row * m_bin_dim + columns],
                      &row_buffer[i * m_bin_dim]);
        }

        return row_buffer;
    }

    /*
      Apply the function op to every row
    */
    template <typename Op>
    void forEachRow(Op op) {
        for (int i = 0; i < m_rows; i++) {
            double* row = getRow(i);
            op(row);
        }
    }

    /*
      Apply the function op to every element in the slice.
      The internal data structure representation is translated to matrix
      coordinates before calling the function.
    */
    template <typename Op>
    void map(Op op) {
        for (int bin = 0; bin < m_bins; bin++) {
            const int columns = (bin < m_bins - 1) ? m_bin_dim : m_bin_dim - 1;
            for (int row = 0; row < m_rows; row++) {
                for (int col = 0; col < columns; col++) {
                    const int bin_idx = row * m_bin_dim + col;

                    // Translate into corresponding indices in the matrix
                    const int i = m_offset + row;
                    const int j = bin * m_bin_dim + col;
                    m_data[bin][bin_idx] = op(i, j, m_data[bin][bin_idx]);
                    // std::cout << "columns=" << columns
                    //           << ", bin=" << bin
                    //           << ", row=" << row
                    //           << ", col=" << col
                    //           << ", bin_idx=" << bin_idx
                    //           << ", i=" << i
                    //           << ", j=" << j
                    //           << ", data=" << m_data[bin][bin_idx] << "\n";
                }
            }
        }
    }

    /*
      Return the raw bin array
    */
    double* getBins() {
        return m_data[0];
    }

    /* Transpose each bin */
    void transpose() {
        for (int bin = 0; bin < m_bins; bin++) {
            for (int row = 0; row < m_bin_dim; row++) {
                for (int col = 0; col < row; col++) {
                    const int bin_idx = row * m_bin_dim + col;
                    const int transposed_bin_idx = col * m_bin_dim + row;
                    std::swap(m_data[bin][bin_idx], m_data[bin][transposed_bin_idx]);
                }
            }
        }
    }
};
