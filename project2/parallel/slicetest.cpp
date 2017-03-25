#include <iostream>
#include <assert.h>
#include <mpi.h>
#include "slice.h"
#include <vector>

void slicetest(int n, int rank, int size) {
    Slice B(n, size, rank);

    // For debugging the tests
    auto printer = [&] (int i, int j, double val) {
        std::cout << "i=" << i
        << ", j=" << j
        << ", val=" << val << "\n";
        return val;
    };


    // Produce test values and check that i, j are inside bounds
    auto f = [&n] (int i, int j, double) {
        assert(i >= 0 && i < n && j >= 0 && j < n);
        return n * i + j;
    };
    B.map(f);


    // Check that when iterating using map, the function is called on
    // every expected coordinate (i, j). Also check that it is not called
    // on (i, j) outside of the slice.
    std::vector<std::vector<int>> seen(n);
    for (auto& row : seen) {
        row.reserve(n);
    }

    auto check_coverage = [&seen, &n] (int i, int j, double val) {
        #pragma omp critical
        {
            seen[i][j]++;
        }
        return val;
    };
    B.map(check_coverage);

    bool covered = true;
    bool unwanted = false;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // We are within the range
            if (i >= B.offset && i < B.offset + B.rows) {
                if (seen[i][j] != 1) {
                    covered = false;
                }
            }
            else {
                if (seen[i][j] != 0)
                    unwanted = true;
            }
        }
    }
    assert(covered);
    assert(!unwanted);


    // Check that we read out the expected elements when using forEachRow,
    // and negate each element to check that write-back works.
    auto check_forEachRow = [=] (int row, double* data) mutable {
        for (int col = 0; col < n; col++) {
            assert(data[col] == f(row, col, 0));
            data[col] = -data[col];
        }
    };
    auto check_negated = [&] (int i, int j, double val) {
        assert(val == -f(i, j, 0));
        return -val;
    };
    B.forEachRow(check_forEachRow);
    B.map(check_negated);


    // Check that we read out the expected transposed elements
    MPI_Alltoall(B.getSendBuffer(), B.sub_size, MPI_DOUBLE,
                 B.getRecvBuffer(), B.sub_size, MPI_DOUBLE, MPI_COMM_WORLD);

    B.transpose();

    auto check_transposed = [&] (int row, double* data) {
        for (int col = 0; col < n; col++) {
            assert(data[col] == f(col, row, 0));
        }
    };
    B.forEachRow(check_transposed);


    // Check that the elements are negated, this time using map
    auto checkNeg = [&] (int i, int j, double val) {
        assert(val == f(j, i, 0));
        return val;
    };
    B.map(checkNeg);


    // Check that the largest value in the slice is the expected one
    double largest = 0;
    auto computeLargest = [&largest] (int i, int j, double val) {
        #pragma omp critical
        {
            if (val > largest)
                largest = val;
        }
        return val;
    };
    B.map(computeLargest);
    assert(largest == f(n - 1, B.offset + B.rows - 1, 0));
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = atoi(argv[1]);

    slicetest(n, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
