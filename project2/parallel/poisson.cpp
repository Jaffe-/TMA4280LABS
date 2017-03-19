#include <mpi.h>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <iostream>
#include "slice.h"
#include <memory.h>

using real = double;

extern "C" {
    void fst_(real *v, int *n, real *w, int *nn);
    void fstinv_(real *v, int *n, real *w, int *nn);
}

template <double f(double, double)>
struct Poisson {
    int n, m, rank, num_procs;
    double h;
    double* z;
    double* diag;
    double* grid;
    int* counts;
    int* displacements;
    Slice B, BT;

    Poisson(int n, int rank, int num_procs)
        : n(n),
          m(n - 1),
          rank(rank),
          num_procs(num_procs),
          h(1.0 / n),
          B(m, num_procs, rank),
          BT(m, num_procs, rank)

    {
        const int elements_per_proc = std::pow(n / num_procs, 2);
        z = new double[4 * n];
        diag = new double[m];
        grid = new double[n + 1];
        counts = new int[num_procs];
        displacements = new int[num_procs];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_procs; i++) {
            counts[i] = elements_per_proc;
            displacements[i] = i * elements_per_proc;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n + 1; i++) {
            grid[i] = i * h;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            diag[i] = 2.0 * (1.0 - cos((i + 1) * M_PI / n));
        }
    }

    ~Poisson() {
        delete[] z;
        delete[] diag;
        delete[] grid;
        delete[] counts;
        delete[] displacements;
    }

    void run() {
        auto apply_f = [this] (int i, int j, double) {
            return h * h * f(grid[i], grid[j]);
        };

        auto solve_x = [this] (int i, int j, double val) {
            return val / (diag[i] + diag[j]);
        };

        auto fst = [this] (double* vec) {
            int nn = 4 * n;
            fst_(vec, &n, z, &nn);
        };

        auto fstinv = [this] (double* vec) {
            int nn = 4 * n;
            fstinv_(vec, &n, z, &nn);
        };

        MPI_Barrier(MPI_COMM_WORLD);

        B.map(apply_f);
        B.forEachRow(fst);

        MPI_Alltoallv(B.data, counts, displacements, MPI_DOUBLE,
                      BT.data, counts, displacements, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        BT.transpose();
        BT.forEachRow(fstinv);
        BT.map(solve_x);
        BT.forEachRow(fst);

        MPI_Alltoallv(BT.data, counts, displacements, MPI_DOUBLE,
                      B.data, counts, displacements, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        B.transpose();
        B.forEachRow(fstinv);
    }

    template <double u(double, double)>
    double largestError() {
        double largest = 0;
        auto computeError = [this, &largest] (int i, int j, double val) {
            const double difference = std::abs(val - u(grid[i], grid[j]));
            if (difference > largest) {
                #pragma omp critical
                largest = difference;
            }
            return val;
        };

        B.map(computeError);

        return largest;
    }

    double largest() {
        double largest = 0;
        auto computeLargest = [this, &largest] (int i, int j, double val) {
            if (val > largest) {
                #pragma omp critical
                largest = val;
            }
            return val;
        };

        B.map(computeLargest);

        return largest;
    }
};

// The solution
double test_u(double x, double y) {
    return sin(M_PI * x) * sin(2 * M_PI * y);
}

// The 2D function on the right hand side
double test_f(double x, double y) {
    return 5 * std::pow(M_PI, 2) * test_u(x, y);
}

double test_f2(double x, double y) {
    return 2 * (y - y*y + x - x*x);
}

double test_u2(double x, double y) {
    return (x - x*x) * (y - y*y);
}

void slicetest(int n, int rank, int size) {
    Slice B(n, size, rank);
    Slice BT(n, size, rank);

    int* counts = new int[size];
    int* displacements = new int[size];
    const int elements_per_proc = std::pow(std::ceil((double)n / size), 2);

    for (int i = 0; i < size; i++) {
        counts[i] = elements_per_proc;
        displacements[i] = i * elements_per_proc;
    }

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

    // Fill test values
    B.map(f);

    bool** seen = new bool*[n];
    for (int i = 0; i < n; i++) {
        seen[i] = new bool[n] {};
    }
    auto check_coverage = [seen, &n] (int i, int j, double val) {
        seen[i][j] += 1;
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
        delete seen[i];
    }
    delete seen;
    assert(covered);
    assert(!unwanted);

    // Check that we read out the expected elements when using forEachRow,
    // and negate each element to check that write-back works.
    int r = B.offset;
    auto check_forEachRow = [&] (double* row) {
        for (int col = 0; col < n; col++) {
            assert(row[col] == f(r, col, 0));
            row[col] = -row[col];
        }
        r++;
    };
    // Check that values are negative and make them positive again
    auto check_negated = [&] (int i, int j, double val) {
        assert(val == -f(i, j, 0));
        return -val;
    };
    B.forEachRow(check_forEachRow);
    B.map(check_negated);

    MPI_Alltoallv(B.data, counts, displacements, MPI_DOUBLE,
                  BT.data, counts, displacements, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    BT.transpose();

    // Check that we read out the expected transposed elements
    r = BT.offset;
    auto check_transposed = [&] (double* row) {
        for (int col = 0; col < n; col++) {
            assert(row[col] == f(col, r, 0));
        }
        r++;
    };

    BT.forEachRow(check_transposed);

    // Check that the elements are negated, this time using
    // map
    auto checkNeg = [&] (int i, int j, double val) {
        assert(val == f(j, i, 0));
        return val;
    };
    BT.map(checkNeg);
}

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool is_power_of_two = false;
    for (int i = 0; i < 32; i++) {
        if (pow(2, i) == size)
            is_power_of_two = true;
    }
    assert(is_power_of_two);

    int n = atoi(argv[1]);

    slicetest(n, rank, size);
    Poisson<test_f2> poisson(n, rank, size);
    poisson.run();
    double err = poisson.largestError<test_u2>();
    if (rank == 0) {
        double* errs = new double[size];
        MPI_Gather(&err, 1, MPI_DOUBLE, errs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double largest = 0;
        for (int i = 0; i < size; i++) {
            if (errs[i] > largest) {
                largest = errs[i];
            }
        }

        std::cout << "Largest error: " << largest << ", h*h = " << poisson.h * poisson.h << "\n";
    } else {
        MPI_Gather(&err, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    std::cout << err << "\n";

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
