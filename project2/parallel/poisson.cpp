#include <mpi.h>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <iostream>
#include "slice.h"

using real = double;

extern "C" {
    void fst_(real *v, int *n, real *w, int *nn);
    void fstinv_(real *v, int *n, real *w, int *nn);
}

template <double f(int, int)>
struct Poisson {
    int n, m, rank, num_procs;
    double h;
    double* z;
    double* diag;
    double* grid;
    int* counts;
    int* displacements;

    Poisson(int n, int rank, int num_procs)
        : n(n),
          m(n - 1),
          rank(rank),
          num_procs(num_procs),
          h(1.0 / n)
    {
        const int cols_per_proc = n / num_procs;;
        z = new double[4 * n];
        diag = new double[m];
        grid = new double[n + 1];
        counts = new int[num_procs];
        displacements = new int[num_procs];

        for (int i = 0; i < num_procs; i++) {
            counts[i] = cols_per_proc;;
            displacements[i] = i * cols_per_proc;
        }

        for (int i = 0; i < n + 1; i++) {
            grid[i] = i * h;
        }

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
        Slice B(m, num_procs, rank);
        Slice BT(m, num_procs, rank);

        auto apply_f = [this] (int i, int j, double) {
            return h * h * f(grid[i], grid[j]);
        };

        auto solve_x = [=] (int i, int j, double val) {
            return val / (diag[i] + diag[j]);
        };

        auto fst = [=] (double* vec) {
            int nn = 4 * n;
            fst_(vec, &n, z, &nn);
        };

        auto fstinv = [=] (double* vec) {
            int nn = 4 * n;
            fstinv_(vec, &n, z, &nn);
        };

        MPI_Barrier(MPI_COMM_WORLD);

        B.map(apply_f);
        B.forEachRow(fst);

        MPI_Alltoallv(B.getBins(), counts, displacements, MPI_DOUBLE,
                      BT.getBins(), counts, displacements, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        BT.transpose();
        BT.forEachRow(fstinv);
        BT.map(solve_x);
        BT.forEachRow(fst);

        MPI_Alltoallv(BT.getBins(), counts, displacements, MPI_DOUBLE,
                      B.getBins(), counts, displacements, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        B.transpose();
        B.forEachRow(fstinv);
    }
};

// The 2D function on the right hand side
double unity(int x, int y) {
    return 1;
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

    Poisson<unity> poisson(n, rank, size);
    poisson.run();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
