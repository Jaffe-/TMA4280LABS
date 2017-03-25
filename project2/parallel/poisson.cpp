#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <iostream>
#include "slice.h"
#include <limits>

using real = double;

extern "C" {
    void fst_(real *v, int *n, real *w, int *nn);
    void fstinv_(real *v, int *n, real *w, int *nn);
}

template <double f(double, double)>
struct Poisson {
    int n, m, rank, num_procs;
    double h;
    double* diag;
    double* grid;
    Slice B;

    Poisson(int n, int rank, int num_procs)
        : n(n),
          m(n - 1),
          rank(rank),
          num_procs(num_procs),
          h(1.0 / n),
          B(m, num_procs, rank)
    {
        diag = new double[m];
        grid = new double[n + 1];

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
        delete[] diag;
        delete[] grid;
    }

    void run() {
        auto apply_f = [this] (int i, int j, double) {
            return h * h * f(grid[i], grid[j]);
        };

        auto solve_x = [this] (int i, int j, double val) {
            return val / (diag[i] + diag[j]);
        };

        auto fst = [=] (int r, double* vec) {
            int nn = 4 * n;
            fst_(vec, &n, &vec[n], &nn);
        };

        auto fstinv = [=] (int r, double* vec) {
            int nn = 4 * n;
            fstinv_(vec, &n, &vec[n], &nn);
        };

        MPI_Barrier(MPI_COMM_WORLD);

        B.map(apply_f);
        B.forEachRow(fst);

        MPI_Alltoall(B.getSendBuffer(), B.sub_size, MPI_DOUBLE,
                     B.getRecvBuffer(), B.sub_size, MPI_DOUBLE, MPI_COMM_WORLD);

        B.transpose();
        B.forEachRow(fstinv);
        B.map(solve_x);
        B.forEachRow(fst);

        MPI_Alltoall(B.getSendBuffer(), B.sub_size, MPI_DOUBLE,
                     B.getRecvBuffer(), B.sub_size, MPI_DOUBLE, MPI_COMM_WORLD);

        B.transpose();
        B.forEachRow(fstinv);
    }

    template <double u(double, double)>
    double largestError() {
        double largest = 0;
        auto computeError = [this, &largest] (int i, int j, double val) {
            const double difference = std::abs(val - u(grid[i], grid[j]));
            #pragma omp critical
            {
            if (difference > largest)
                largest = difference;
            }
            return val;
        };
        B.map(computeError);

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

bool power_of_two(int n) {
    return (n & (n >> 1)) == 0;
}

int run(int argc, char** argv) {
    int rank, size;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2) {
        if (rank == 0)
            std::cout << "Usage:\n"
                      << "  poisson n (power of two)\n\n";
        return 1;
    }

    int n = atoi(argv[1]);

    if (!power_of_two(size)) {
        if (rank == 0)
            std::cout << "Number of processes has to be a power of 2\n\n";
        return 1;
    }

    if (!power_of_two(n)) {
        if (rank == 0)
            std::cout << "n has to be a power of 2\n\n";
        return 1;
    }

    Poisson<test_f> poisson(n, rank, size);
    poisson.run();
    double err = poisson.largestError<test_u>();
    double max_err;
    std::cout << err << "\n";
    MPI_Reduce(&err, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        std::cout << "n        : " << n << "\n"
                  << "max error: " << max_err << "\n"
                  << "h        : " << std::pow(poisson.h, 2) << "\n"
                  << "ratio    : " << max_err/std::pow(poisson.h, 2) << "\n\n";
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int err = run(argc, argv);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return err;
}
