/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846
#define true 1
#define false 0

typedef double real;
typedef int bool;

// Function prototypes
real *mk_1D_array(size_t n, bool zero);
real **mk_2D_array(size_t n1, size_t n2, bool zero);
void transpose(real **bt, real **b, size_t m);
real sol(real x, real y);
real rhs(real x, real y);
void fst_(real *v, int *n, real *w, int *nn);
void fstinv_(real *v, int *n, real *w, int *nn);

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("Usage:\n");
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  n: the problem size (must be a power of 2)\n");
    }

    // The number of grid points in each direction is n+1
    // The number of degrees of freedom in each direction is n-1
    int n = atoi(argv[1]);
    int m = n - 1;
    int nn = 4 * n;
    real h = 1.0 / n;

    // Grid points
    real *grid = mk_1D_array(n+1, false);
    for (size_t i = 0; i < n+1; i++) {
        grid[i] = i * h;
    }

    // The diagonal of the eigenvalue matrix of T
    real *diag = mk_1D_array(m, false);
    for (size_t i = 0; i < m; i++) {
        diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n));
    }

    // Initialize the right hand side data
    real **b = mk_2D_array(m, m, false);
    real **bt = mk_2D_array(m, m, false);
    real *z = mk_1D_array(nn, false);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            b[i][j] = h * h * rhs(grid[i + 1], grid[j + 1]);
        }
    }

    // Calculate Btilde^T = S^-1 * (S * B)^T
    for (size_t i = 0; i < m; i++) {
        fst_(b[i], &n, z, &nn);
    }
    transpose(bt, b, m);
    for (size_t i = 0; i < m; i++) {
        fstinv_(bt[i], &n, z, &nn);
    }

    // Solve Lambda * Xtilde = Btilde
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = bt[i][j] / (diag[i] + diag[j]);
        }
    }

    // Calculate X = S^-1 * (S * Xtilde^T)
    for (size_t i = 0; i < m; i++) {
        fst_(bt[i], &n, z, &nn);
    }
    transpose(b, bt, m);
    for (size_t i = 0; i < m; i++) {
        fstinv_(b[i], &n, z, &nn);
    }

    // Calculate maximal value of solution
    double u_max = 0.0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            double difference = fabs(b[i][j] - sol(grid[i + 1], grid[j + 1]));
            if (difference > u_max) {
                u_max = difference;
            }
        }
    }

    printf("u_max = %e\n", u_max);

    return 0;
}

/* real sol(real x, real y) { */
/*     return (x - x*x) * (y - y*y); */
/* } */

/* real rhs(real x, real y) { */
/*     return 2 * (y - y*y + x - x*x); */
/* } */

real sol(real x, real y) {
    return sin(PI * x) * sin(2 * PI * y);
}

real rhs(real x, real y) {
    return 5 * pow(PI, 2) * sol(x, y);
}

void transpose(real **bt, real **b, size_t m)
{
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = b[j][i];
        }
    }
}

real *mk_1D_array(size_t n, bool zero)
{
    if (zero) {
        return (real *)calloc(n, sizeof(real));
    }
    return (real *)malloc(n * sizeof(real));
}

real **mk_2D_array(size_t n1, size_t n2, bool zero)
{
    real **ret = (real **)malloc(n1 * sizeof(real *));

    if (zero) {
        ret[0] = (real *)calloc(n1 * n2, sizeof(real));
    }
    else {
        ret[0] = (real *)malloc(n1 * n2 * sizeof(real));
    }

    for (size_t i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}
