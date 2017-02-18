#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../unittest.h"
#include "../compute_sum.h"
#include <mpi.h>

double get_element(double x, int n)
{
    return ((n % 2) ? (-1) : 1) * pow(x, 2 * n + 1) / (2 * n + 1);
}

double *generate_elements(int n)
{
    double *vec = malloc(2 * sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        vec[i] = 16 * get_element(1.0/5, i);
        vec[n+i] = -4 * get_element(1.0/239, i);
    }

    return vec;
}

double finalize(double sum)
{
    return sum;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Exactly one argument expected\n");
        return -1;
    }
    int n = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    compute_sum(n, 2, &generate_elements, &finalize);

    return 0;
}
