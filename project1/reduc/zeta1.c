#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "compute_sum.h"

double get_element(int n)
{
    return pow(n, -2);
}

double *generate_elements(int n)
{
    double *vec = malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++) {
        vec[i] = get_element(1 + i);
    }

    return vec;
}

double finalize(double sum)
{
    return sqrt(6 * sum);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Exactly one argument expected\n");
        return -1;
    }
    int n = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    compute_sum(n, 1, &generate_elements, &finalize);

    return 0;
}
