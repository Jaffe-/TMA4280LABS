#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <stdbool.h>

double sum(double array[], int n)
{
    double sum = 0;

    for (int i = 0; i < n; i++) {
        sum += array[i];
    }

    return sum;
}

/* Common function for computing partial sums */
void compute_sum(int n, int elements_per_term, double* (*generate_elements)(int), double (*finalize)(double))
{
    int rank, size;
    MPI_Status status;
    double t_start, t_finished;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    bool is_power_of_two = false;
    for (int i = 0; i < 32; i++) {
        if (pow(2, i) == size)
            is_power_of_two = true;
    }
    assert(is_power_of_two);

    int n_vec = (n * elements_per_term) / size;
    int n_remains = (n * elements_per_term) % size;

    double *vec = NULL;
    double *my_vec = malloc(sizeof(double) * n_vec);

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    if (rank == 0) {
        /* Create vector of values for each other process */
        vec = generate_elements(n);
    }

    MPI_Scatter(&vec[n_remains], n_vec, MPI_DOUBLE,
                my_vec, n_vec, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double partial_sum = sum(my_vec, n_vec);

    /* The root process also has to sum up the remaining elements */
    if (rank == 0) {
        partial_sum += sum(vec, n_remains);
    }

    double final_sum;
    MPI_Reduce(&partial_sum, &final_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double result = finalize(final_sum);
        t_finished = MPI_Wtime();
        printf("Result: %f\nError: %.17g\nTime: %f\n", result, fabs(M_PI - result), t_finished - t_start);

        free(vec);
    }

    free(my_vec);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
