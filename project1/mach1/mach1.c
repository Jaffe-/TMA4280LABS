#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../unittest.h"
#include <mpi.h>

double sum(double array[], int n)
{
    double sum = 0;

    for (int i = 0; i < n; i++) {
        sum += array[i];
    }

    return sum;
}

double get_element(double x, int n)
{
    return ((n % 2) ? (-1) : 1) * pow(x, 2 * n + 1) / (2 * n + 1);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Status status;

    double t_start, t_finished;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    t_start = MPI_Wtime();

    bool is_power_of_two = false;
    for (int i = 0; i < 32; i++) {
        if (pow(2, i) == size)
            is_power_of_two = true;
    }
    assert(is_power_of_two);

    if (argc != 2) {
        printf("Exactly one argument expected\n");
        return -1;
    }
    int n = atoi(argv[1]);
    int n_vec = n / size;
    int n_remains = n % size;

    if (rank == 0) {
        /* Create vector of values for each other process */
        double *vec = malloc(2 * sizeof(double) * n_vec);
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < n_vec; j++) {
                int idx = n_vec * i + n_remains + j + 1;
                vec[j] = 16 * get_element(1.0/5, idx);
                vec[n_vec + j] = - 4 * get_element(1.0/239, idx);
            }
            MPI_Send(vec, 2 * n_vec, MPI_DOUBLE, i, 100, MPI_COMM_WORLD);
        }
        free(vec);

        double result_sum = 0;

        /* The root process sums up the first n_vec + n_remains elements */
        for (int i = 0; i <= n_vec + n_remains; i++) {
            result_sum += 16 * get_element(1.0/5, i) - 4 * get_element(1.0/239, i);
        }

        /* Receive partial sums from each process and accumulate it */
        for (int i = 1; i < size; i++) {
            double result;

            MPI_Recv(&result, 1, MPI_DOUBLE, i, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            result_sum += result;
        }

        t_finished = MPI_Wtime();
        printf("Result: %f, time = %f\n", result_sum, t_finished - t_start);
    }
    else {
        double *vec = malloc(2 * sizeof(double) * n_vec);
        double result = 0;

        MPI_Recv(vec, 2 * n_vec, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        result = sum(vec, n_vec);
        MPI_Send(&result, 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD);
        free(vec);
    }

    MPI_Finalize();
    return 0;
}
