#include <mpi.h>
#include <memory>

using real = double;

extern "C" {
    void fst_(real *v, int *n, real *w, int *nn);
    void fstinv_(real *v, int *n, real *w, int *nn);
}

struct Slice {
    int m_n;
    int m_row_offset;
    int m_num_rows;
    double** m_data;

    Slice(int n, double h, int num_procs, int rank)
        : m_n(n)
    {
        const int rows_per_proc = n / num_procs;
        m_num_rows = rows_per_proc;
        m_row_offset = rows_per_proc * rank;

        m_data = new double*[rows_per_proc];
        m_data[0] = new double[rows_per_proc * n] {};
        for (int i = 1; i < rows_per_proc; i++) {
            m_data[i] = m_data[i-1] + n;
        }
    }

    template <void op(double*)>
    void forEachRow() {
        for (int i = 0; i < m_num_rows; i++) {
            op(m_data[i]);
        }
    }

    template <double op(int, int, double)>
    void map() {
        for (int i = 0; i < m_num_rows; i++) {
            for (int j = 0; j < m_n; j++) {
                m_data[i][j] = op(m_row_offset + i, j, m_data[i][j]);
            }
        }
    }
};

template <double f(int, int)>
struct Poisson {
    int m_n;
    double m_h;
    double* z;
    double* diag;

    double apply_f(int i, int j, double) {
        return m_h * m_h * f(i, j);
    }

    double solve_x(int i, int j, double val) {
        return val / (diag[i] + diag[j]);
    }

    double fst(double* vec) {
        int nn = 4 * m_n;
        fst_(vec, &m_n, z, &nn);
    }

    double fstinv(double* vec) {
        int nn = 4 * m_n;
        fstinv_(vec, &m_n, z, &nn);
    }

    double run(int n, int num_procs) {
        Slice B, BT;
        int* counts = new int[num_procs];
        int* displacements = new int[num_procs];

        for (int i = 0; i < num_procs; i++) {
            
        }

        z = new double[n];
        diag = new double[n];

        B.map<apply_f>();
        B.forEachRow<fst>();

        // TODO: make counts/displacement arrays
        MPI_Alltoallv(B.m_data, send_counts, send_displacements, MPI_DOUBLE,
                      BT.m_data, recv_counts, recv_displacements, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        BT.forEachRow<fstinv>();
        BT.map<solve_x>();
        BT.forEachRow<fst>();

        // TODO: make counts/displacement arrays
        MPI_Alltoallv(BT.m_data, send_counts, send_displacements, MPI_DOUBLE,
                      B.m_data, recv_counts, recv_displacements, MPI_DOUBLE,
                      MPI_COMM_WORLD);

        B.forEachRow<fstinv>();
    }
};

// The 2D function on the right hand side
double unity(int x, int y) {
    return 1;
}

int main(int argc, char** argv) {
    Poisson<unity> poisson;

    MPI_Init(&argc, &argv);

    MPI_Finalize();
}
