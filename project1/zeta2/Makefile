CSRC = ../compute_sum_openmp.c $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)

CFLAGS := -std=gnu11 -fopenmp
LDFLAGS = -lm

zeta2 : $(CSRC)
	mpicc $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -rf *.o
