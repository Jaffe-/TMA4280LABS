CSRC = ../compute_sum_openmp.c $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)

CFLAGS := -std=gnu11 -fopenmp
LDFLAGS = -lm

mach2 : $(CSRC)
	mpicc $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -rf *.o
