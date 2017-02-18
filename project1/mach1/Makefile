CSRC = ../compute_sum.c $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)

CFLAGS := -std=gnu11
LDFLAGS = -lm

mach1 : $(CSRC)
	mpicc $^ $(LDFLAGS) -o $@

clean:
	rm -rf *.o
