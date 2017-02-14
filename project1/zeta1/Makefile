CSRC = $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)
OBJ = $(CSRC:.c=.o)

CFLAGS := -std=gnu11
LDFLAGS = -lm

zeta1 : $(CSRC)
	mpicc $^ $(LDFLAGS) -o $@

clean:
	rm -rf *.o
