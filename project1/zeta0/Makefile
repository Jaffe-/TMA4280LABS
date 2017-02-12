CSRC = $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)
OBJ = $(CSRC:.c=.o)

CFLAGS := -std=gnu11
LDFLAGS = -lm

zeta0 : $(OBJ)
	gcc $^ $(LDFLAGS) -o $@

utest : CFLAGS += -DDO_UNITTEST
utest : clean zeta0
	./zeta0

vtest : CFLAGS += -DDO_VTEST
vtest : clean zeta0
	./zeta0 > verification_output.txt

%.o : %.c
	gcc $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o
