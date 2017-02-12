CSRC = $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)
OBJ = $(CSRC:.c=.o)

CFLAGS := -std=gnu11
LDFLAGS = -lm

mach0 : $(OBJ)
	gcc $^ $(LDFLAGS) -o $@

utest : CFLAGS += -DDO_UNITTEST
utest : clean mach0
	./mach0

vtest : CFLAGS += -DDO_VTEST
vtest : clean mach0
	./mach0 > verification_output.txt

%.o : %.c
	gcc $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o
