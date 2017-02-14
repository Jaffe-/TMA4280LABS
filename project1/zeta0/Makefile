CSRC = $(wildcard $(m)*.c)
HEADERS = $(wildcard $(m)*.h)
COMP_LINE = gcc $(CSRC) $(CFLAGS) $(LDFLAGS) -o $@

CFLAGS := -std=gnu11
LDFLAGS = -lm

zeta0 : $(CSRC) $(HEADERS) 
	$(COMP_LINE)

utest : clean
	$(COMP_LINE) -DDO_UNITTEST
	./utest

vtest : clean
	$(COMP_LINE) -DDO_VTEST
	./vtest > verification_output.txt

clean:
	rm -rf *.o
