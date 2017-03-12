CCFLAGS = -O3
CC = g++

C = main.cpp rng.cpp history.cpp simfunctions.cpp
H = smctc.hh sampler.hh particle.hh moveset.hh history.hh rng.hh

PFC = sirTest.cpp pffuncs.cpp multinomial.c
SMCC = rng.cpp history.cpp smc-exception.cpp

PFO = sirTest.o pffuncs.o  multinomial.o
SMCO = rng.o history.o smc-exception.o
PFH = pffuncs.hh 

all: pf

pf: $(PFC) $(PFH)
	$(CC) $(CCFLAGS) -c $(PFC) $(SMCC)
	$(CC) $(PFO) $(SMCO) -opf

clean:
	-rm -f *.o
	-rm -f *~
	-rm -f pf
