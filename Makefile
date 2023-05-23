LIBS    = "-lm"
CC 		= "gcc"

DEBUG_CFLAGS = -Wall -g -O0
CFLAGS = -O3

ml: main.c matrice.c ml_network.c ingest.c
	$(CC) $(CFLAGS) -o ml.out main.c matrice.c ml_network.c $(LIBS)

debug: main.c matrice.c ingest.c
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -o prog main.c  matrice.c $(LIBS)
