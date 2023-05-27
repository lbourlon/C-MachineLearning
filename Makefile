LIBS    = -lm
CC 		= gcc

CFLAGS =  -O3
DEBUG_CFLAGS =-Wall -g -Werror


ml: main.c matrice.c ml_network.c ingest.c
	$(CC) $(DEBUG_CFLAGS) -o ml.out main.c matrice.c ml_network.c $(LIBS)

# release: main.c matrice.c ml_network.c ingest.c
# 	$(CC) $(CFLAGS) -o release.out main.c matrice.c ml_network.c $(LIBS)

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./ml.out
