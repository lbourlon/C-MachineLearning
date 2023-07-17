LIBS    = -lm
CC		= gcc

CFLAGS =  -O3
DEBUG_CFLAGS = -Wall -Wextra -pg 

REQ = main.c matrice.c ml_network.c mnist_parser.c

ml: $(REQ) 
	$(CC) $(DEBUG_CFLAGS) -o $@ $^ $(LIBS)

release: $(REQ)
	$(CC) $(DEBUG_CFLAGS) $(CFLAGS) -o $@ $^ $(LIBS)

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./ml
