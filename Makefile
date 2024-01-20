LIBS    = -lm
CC		= gcc

DEBUG_CFLAGS = -Wall -Wextra# -pg 
RELEASE_CFLAGS = -Wall -Wextra -O3

REQ = main.c matrice.c ml_network.c mnist_parser.c

ml: $(REQ) 
	$(CC) $(DEBUG_CFLAGS) -o $@ $^ $(LIBS)

release: $(REQ)
	$(CC) $(RELEASE_CFLAGS) -o $@ $^ $(LIBS)

valgrind:
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./ml
