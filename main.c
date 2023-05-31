#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <time.h>
//#include "ingest.h"

#define NB_OUTPUTS 4
#define NB_INPUTS 6
#define NB_LAYERS 7

// To Achieve Backpropagation I need :
// [x] Keep track of the "activation" "weighted_activat" given feed-forward
// [ ]Calculate my cost function at the end given an expected output
// [ ]Compute it's error and start backpropagation
//    (easier said than understood)

int main(){
    srand48(time(NULL));

    int nb_nodes[NB_LAYERS] = {NB_INPUTS, 12, 32, 7, 34, 24, NB_OUTPUTS};
    float in_vector[NB_INPUTS] = {0.2, 0.5, 0.7, 0.92, 0.45, 0.37};

    network* net = malloc_network(NB_LAYERS, nb_nodes);
    print_network(net);

    feed_forward(net, in_vector);

    free_network(net);
    return 0;
}

