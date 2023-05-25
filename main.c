#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
//#include "ingest.h"

#define NB_LAYERS 4

int main(){
    printf("Vector of sizes: \n");

    int nb_nodes[NB_LAYERS] = {8, 6, 3, 2};
    network net = malloc_network(NB_LAYERS, nb_nodes);

    // printf("Network has %d layers and %d things\n", net.sizes, net.num_layers);

    print_network(net);


    free_network(net);
    return 0;
}

