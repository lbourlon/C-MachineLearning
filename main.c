#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
#include <stdlib.h>
//#include "ingest.h"

// Check how to add biases

#define NB_LAYERS 4
#define NB_INPUTS 3
#define NB_OUTPUTS 2

int main(){

    int nb_nodes[NB_LAYERS] = {NB_INPUTS, 6, 3, NB_OUTPUTS};
    network net = malloc_network(NB_LAYERS, nb_nodes);
    // printf("Network has %d layers and %d things\n", net.sizes, net.num_layers);

    //print_network(net);
    
    float in_vector[NB_INPUTS] = {0.2, 0.4, 0.9};
    
    float* out_vector = feed_forward(net, in_vector);
    print_vect(out_vector, NB_OUTPUTS);


    free(out_vector);
    free_network(net, NB_LAYERS, nb_nodes);
    return 0;
}

