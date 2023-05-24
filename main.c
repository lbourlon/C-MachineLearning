#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
//#include "ingest.h"

#define NB_LAYERS 3

int main(){
    printf("Vector of sizes: \n");

    int nb_nodes[NB_LAYERS] = {3, 2, 3};
    network net = malloc_network(NB_LAYERS, nb_nodes);

    // printf("Network has %d layers and %d things\n", net.sizes, net.num_layers);

    printf("nodes per layer vector : \n");
    for (int layer = 0; layer < net.nb_layers; layer++) {
        printf("| %.0d ", net.nb_nodes[layer]);
        printf("|\n");
    }
    printf("\n");

    printf("Biases List : \n");
    print_vect(net.biases, net.nb_layers);


    printf("Weight Matrices\n");
    for (int layer = 0; layer < net.nb_layers -1; layer++) {

        int cols  = net.nb_nodes[layer];
        int lines = net.nb_nodes[layer + 1];
        print_mat(net.weights[layer], lines, cols);
    }

    return 0;
}

