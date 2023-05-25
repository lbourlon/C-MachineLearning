#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"

typedef struct network_st
{
    int  nb_layers;
    int* nb_nodes;     // number of nodes per layer
    float*   biases;   // list of biases between each leayer
    float*** weights;  // list of weight matrices between each layer
} network;

/*
 *
             (a0)
             (a1)
( w_00 w_01 )     (x1)
( w_10 w_11 )  =  (x2)
( w_20 w_21 )     (x3)
( w_30 w_31 )     (x4)

 *
 * */

float*** malloc_weights(int nb_layers, int* nb_nodes)
{
    float*** weights = malloc(nb_layers * sizeof(float**));

    for (int layer = 0; layer < nb_layers - 1; layer++)
    {
        int cols  = nb_nodes[layer];
        int lines = nb_nodes[layer + 1];

        weights[layer] = malloc_mat(lines, cols);

        fill_mat(weights[layer], lines, cols);
    }

    return weights;
}


network malloc_network(int nb_layers, int* nb_nodes)
{
    network net = (network)
    {
        nb_layers,
        nb_nodes,
        malloc(nb_layers * sizeof(float)),
        malloc_weights(nb_layers, nb_nodes)
    };

    fill_vect(net.biases, net.nb_layers);

    return net;
}

void free_network(network net)
{
    for (int layer = 0; layer < net.nb_layers - 1; layer++)
    {
        int lines = net.nb_nodes[layer + 1];
        free_mat(net.weights[layer], lines);
    }
    free(net.weights);
    free(net.biases);

    return;
}

float sigmoid(float z_value)
{
    return 1 / (1 + expf(z_value));
}

void print_network(network net)
{
    printf("nodes per layer vector : \n");
    for (int layer = 0; layer < net.nb_layers; layer++) {
        printf("| %d ", net.nb_nodes[layer]);
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
}
