#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"

// class Network(object):
//
//     def __init__(self, sizes):
//         self.nb_layers = len(sizes)
//         self.sizes = sizes
//         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
//         self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

typedef struct network_st{
    int  nb_layers;
    int* nb_nodes;  // number of nodes per layer
    float*   biases;   // list of biases between each leayer
    float*** weights;  // list of weight matrices between each layer
} network;


float*** malloc_weights(int nb_layers, int* nb_nodes)
{
    float*** weights = calloc(nb_layers, sizeof(float**));

    for (int layer = 0; layer < nb_layers -1; layer++)
    {
        int cols  = nb_nodes[layer];
        int lines = nb_nodes[layer + 1];

        weights[layer] = malloc_mat(lines, cols);

        fill_mat(weights[layer], lines, cols);
    }
    
    return weights;
}

void free_weights(float*** weights, int nb_layers, int* nb_nodes)
{
    for (int layer = 0; layer < nb_layers - 1; layer++)
    {
        int lines = nb_nodes[layer + 1];
        free_mat(weights[layer], lines);
    }
    free(weights);
}

network malloc_network(int nb_layers, int* nb_nodes)
{
    network net = (network){
        nb_layers,
        nb_nodes,
        malloc_vect(nb_layers),
        malloc_weights(nb_layers, nb_nodes)
    };

    return net;
}

void free_network(network net, int nb_layers, int* nb_nodes){
    free_vect(net.biases);
    free_weights(net.weights, net.nb_layers, nb_nodes);
    return;
}

float sigmoid(float z_value){
    return 1 / (1 + expf(z_value));
}

