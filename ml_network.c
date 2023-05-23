#include <math.h>
#include <stdlib.h>
#include "matrice.h"

// class Network(object):
//
//     def __init__(self, sizes):
//         self.num_layers = len(sizes)
//         self.sizes = sizes
//         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
//         self.weights = [np.random.randn(y, x) 
//                         for x, y in zip(sizes[:-1], sizes[1:])]

typedef struct network_st{
    int  num_layers;
    int* layer_sizes;
    float*   biases_list; // list of biases between each leayer
    float*** weight_matrices; // list of weight matrices between each layer
} network;

//typedef struct network_st network;

/* Mallocs our list of weight matrices */
float*** malloc_weight_matrices(int num_layers, int lines, int cols){
    float*** weight_matrices = calloc(num_layers, sizeof(float**));
    for (int layer = 0; layer < num_layers; ++num_layers) {
        weight_matrices[layer] = malloc_mat(lines, cols);
    }
    
    return weight_matrices;
}

void free_weight_matrices(float*** weight_matrices, int num_layers, int lines){
    for (int layer = 0; layer < num_layers; ++num_layers) {
        free_mat(weight_matrices[layer], lines);
    }
    free(weight_matrices);
}


network create_network(int num_layers, int* sizes){
    network net = (network){
        num_layers,
        sizes,
        malloc_vect(num_layers),
        malloc_weight_matrices(num_layers, 3, 3)
    };

    return net;
}


void free_network(network net){
    free_vect(net.biases_list);
    free_weight_matrices(net.weight_matrices, net.num_layers, 3);

    return;
}

float sigmoid(float z_value){
    return 1 / (1 + expf(z_value));
}

