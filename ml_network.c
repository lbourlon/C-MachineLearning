#include <math.h>
#include <stdlib.h>

// class Network(object):
//
//     def __init__(self, sizes):
//         self.num_layers = len(sizes)
//         self.sizes = sizes
//         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
//         self.weights = [np.random.randn(y, x) 
//                         for x, y in zip(sizes[:-1], sizes[1:])]

typedef struct network_st{
    int num_layers;
    int sizes;
    // float* biases_list; // list of biases between each leayer
    // float*** weight_matrices; // list of weight matrices between each layer
} network;

//typedef struct network_st network;

network create_network(){
    network a = (network) {1, 2};

    return a;
}


void free_network(network net){
    return;
}

float sigmoid(float z_value){
    return 1 / (1 + expf(z_value));
}

