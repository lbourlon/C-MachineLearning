typedef struct network_st{
    int num_layers;
    int sizes;
    // float* biases_list; // list of biases between each leayer
    // float*** weight_matrices; // list of weight matrices between each layer
} network;

//typedef struct network_st network;
network create_network();
