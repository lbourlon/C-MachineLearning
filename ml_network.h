typedef struct network_st{
    int  nb_layers;
    int* nb_nodes;  // number of nodes per layer
    float*   biases;   // list of biases between each leayer
    float*** weights;  // list of weight matrices between each layer
} network;

network malloc_network(int nb_layers, int* nb_nodes);
void free_network(network net, int nb_layers, int* nb_nodes);

float sigmoid(float z_value);
