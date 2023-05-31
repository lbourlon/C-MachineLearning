typedef struct network_st{
    int  nb_layers;
    int* nb_nodes;     // number of nodes per layer
    float**   biases;  // list of bias      lists between each layer
    float*** weights;  // list of weight matrices between each layer

    float** w_activation; //list of weighted activation (pre sigmoid)
    float** s_activation; //list of activations (post sigmoid)
} network;

network* malloc_network(int nb_layers, int* nb_nodes);
void free_network(network* net);

void feed_forward(network* net, float* input_vector);

void print_network(network* net);
