#include <stddef.h>
typedef struct network_st{
    int  nb_layers;
    int* nb_nodes;     // number of nodes per layer
    float**   biases;  // list of bias      lists between each layer
    float*** weights;  // list of weight matrices between each layer

    float** w_activation; //list of weighted activation (pre sigmoid)
    float** s_activation; //list of activations (post sigmoid)
} network;

typedef struct cost_data_st{
    size_t size_out;
    float* actual_output;
    float* desired_output;
} cost_data;

network* malloc_network(int nb_layers, int* nb_nodes);
void free_network(network* net);

void feed_forward(network* net, float* input_vector);

void print_network(network* net);

cost_data* malloc_cost_data(size_t size_out, size_t n);
void free_cost_data(cost_data* cdt, size_t n);
float cost_function(cost_data* cdt, size_t nb_training_examples);
