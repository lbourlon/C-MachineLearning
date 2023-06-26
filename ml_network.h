#include <stddef.h>
#include <stdint.h>

typedef struct network_st {
    int layers;
    int* nodes;         // number of nodes per layer
    double** biases;     // list of bias      lists between each layer
    double*** weights;   // list of weight matrices between each layer

    int size_in;
    int size_out;
} network;

typedef struct activations_st {
    double** a;
    double** z;      // z[0] should always  = 0
    double** error;  // same for error[0]

    double* nw_input;
    double* nw_output;
    double* last_z;
    double* last_error;
} activations;



network* network_malloc(int nb_layers, int* nb_nodes);
void network_free(network* net);

activations* activations_malloc(network* net, double* input_vector);
void activations_free(activations* act, int layers);
void activations_print(network* net, activations* act, int which);

void nw_mini_batch(network* net, double** images, uint8_t* labels, size_t batch_size);
void network_feed_forward(network* net, activations* act);

void print_network(network* net);
