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

network* nw_malloc(int nb_layers, int* nb_nodes);
void nw_free(network* net);

activations* activations_malloc(network* net, double* input_vector);
void activations_free(activations* act, int layers);
void activations_print(network* net, activations* act, int which);

void nw_feed_forward(network* net, activations* act);
void nw_stochastic_gradient_descent(network* net, const char* images_path, const char* labels_path, int tot_batches, int batch_size, int epochs);

void nw_evaluate(network* net, const char* images_path, const char* labels_path, const int mode);
void nw_print(network* net);
