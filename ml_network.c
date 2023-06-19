#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"

typedef struct network_st{
    int layers;
    int* nodes;     // number of nodes per layer
    float**   biases;  // list of bias      lists between each layer
    float*** weights;  // list of weight matrices between each layer

    float** z; //list of weighted activation (pre sigmoid)
    float** a; //list of activations (post sigmoid)
} network;

typedef struct activations_st{
    int layers;
    float** a;
    float** z;      // z[0] should always  = 0
    float** error;  // same for error[0]
} activations;

typedef struct tuple_st{
    float input;           // number of nodes per layer
    float desired_output;  // number of nodes per layer
} tuple;


typedef struct cost_data_st{
    size_t size_out;
    float* actual_output;
    float* desired_output;
} cost_data;

cost_data* malloc_cost_data(size_t size_out, size_t n)
{
    cost_data* cdt = malloc(n * sizeof(cost_data));

    cdt->size_out = size_out;
    cdt->actual_output = malloc(size_out * sizeof(float*));
    cdt->desired_output = malloc(size_out * sizeof(float*));

    return cdt;
}


void free_cost_data(cost_data* cdt, size_t n){
    for (size_t x = 0; x < n; x++) {
        free(cdt[x].desired_output);
        free(cdt[x].desired_output);
    }
    free(cdt);
}

float cost_function(cost_data* cdt, size_t nb_training_examples)
{
    float sum = 0.0;
    for (size_t x = 0; x < nb_training_examples; x++)
    {
        float* temp = malloc(cdt->size_out * sizeof(float));
        for (size_t i = 0; i < cdt->size_out; i++)
        {
            temp[i] = cdt->desired_output[i] - cdt->actual_output[i];
        }
        sum += vect_norm(temp, cdt->size_out);
        free(temp);
    }

    return sum / (2*nb_training_examples);
}

activations* malloc_activations(int layers, float* input_vector,  int* nb_nodes)
{
    activations* act = malloc(sizeof(activations));

    act->z = calloc((layers), sizeof(float*));
    act->a = calloc((layers), sizeof(float*));
    act->error = calloc((layers), sizeof(float*));

    act->layers = layers;

    for (int layer = 0; layer < act->layers; layer++)
    {
        int cols  = nb_nodes[layer];

        act->z[layer]  = malloc_vect(cols);
        act->a[layer]  = malloc_vect(cols);
        act->error[layer]  = malloc_vect(cols);

    }

    int cols  = nb_nodes[0];
    for (int c = 0; c < cols; c++)
        act->a[0][c] = input_vector[c];

    return act;
}

void free_activations(activations* act)
{
    for (int layer = 0; layer < act->layers; layer++){
        free(act->z[layer]);
        free(act->a[layer]);
        free(act->error[layer]);
    }

    free(act->z);
    free(act->a);
    free(act->error);

    free(act);
}


network* malloc_network(int nb_layers, int* nb_nodes)
{
    network* net = malloc(sizeof(network));

    net->layers  = nb_layers;
    net->nodes  = malloc((nb_layers) * sizeof(float));

    net->z = malloc((nb_layers) * sizeof(float*));
    net->a = malloc((nb_layers) * sizeof(float*));

    net->biases  = malloc((nb_layers - 1) * sizeof(float*));
    net->weights = malloc((nb_layers - 1) * sizeof(float**));

    for (int l= 0; l < nb_layers; l++) net->nodes[l] = nb_nodes[l];

    for (int layer = 0; layer < nb_layers; layer++)
    {
        int cols = nb_nodes[layer];
        int rows = nb_nodes[layer + 1];

        net->z[layer]  = malloc_vect(cols);
        net->a[layer]  = malloc_vect(cols);

        if (layer < nb_layers - 1) {
            net->biases[layer]  = malloc_vect(rows);
            net->weights[layer] = malloc_mat(rows, cols);

            fill_vect(net->biases[layer], rows);
            fill_mat(net->weights[layer], rows, cols);
        }

    }

    return net;
}

void free_network(network* net)
{
    for (int layer = 0; layer < net->layers; layer++)
    {
        if (layer < net->layers - 1) {
            int rows = net->nodes[layer + 1];
            free(net->biases[layer]);
            free_mat(net->weights[layer], rows);
        }

        free(net->z[layer]);
        free(net->a[layer]);

    }
    free(net->z);
    free(net->a);
    free(net->biases);
    free(net->weights);
    free(net->nodes);
    free(net);
}



/* a' = sigma(wa + b) */
float sigmoid(float z){
    return 1.0 / (1.0 + exp(-z));
}


float sigmoid_deriv(float z){
    
    float sig = sigmoid(z);
    return sig * (1.0 - sig);
}

void feed_forward(network* net, activations* act)
{
    int rows = 0, cols = 0;
    for(int layer = 0; layer < net->layers - 1; layer++)
    {
        cols  = net->nodes[layer];
        rows = net->nodes[layer + 1];

        M_times_a_plus_b(net->weights[layer],
                         act->a[layer],
                         net->biases[layer],
                         act->z[layer+1],
                         rows,
                         cols);

        for (int r = 0; r < rows; r++){
            act->a[layer + 1][r] = sigmoid(act->z[layer + 1][r]);
        }

        // print_vect(act->a[layer+1], rows);
    }
}

/* Multiplies the matrix in the wrong order as if it were transposing it
 * but without shifting any memory arround.
 * As such, this function expects a 
 * error[rows]      // error vector
 * mat[rows][cos]   // weight matrix
 * z[rows]          // weighted activations
 * prior_error[cols]   // output vector (error of l-1)
 * */
void backprop_step(float** mat, float* error, float* z,  float* prior_error, int rows, int cols)
{
    for (int c = 0; c < cols; c++){
        prior_error[c] = 0;
        for (int r = 0; r < rows; r++) {
            prior_error[c] += mat[r][c] * error[r];
        }

        prior_error[c] *= sigmoid_deriv(z[c]);
    }
}

void backprop(network* net, float** in_vectors, float** expected_out, size_t iter)
{
    int layers = net->layers;
    int rows = net->nodes[layers-1];

    activations** acts = malloc(iter * sizeof(activations*));

    float Cost = 0;
    // Calculates the first error vectors
    for (size_t n = 0; n < iter; n++)
    {
        acts[n] = malloc_activations(layers, in_vectors[n], net->nodes);
        activations* act = acts[n];

        feed_forward(net, act);

        float Cost_x = 0, tmp = 0;
        for (int j = 0; j < rows; j++) // See readme for equation (not strictly needed for backprop)
        {
            tmp = expected_out[n][j] - act->a[layers - 1][j];
            Cost_x += (tmp * tmp) / 2;
            Cost += Cost_x;
        }
        // printf("iteration %zu | ", n);
        // printf("C_x : %.8f\n", Cost_x);

        // Calculates first error
        for (int j = 0; j < rows; j++) // see readme (TODO)
        {
            float da_Cx = act->a[layers - 1][j] - expected_out[n][j];
            act->error[layers-1][j] = da_Cx * sigmoid_deriv(act->z[layers - 1][j]);
        }

        // Backpropagates error
        for(int layer = layers - 2; layer > 0; layer--)
        { 
            int cols = net->nodes[layer];
            int rows = net->nodes[layer + 1];

            backprop_step(net->weights[layer],
                          act->error[layer + 1],
                          act->z[layer],
                          act->error[layer],
                          rows,
                          cols);
        }
    }

    printf("Cost %.8f\n", Cost);

    const float learning_rate = 5.0;
    const float learning_coeff = learning_rate / iter;

    // Gradient Descent
    for(int layer = 0; layer < net->layers - 2; layer++)
    {
        int cols = net->nodes[layer];
        int rows = net->nodes[layer + 1];

        for (int r = 0; r < rows; r++) {
            float d_err = 0;
            for (size_t x = 0; x < iter; x++)
            {
                d_err    += acts[x]->error[layer][r];
            }

            net->biases[layer][r]     -= learning_coeff * d_err;

            float d_weight = 0;
            for (int c = 0; c < cols; c++)
            {
                for (size_t x = 0; x < iter; x++) {
                    d_weight += acts[x]->error[layer+1][r] * acts[x]->a[layer][c];
                }
                net->weights[layer][r][c] -= learning_coeff * d_weight;
            }
        }
    }

    for (size_t n = 0; n < iter; n++)
        free_activations(acts[n]);
    free(acts);
}

// Fisher–Yates shuffle
void shuffle(int* list, int size)
{
    int temp;
    for(int i = size - 1; i > 0; i--)
    {
        int j = (rand() % i) + 1;

        temp    = list[j];
        list[j] = list[i];
        list[i] = temp;
    }
}


void print_network(network* net)
{
    printf("Nodes per layer vector : \n");
    for (int layer = 0; layer < net->layers; layer++) {
        printf("| %3d ", net->nodes[layer]);
        printf("|\n");
    }
    printf("\n");

    printf("Biases List : \n");
    for (int layer = 0; layer < net->layers -1; layer++)
    {
        int rows = net->nodes[layer + 1];
        print_vect(net->biases[layer], rows);
    }

    printf("Weight Matrices\n");
    for (int layer = 0; layer < net->layers -1; layer++)
    {
        int cols = net->nodes[layer];
        int rows = net->nodes[layer + 1];
        print_mat(net->weights[layer], rows, cols);
    }
}
