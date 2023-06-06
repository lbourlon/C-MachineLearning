#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"

typedef struct network_st{
    int  nb_layers;
    int* nb_nodes;     // number of nodes per layer
    float**   biases;  // list of bias      lists between each layer
    float*** weights;  // list of weight matrices between each layer

    float** z; //list of weighted activation (pre sigmoid)
    float** a; //list of activations (post sigmoid)
} network;

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

network* malloc_network(int nb_layers, int* nb_nodes)
{
    network *net = malloc(sizeof(network));

    net->nb_layers  = nb_layers;
    net->nb_nodes   = malloc((nb_layers) * sizeof(float));

    net->z = malloc((nb_layers) * sizeof(float*));
    net->a = malloc((nb_layers) * sizeof(float*));

    net->biases  = malloc((nb_layers - 1) * sizeof(float*));
    net->weights = malloc((nb_layers - 1) * sizeof(float**));

    for (int l= 0; l < nb_layers; l++) net->nb_nodes[l] = nb_nodes[l];


    for (int layer = 0; layer < nb_layers; layer++)
    {
        int cols  = nb_nodes[layer];
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
    for (int layer = 0; layer < net->nb_layers; layer++)
    {
        if (layer < net->nb_layers - 1) {
            int rows = net->nb_nodes[layer + 1];
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
    free(net->nb_nodes);
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

void feed_forward(network* net, float* input_vector)
{
    int cols  = net->nb_nodes[0];
    for (int c = 0; c < cols; c++) net->a[0][c] = input_vector[c];

    int rows = 0;
    for(int layer = 0; layer < net->nb_layers - 1; layer++)
    {
        cols  = net->nb_nodes[layer];
        rows = net->nb_nodes[layer + 1];

        M_times_a_plus_b(net->weights[layer],
                         net->a[layer],
                         net->biases[layer],
                         net->z[layer+1],
                         rows,
                         cols);

        for (int r = 0; r < rows; r++){
            net->a[layer + 1][r] = sigmoid(net->z[layer + 1][r]);
        }

        // printf("\nActivation %d : \n", layer);
        // print_vect(net->a[layer+1], rows);
    }
}


// Ugly to get proof of concept going
void backprop(network* net, float** in_vectors, float** expected_out, size_t iter)
{
    int nbl = net->nb_layers;
    int L = nbl - 1;
    int outs = net->nb_nodes[L];


    // where nabla C is the partial derivative of C by a

    float cost = 0, tmp = 0;

    float* nabla_a_C = malloc_vect(outs);
    float* error = malloc_vect(outs);

    for (size_t n = 0; n < iter; n++)
    {
        feed_forward(net, in_vectors[n]);

        for (int o = 0; o < outs; o++) {
            tmp = expected_out[n][o] - net->a[L][o];
            cost += tmp * tmp;

            nabla_a_C[o] += net->a[L][o] + expected_out[n][o];
        }
    }

    cost /= 2*iter;


    // error da = nabla C ⊙  󰘫'(-z)
    for (int o = 0; o < outs; o++)
    {
        nabla_a_C[o] /= iter; 
        printf("nabla C is %.5f\n", nabla_a_C[o]);
        error[o] = nabla_a_C[o] * sigmoid_deriv(net->z[L][o]);
    }

    printf("Cost is %.5f\n", cost);

    printf("Error : \n");
    print_vect(error, outs);
}

// Fisher–Yates_shuffle
void shuffle(tuple* list, int size)
{
    tuple temp;
    for(int i = size - 1; i > 0; i--)
    {
        int j = (rand() % i) + 1;

        temp    = list[j];
        list[j] = list[i];
        list[i] = temp;
    }
}


// Add test data later
// void stochastic_gradient_descent(tuple* training_data, int epochs, int td_size, int mini_batch_size)
// {
//     for (int e = 0; e < epochs; e++)
//     {
//         shuffle(training_data, td_size);
//         int nb_batches = td_size / mini_batch_size;
//         
//         for (int nb = 0; nb_batches; nb++)
//         {
//             tuple current_mini_batch = training_data[nb * mini_batch_size];
//             // Then I can pass this subset of training data (minibatch)
//             // along with eta (the learning rate)
//             // All of this is useless before I implement backpropagation
//         }
//     
//     }
//     
// }


void print_network(network* net)
{
    printf("Nodes per layer vector : \n");
    for (int layer = 0; layer < net->nb_layers; layer++) {
        printf("| %3d ", net->nb_nodes[layer]);
        printf("|\n");
    }
    printf("\n");

    printf("Biases List : \n");
    for (int layer = 0; layer < net->nb_layers -1; layer++)
    {
        int rows = net->nb_nodes[layer + 1];
        print_vect(net->biases[layer], rows);
    }

    printf("Weight Matrices\n");
    for (int layer = 0; layer < net->nb_layers -1; layer++)
    {
        int cols = net->nb_nodes[layer];
        int rows = net->nb_nodes[layer + 1];
        print_mat(net->weights[layer], rows, cols);
    }
}
