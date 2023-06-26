#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"

typedef struct network_st {
    int layers;
    int* nodes;         // number of nodes per layer
    double** biases;     // list of bias between each layer
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


void print_network(network* net)
{
    printf("Nodes per layer vector : \n");
    for (int l = 0; l < net->layers; l++) {
        printf("| %3d ", net->nodes[l]);
        printf("|\n");
    }
    printf("\n");

    printf("Biases List : \n");
    for (int l = 1; l < net->layers; l++)
    {
        int rows = net->nodes[l];
        print_vect(net->biases[l], rows);
    }

    printf("Weight Matrices\n");
    for (int l = 1; l < net->layers; l++)
    {
        int cols = net->nodes[l-1];
        int rows = net->nodes[l];
        print_mat(net->weights[l], rows, cols);
    }
}

void activations_print(network* net, activations* act, int which) {
    char* yo[] = {"act", "zed", "err"};
    printf("----------------------------------------\n");

    if(which == 0){
        for (int l = 0; l < net->layers; l++) {
            int rows = net->nodes[l];
            printf("%s[%d] |", yo[which], l);

            for (int r = 0; r < rows; r++) printf(" %.4f", act->a[l][r]);
            printf("|\n");
        }
    }

    if (which == 1) {
        printf("%s[0] | %.4f|\n", yo[which], act->z[0][0]);

        for (int l = 1; l < net->layers; l++) {
            int rows = net->nodes[l];

            printf("%s[%d] |", yo[which], l);
            for (int r = 0; r < rows; r++) printf(" %.4f", act->z[l][r]);
            printf("|\n");
        }
    }
    if (which == 2){
        printf("%s[0] | %.4f|\n", yo[which], act->error[0][0]);
        for (int l = 1; l < net->layers; l++) {
            int rows = net->nodes[l];

            printf("%s[%d] |", yo[which], l);
            for (int r = 0; r < rows; r++) printf(" %.9f", act->error[l][r]);
            printf("|\n");
        }
    }


    printf("\n----------------------------------------\n");
}

activations* activations_malloc(network* net, double* input_vector)
{
    activations* act = malloc(sizeof(activations));

    act->z = calloc((net->layers), sizeof(double*));
    act->a = calloc((net->layers), sizeof(double*));
    act->error = calloc((net->layers), sizeof(double*));

    act->a[0]  = malloc_vect(net->nodes[0]);
    act->z[0]  = malloc_vect(1);
    act->error[0]  = malloc_vect(1);

    for (int l = 1; l < net->layers; l++)
    {
        int cols  = net->nodes[l];

        act->z[l]  = malloc_vect(cols);
        act->a[l]  = malloc_vect(cols);
        act->error[l]  = malloc_vect(cols);

    }

    int cols  = net->nodes[0];
    for (int c = 0; c < cols; c++)
        act->a[0][c] = input_vector[c];

    act->nw_input = act->a[0];
    act->nw_output = act->a[net->layers - 1];
    act->last_z = act->z[net->layers - 1];
    act->last_error = act->error[net->layers -1];

    return act;
}

void activations_free(activations* act, int layers)
{
    for (int l = 0; l < layers; l++){
        free(act->z[l]);
        free(act->a[l]);
        free(act->error[l]);
    }

    free(act->z);
    free(act->a);
    free(act->error);

    free(act);
}


network* network_malloc(int layers, int* nodes)
{
    network* net = malloc(sizeof(network));

    net->layers = layers;
    net->size_in = nodes[0];
    net->size_out = nodes[layers - 1];
    net->nodes = malloc((layers) * sizeof(double));

    net->biases = malloc(layers * sizeof(double*));
    net->weights = malloc(layers * sizeof(double**));

    for (int l= 0; l < layers; l++)
        net->nodes[l] = nodes[l];

    net->biases[0]  = malloc_vect(1);
    net->weights[0] = malloc_mat(1, 1);

    for (int l = 1; l < layers; l++)
    {
        int cols = nodes[l-1];
        int rows = nodes[l];

        net->biases[l]  = malloc_vect(rows);
        net->weights[l] = malloc_mat(rows, cols);

        fill_vect(net->biases[l], rows);
        fill_mat(net->weights[l], rows, cols);
    }

    return net;
}

void network_free(network* net)
{
    free_mat(net->weights[0], 1);
    free(net->biases[0]);
    
    for (int l = 1; l < net->layers; l++)
    {
        int rows = net->nodes[l];
        free_mat(net->weights[l], rows);
        free(net->biases[l]);
    }
    free(net->biases);
    free(net->weights);
    free(net->nodes);
    free(net);
}

/* a' = sigma(wa + b) */
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double d_sigmoid(double z) {
    double sig = sigmoid(z);
    return sig * (1.0-sig);
}

// looks good
void network_feed_forward(network* net, activations* act)
{
    int rows = 0, cols = 0;
    for(int l = 1; l < net->layers; l++)
    {
        cols = net->nodes[l-1];
        rows = net->nodes[l];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                act->z[l][r] += net->weights[l][r][c] * act->a[l-1][c]; 
            }
            act->a[l][r] = sigmoid(act->z[l][r] + net->biases[l][r]);
        }
    }
}


/* Multiplies the matrix in the wrong order as if it were transposing it but without shifting any memory arround. Might be slower, because of 
 * non contiguous jumps in memory, but easier than transposing.
 * */

// Looks good
void backprop_error(network* net, activations* act)
{
    for(int l = net->layers - 2; l > 0; l--)
    { 
        int cols = net->nodes[l];
        int rows = net->nodes[l+1];

        for (int c = 0; c < cols; c++) {
            for (int r = 0; r < rows; r++) {
                act->error[l][c] += net->weights[l+1][r][c] * act->error[l+1][r];
            }
            act->error[l][c] *= d_sigmoid(act->z[l][c]);
        }
    }
}

// looks good
void nw_gradient_descent(network* net, activations** acts, double learning_coeff, size_t batch_size) {
    for(int l = net->layers - 1; l > 0; l--)
    {
        int cols = net->nodes[l-1];
        int rows = net->nodes[l];

        for (int r = 0; r < rows; r++) {
            double d_weight = 0;
            for (int c = 0; c < cols; c++)
            {
                for (size_t x = 0; x < batch_size; x++) 
                {
                    d_weight += acts[x]->error[l][r] * acts[x]->a[l-1][c];
                }
                net->weights[l][r][c] -= learning_coeff * d_weight;
            }
            // printf("d_err : %0.3f\n",d_weight);
        }

        for (int r = 0; r < rows; r++) {
            double d_err = 0;
            for (size_t x = 0; x < batch_size; x++) {
                d_err += acts[x]->error[l][r];
            }
            net->biases[l][r] -= learning_coeff * d_err;
        }
    }
}

void nw_mini_batch(network* net, double** images, uint8_t* labels, size_t batch_size)
{
    activations** acts = malloc(batch_size * sizeof(activations*));

    double Cost = 0;
    for (size_t x = 0; x < batch_size; x++)
    {
        acts[x] = activations_malloc(net, images[x]);
        activations* act = acts[x];

        // 1st step
        network_feed_forward(net, act);

        double* expected_out = calloc(net->size_out, sizeof(double));
        expected_out[(int)labels[x]] = 1.0;

        double tmp_f = 0;
        for (int k = 0; k < net->size_out; k++)
        {
            tmp_f = expected_out[k] - act->nw_output[k];
            Cost += (tmp_f * tmp_f);

            // 2nd step // this is good
            double da_Cx = act->nw_output[k] - expected_out[k];
            act->last_error[k] = da_Cx * d_sigmoid(act->last_z[k]);

            //printf("da_Cx %f, last_error : %f\n", da_Cx, act->last_error[k]);
        }

        print_vect(act->nw_output, net->size_out);
        print_vect(expected_out, net->size_out);
        print_vect(act->last_error, net->size_out);

        // for (int m = 0; m < net->size_out; m++) {
        //
        //     printf("err_1 : %f | %f : err_2\n",act->error[net->layers-1][m], act->last_error[m]);
        // }
        //
        // activations_print(net, act, 2);

        // 3rd step  // issue here
        // activations_print(net, act,1);
        backprop_error(net, act);
        // activations_print(net, act,2);

        free(expected_out);

        // printf("iteration %zu | ", x);
        // printf("C_x : %.8f\n",tmp_f * tmp_f);
    }

    Cost /= (batch_size);
    printf("Cost %.6f\n", Cost);

    const double learning_rate = 0.1;
    const double learning_coeff = learning_rate / batch_size;

    // print_network(net);
    nw_gradient_descent(net, acts, learning_coeff, batch_size);
    // print_network(net);
    // activations_print(net, acts[0], 2);


    // for (int i = 0; i < net->size_in; i++){
    //     printf("a : %f | %f : i\n", acts[0]->a[0][i], images[0][i]);
    // } 
    // for (int i = 0; i < net->nodes[1]; i++) {
    //     printf("z : %f | %f : b\n", acts[0]->z[0][i], net->biases[0][i]);
    // }

    // for (int i = 0; i < net->size_in; i++){
    //     printf("e : %f | %f : i\n", acts[0]->error[0][i], images[0][i]);
    // } 


    for (size_t x = 0; x < batch_size; x++)
        activations_free(acts[x], net->layers);
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

