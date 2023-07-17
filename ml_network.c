#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrice.h"
#include "mnist_parser.h"

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


void nw_print(network* net)
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

void activations_reset(network* net, activations* act, double* input_vector) {
    int cols  = net->nodes[0];
    memcpy(act->a[0], input_vector, cols * sizeof(double));

    for (int l = 1; l < net->layers; l++) {
        int cols  = net->nodes[l];

        memset(act->z[l], 0, cols);
        memset(act->a[l], 0, cols);
        memset(act->error[l], 0, cols);
    }
}

activations* activations_malloc(network* net, double* input_vector) {
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


network* nw_malloc(int layers, int* nodes)
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

void nw_free(network* net)
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

void nw_feed_forward(network* net, activations* act)
{
    int rows = 0, cols = 0;
    for(int l = 1; l < net->layers; l++)
    {
        cols = net->nodes[l-1];
        rows = net->nodes[l];

        for (int r = 0; r < rows; r++) {
            act->z[l][r] = 0.0;

            for (int c = 0; c < cols; c++) {
                act->z[l][r] += net->weights[l][r][c] * act->a[l-1][c]; 
            }
            act->z[l][r] += net->biases[l][r];
            act->a[l][r] = sigmoid(act->z[l][r]);
        }
    }
}


void backprop_error(network* net, activations* act)
{
    /* Multiplies the matrix in the wrong order as if it were transposing it but without shifting any memory arround. Might be slower, because of 
    non contiguous jumps in memory, but easier than transposing. */
    for(int l = net->layers - 2; l > 0; l--)
    { 
        int cols = net->nodes[l];
        int rows = net->nodes[l+1];

        for (int c = 0; c < cols; c++) {
            act->error[l][c] = 0.0;
            for (int r = 0; r < rows; r++) {
                act->error[l][c] += net->weights[l+1][r][c] * act->error[l+1][r];
            }
            act->error[l][c] *= d_sigmoid(act->z[l][c]);
        }
    }
}

void nw_gradient_descent(network* net, activations** acts, double learning_coeff, size_t batch_size) {
    for(int l = net->layers - 1; l > 0; l--)
    {
        int cols = net->nodes[l-1];
        int rows = net->nodes[l];

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++)
            {
                double d_weight = 0.0;
                for (size_t x = 0; x < batch_size; x++) 
                {
                    d_weight += acts[x]->error[l][r] * acts[x]->a[l-1][c];
                }
                net->weights[l][r][c] -= learning_coeff * d_weight;
            }
        }

        for (int r = 0; r < rows; r++) {
            double d_err = 0.0;
            for (size_t x = 0; x < batch_size; x++) {
                d_err += acts[x]->error[l][r];
            }
            net->biases[l][r] -= learning_coeff * d_err;
        }
    }
}

void nw_mini_batch(network* net, activations** acts, double** images,  uint8_t* labels, size_t batch_size) {

    // double Cost = 0.0;
    for (size_t x = 0; x < batch_size; x++) {
        // (perf) about 5% of execution time of program
        activations_reset(net, acts[x], images[x]);

        // (perf) about 43% of execution time of program 
        // 1st step
        nw_feed_forward(net, acts[x]);

        double* expected_out = calloc(net->size_out, sizeof(double));
        expected_out[(int)labels[x]] = 1.0;

        // Cost calculation
        // double tmp_f = 0.0;
        // for (int k = 0; k < net->size_out; k++)
        // {
        //     tmp_f = expected_out[k] - acts[x]->nw_output[k];
        //     Cost += (tmp_f * tmp_f);
        // }

        // calculation of last error from dC / da
        for (int k = 0; k < net->size_out; k++)
        {
            // 2nd step
            double da_Cx = acts[x]->nw_output[k] - expected_out[k];
            acts[x]->last_error[k] = da_Cx * d_sigmoid(acts[x]->last_z[k]);
        }

        //3rd step
        backprop_error(net, acts[x]);

        free(expected_out);
    }
    // Cost /= (batch_size);
    // printf("Cost %.6f\n", Cost);

    const double learning_rate = 4.95;
    const double learning_coeff = learning_rate / batch_size;

    // (perf) about 50% of execution time of program
    nw_gradient_descent(net, acts, learning_coeff, batch_size);
}

void nw_stochastic_gradient_descent(network* net, const char* images_path, const char* labels_path, int tot_batches, int batch_size, int epochs){

    const int tot_images = batch_size*tot_batches;

    double** images;
    uint8_t* labels;

    parse_labels_and_images(&images, &labels, images_path, labels_path, tot_images, 0);

    activations** acts = malloc(batch_size * sizeof(activations*));

    for (int x = 0; x < batch_size; x++)
        acts[x] = activations_malloc(net, images[x]);

    for (int e = 0; e < epochs; e++) {
        printf("Epoch = [%02d / %02d]\n", e+1, epochs);

        shuffle_imgs_and_lables(labels, images, tot_images);

        for (int s = 0; s < tot_batches; s++) {
            int batch_offset = batch_size * s; 
            nw_mini_batch(net, acts, &images[batch_offset], &labels[batch_offset], batch_size);
        }
    }

    for (int x = 0; x < batch_size; x++)
        activations_free(acts[x], net->layers);

    free(acts);
    free_labels_and_images(images, labels, tot_images);
}

void nw_evaluate(network* net, const char* images_path, const char* labels_path, const int mode){
    int tot_images = 5000;
    int offset = 0;
    if(mode == 1) {
        offset = 4999;
        printf("Checking against the 'hard' images\n");
    } else {
        printf("Checking against the 'easy' images\n");
    }

    double** images;
    uint8_t* labels;

    parse_labels_and_images(&images, &labels, images_path, labels_path, tot_images, offset);


    float success_rate = 0.0;
    for (int img = 0; img < tot_images; img++)
    {
        activations* act = activations_malloc(net, images[img]);
        nw_feed_forward(net, act);

        double max_val = 0.0;
        uint8_t imax = 0;

        for(int o = 0; o < net->size_out; o++) {
            if(max_val <= act->nw_output[o]) {
                max_val = act->nw_output[o];
                imax = o;
            }
        }

        // printf("nw_output / expected : %d / %d\n", imax, test_labels[img]);

        if (imax == labels[img]) {
            success_rate += 1.0;
        } 

        activations_free(act, net->layers);
    }

    printf("Success : [%0.f / %d]", success_rate, tot_images);
    success_rate *= 100.0 / (float)tot_images;
    printf("| Accuracy : %f\n", success_rate);

    free_labels_and_images(images, labels, tot_images);
}

// Fisher–Yates shuffle
void shuffle_double(double* list, int size)
{
    uint8_t temp;
    for(int i = size - 1; i > 0; i--)
    {
        uint8_t j = (rand() % i) + 1;

        temp    = list[j];
        list[j] = list[i];
        list[i] = temp;
    }
}
