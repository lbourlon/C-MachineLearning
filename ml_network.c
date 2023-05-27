#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrice.h"

typedef struct network_st{
    int  nb_layers;
    int* nb_nodes;     // number of nodes per layer
    float*   biases;   // list of biases between each leayer
    float*** weights;  // list of weight matrices between each layer
} network;


float*** malloc_weights(int nb_layers, int* nb_nodes)
{
    float*** weights = malloc(nb_layers * sizeof(float**));

    for (int layer = 0; layer < nb_layers - 1; layer++)
    {
        int cols  = nb_nodes[layer];
        int lines = nb_nodes[layer + 1];

        weights[layer] = malloc_mat(lines, cols);

        fill_mat(weights[layer], lines, cols);
    }

    return weights;
}


network* malloc_network(int nb_layers, int* nb_nodes)
{
    network *net = malloc(sizeof(network));

    net->nb_layers = nb_layers;
    net->nb_nodes  = malloc((nb_layers) * sizeof(float));
    for (int l = 0; l < nb_layers; l++)
        net->nb_nodes[l] = nb_nodes[l];    

    net->biases  = malloc((nb_layers -1) * sizeof(float));
    fill_vect(net->biases, (nb_layers - 1));

    net->weights = malloc_weights(nb_layers, nb_nodes);


    return net;
}

void free_network(network* net)
{
    for (int layer = 0; layer < net->nb_layers - 1; layer++)
    {
        int lines = net->nb_nodes[layer + 1];
        free_mat(net->weights[layer], lines);
    }
    free(net->weights);
    free(net->nb_nodes);
    free(net->biases);
    free(net);

    return;
}

/* a' = sigma(wa + b) */
float sigmoid(float z){
    printf("Z : %.2f ", z);
    return 1.0 / (1.0 + exp(-z));
}

float* feed_forward(network* net, float* input_vector){
    float* curr_activation = input_vector;
    float* next_activation = NULL;

    int lines = 0, cols = 0;
    for(int layer = 0; layer < net->nb_layers - 1; layer++)
    {
        cols  = net->nb_nodes[layer];
        lines = net->nb_nodes[layer + 1];
        next_activation = multiply_mat_vect(net->weights[layer], 
                                            curr_activation, 
                                            lines,
                                            cols);

        for (int l = 0; l < lines; l++)
            next_activation[l] = sigmoid(next_activation[l] + net->biases[layer]);


        if(layer !=0) free(curr_activation);
        curr_activation = next_activation;

        printf("\nActivation %d : \n", layer);
        print_vect(next_activation, lines);

    }

    return next_activation;
}


void print_network(network* net)
{
    printf("nodes per layer vector : \n");
    for (int layer = 0; layer < net->nb_layers; layer++) {
        printf("| %d ", net->nb_nodes[layer]);
        printf("|\n");
    }
    printf("\n");

    printf("Biases List : \n");
    print_vect(net->biases, net->nb_layers -1);

    printf("Weight Matrices\n");
    for (int layer = 0; layer < net->nb_layers -1; layer++) {

        int cols  = net->nb_nodes[layer];
        int lines = net->nb_nodes[layer + 1];
        print_mat(net->weights[layer], lines, cols);
    }
}
