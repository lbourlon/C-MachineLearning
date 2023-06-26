#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <time.h>
#include "matrice.h"
#include "ml_network.h"
#include "mnist_parser.h"

const char* TRAIN_IMGS = "./mnist/train/images.idx3";
const char* TRAIN_LBLS = "./mnist/train/labels.idx1";

const char* VALIDATE_IMGS = "./mnist/validate/images.idx3";
const char* VALIDATE_LBLS = "./mnist/validate/labels.idx1";

// Actual image

#define INPUTS 784
#define OUTPUTS 10
#define LAYERS 3


int main(){
    int nb_nodes[LAYERS] = {INPUTS,  13, OUTPUTS};

    // srand48(time(NULL));
    srand48(0);

    network* net = network_malloc(LAYERS, nb_nodes);

    print_network(net);
    //SGD 
    
    int epochs = 3000;
    int batch_size = 19;
    for (int e = 0; e < epochs; e++) {
        int batch_offset = batch_size * e;

        double** images = parse_images(TRAIN_IMGS, batch_size, batch_offset);
        uint8_t* labels = parse_labels(TRAIN_LBLS, batch_size, batch_offset);

        nw_mini_batch(net, images, labels, batch_size);

        free(labels);
        for(int i = 0; i < batch_size; i++) free(images[i]);
        free(images);
    }
    // print_network(net);

    int nb_img = 10;
    double** images = parse_images(VALIDATE_IMGS, nb_img, 0);
    uint8_t* labels = parse_labels(VALIDATE_LBLS, nb_img, 0);

    // testing output 
    // int img = 0;
    for (int img = 0; img <= 1; img++)
    {
        // print_img(images[0], labels[0]);

        activations* act = activations_malloc(net, images[img]);
        // activations_print(net, act, 0);
        network_feed_forward(net, act);

        // print_img(act->nw_input, labels[img]);

        printf("\nOutput test for img of a %d\n", labels[img]);
        int output_size = net->nodes[net->layers-1];
        for (int o = 0; o < output_size; o++) {
            // printf("\no[%d] -> %.20f", o, act->nw_output[o]);
            printf("\n%d -> %.30f", o, act->a[net->layers-1][o]);
        }
        printf("\n");


        // activations_print(net, act, 0);
        activations_free(act, net->layers);
    }

    free(labels);
    for(int i = 0; i < nb_img; i++) free(images[i]);
    free(images);

    network_free(net);
    return 0;
}

