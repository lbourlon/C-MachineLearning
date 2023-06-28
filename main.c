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

#define INPUTS 784
#define OUTPUTS 10
#define LAYERS 4

int main(){
    int nb_nodes[LAYERS] = {INPUTS, 30, 30, OUTPUTS};

    srand48(time(NULL));
    // srand48(0);

    network* net = network_malloc(LAYERS, nb_nodes);
    
    const int tot_images = 59001;
    const int tot_batches = 590;
    const int batch_size = 100;

    const int epochs = 50;

    int offset = 0;

    double** images = parse_images(TRAIN_IMGS, tot_images, offset);
    uint8_t* labels = parse_labels(TRAIN_LBLS, tot_images, offset);
    for (int e = 0; e < epochs; e++) {
        // for (int i = 0; i < 5; i++) {
        //     print_img(images[i], labels[i]);
        // }
        printf("Epoch = [%02d / %d]\n", e, epochs);

        shuffle_imgs_and_lables(labels, images, tot_images);
        // for (int i = 0; i < 5; i++) {
        //     print_img(images[i], labels[i]);
        // }

        for (int s = 0; s < tot_batches; s++) {
            int batch_offset = batch_size * s; 
            nw_mini_batch(net, &images[batch_offset], &labels[batch_offset], batch_size);
        }
    }

    free(labels);
    for(int i = 0; i < batch_size; i++) free(images[i]);
    free(images);
    // print_network(net);

    int nb_img_tests = 2000;
    offset = 0;
    double** test_images = parse_images(VALIDATE_IMGS, nb_img_tests, offset);
    uint8_t* test_labels = parse_labels(VALIDATE_LBLS, nb_img_tests, offset);

    float success_rate = 0.0;
    for (int img = 0; img < nb_img_tests; img++)
    {
        activations* act = activations_malloc(net, test_images[img]);

        network_feed_forward(net, act);

        double max_val = 0.0;
        uint8_t imax = 0;

        for(int o = 0; o < OUTPUTS; o++) {
            if(max_val <= act->nw_output[o]) {
                max_val = act->nw_output[o];
                imax = o;
            }
        }

        printf("nw_output / expected : %d / %d\n", imax, test_labels[img]);

        if (imax == test_labels[img]) {
            success_rate += 1.0;
        } 


        // printf("\nOutput test for img of a %d\n", test_labels[img]);
        // int output_size = net->nodes[net->layers-1];
        // for (int o = 0; o < output_size; o++) {
        //     // printf("\no[%d] -> %.20f", o, act->nw_output[o]);
        //     printf("\n%d -> %.30f", o, act->a[net->layers-1][o]);
        // }
        // printf("\n");


        activations_free(act, net->layers);
    }

    printf("Success : [%0.f / %d]", success_rate, nb_img_tests);
    success_rate *= 100.0 / (float)nb_img_tests;
    printf("| Rate : %f\n", success_rate);

    free(test_labels);
    for(int i = 0; i < nb_img_tests; i++) free(test_images[i]);
    free(test_images);

    network_free(net);
    return 0;
}

