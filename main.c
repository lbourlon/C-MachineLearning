#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <time.h>
#include "mnist_parser.h"
//#include "ingest.h"

const char* TRN_IMAGES = "./mnist/train/images.idx3";
const char* TRN_LABELS = "./mnist/train/labels.idx1";

#define OUTPUTS 4
#define INPUTS 6
#define LAYERS 5

#define ITER 3

// To Achieve Backpropagation I need :
// [x] Keep track of the "activation" "weighted_activat" given feed-forward
// [x]Calculate my cost function at the end given an expected output
// [x]Compute it's error and start backpropagation
//    (easier said than understood)

int main(){
    int nb_nodes[LAYERS] = {INPUTS, 12, 4, 7,  OUTPUTS};

    float** expected_out = malloc_mat(ITER, INPUTS);
    float** in_vectors = malloc_mat(ITER, INPUTS);

    srand48(0); //predictable random outputs
    fill_mat(expected_out,ITER, INPUTS);
    fill_mat(in_vectors, ITER, INPUTS);

    print_mat(expected_out, ITER, INPUTS);
    print_mat(in_vectors, ITER, INPUTS);

    // srand48(time(NULL));

    network* net = malloc_network(LAYERS, nb_nodes);
    // activations* act = malloc_activations(LAYERS, in_vectors[0], nb_nodes);
    // feed_forward(net, act);

    // for (int i = 0; i < 15; i++) {
    //     backprop(net, in_vectors, expected_out, ITER);
    // }


    int batch_size = 100;
    int batch_offset = 0;

    uint8_t* labels = parse_labels(TRN_LABELS, batch_size, batch_offset);
    float** images = parse_images(TRN_IMAGES, batch_size, batch_offset);
    
    int target_img = 2;
    print_img(images[target_img], labels[target_img]);


    free(labels);
    free(images);

    // free_activations(act);
    free_network(net);

    free_mat(expected_out, ITER);
    free_mat(in_vectors, ITER);

    return 0;
}

