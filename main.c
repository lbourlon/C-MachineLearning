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
#define LAYERS 3

int main(){
    int nb_nodes[LAYERS] = {INPUTS, 16, OUTPUTS};

    srand48(time(NULL));
    // srand48(0);

    network* net = nw_malloc(LAYERS, nb_nodes);
    
    const int tot_batches = 5900 / 3;
    const int batch_size = 30;

    const int epochs = 7;

    nw_stochastic_gradient_descent(net, TRAIN_IMGS, TRAIN_LBLS, tot_batches, batch_size, epochs);

    nw_evaluate(net, VALIDATE_IMGS, VALIDATE_LBLS, 0);
    nw_evaluate(net, VALIDATE_IMGS, VALIDATE_LBLS, 1);


    nw_free(net);
    return 0;
}

