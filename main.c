#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <time.h>
//#include "ingest.h"

#define NB_OUTPUTS 4
#define NB_INPUTS 6
#define NB_LAYERS 7

#define NB_TRAININGS 3

// To Achieve Backpropagation I need :
// [x] Keep track of the "activation" "weighted_activat" given feed-forward
// [ ]Calculate my cost function at the end given an expected output
// [ ]Compute it's error and start backpropagation
//    (easier said than understood)

int main(){
    srand48(time(NULL));

    int nb_nodes[NB_LAYERS] = {NB_INPUTS, 12, 32, 7, 34, 24, NB_OUTPUTS};

    float in_vectors[NB_TRAININGS][NB_INPUTS] ={{0.20, 0.52, 0.72, 0.35, 0.90, 0.37},
                                                {0.42, 0.03, 0.61, 0.48, 0.64, 0.22},
                                                {0.89, 0.15, 0.20, 0.10, 0.45, 0.31}
                                            };

    float expected_out[NB_TRAININGS][NB_OUTPUTS] =   {{0.7, 0.92, 0.45, 0.37},
                                                    {0.2, 0.33, 0.55, 0.48},
                                                    {0.1, 0.89, 0.31, 0.58}};
    
    network* net = malloc_network(NB_LAYERS, nb_nodes);
    print_network(net);


    /* General Idea is to:
     * 1. Feed forward n times, saving each time the last activation
     * 2. For each iteration, track the input vector x, and expected y(x)
     * 3. Calculate cost function on it
     **/


    float cost = 0;
    float tmp = 0;
    for (int nt = 0; nt < NB_TRAININGS; nt++)
    {
        feed_forward(net, in_vectors[nt]);

        for (int i = 0; i < NB_OUTPUTS; i++) {
            tmp = expected_out[nt][0] - net->s_activation[net->nb_layers-1][0];

            cost += tmp * tmp;
        }
    }

    cost /= 2*NB_TRAININGS;
    printf("Cost is %.2f\n", cost);

    free_network(net);

    return 0;
}

