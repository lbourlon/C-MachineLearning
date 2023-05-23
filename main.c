#include "matrice.h"
#include "ml_network.h"
#include <stdio.h>
//#include "ingest.h"

int main(){
    int lines = 3;
    int cols = 4;
    //float** mat = malloc_mat(lines, cols);

    float v[cols];
    v[0] = 0; v[1] = 1; v[2] = 2; v[3] = 3;


    network net = create_network();

    printf("Network has %d layers and %d things\n", net.sizes, net.num_layers);

    // fill_mat(mat, lines, cols);
    // print_mat(mat, lines, cols);
    //
    // float out_vect[cols];
    //
    // print_vect(v, cols);
    // multiply_mat_vect(mat, v, out_vect, lines, cols);
    // print_vect(out_vect, cols);
    //
    // free_mat(mat, lines);

    return 0;
}
