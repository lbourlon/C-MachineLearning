#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* malloc_vect(int lines){
    float* vect = calloc(lines, sizeof(float));
    return vect;
}

float** malloc_mat(int lines, int cols){
  // Allocation de matrice de taille I x J 
  float** mat = malloc(lines * sizeof(float *));
  for(int l = 0; l < lines; ++l) mat[l] = malloc(cols * sizeof(float));
  
  return mat;
}

void free_mat(float** mat, int lines){
    for(int l = 0; l < lines; ++l) free(mat[l]);
    free(mat);
}

void print_vect(float* vect, int lines){
    for (int l = 0; l < lines; ++l) {
        printf("| %.3f ", vect[l]);
        printf("|\n");
    }
    printf("\n");
}

void print_mat(float** mat, int lines, int cols){
    printf("mat[%d][%d] = \n", lines, cols);

    for (int l = 0; l < lines; l++) {
        printf(" |");
        for (int c = 0; c < cols; c++) {
            printf(" %.2f", mat[l][c]);
        }
        printf(" |\n");
    }
    printf("\n");
}

float* multiply_mat_vect(float** mat, float* in_vect, int lines, int cols)
{
    float* out_vect = calloc(lines, sizeof(float));

    for (int l = 0; l < lines; l++) {
        for (int c = 0; c < cols; c++) {
            out_vect[l] += mat[l][c] * in_vect[c];
        }
    }

    return out_vect;
}



/* Fills a Vector with random stuff */
void fill_vect(float* vect, int lines){
    for (int l = 0; l < lines; ++l) {
        vect[l] = drand48();
    }
}

/* Fills a Matrix with random stuff */
void fill_mat(float** mat, int lines, int cols){
    for (int l = 0; l < lines; ++l) {
        for (int c = 0; c < cols; ++c) {
            mat[l][c] = drand48();
        }
    }
}
