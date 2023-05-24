#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* malloc_vect(int lines){
    float* vect = calloc(lines, sizeof(float));
    return vect;
}

float** malloc_mat(int lines, int cols){
  // Allocation de matrice de taille I x J 
  float** mat = calloc(lines, sizeof(float*));
  for(int l = 0; l < lines; ++l) mat[l] = calloc(cols, sizeof(float));
  
  return mat;
}

void multiply_mat_vect(float** mat, float* in_vect, float* out_vect, int lines, int cols){
    for (int l = 0; l < lines; ++l) {
        for (int c = 0; c < cols; ++c) {
            out_vect[c] = mat[l][c] * in_vect[c];
        }
    }
}

void free_vect(float* vect){
    free(vect);
}

void free_mat(float** mat, int lines){
    for(int l = 0; l < lines; ++l) free(mat[l]);
    free(mat);
}

void print_mat(float** matrice, int lines, int cols){
    for (int i = 0; i <= lines; ++i) {printf("----");} printf("\n");

    for (int i = 0; i < lines; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("| %.0f ", matrice[i][j]);
        }
        printf("|\n");
    }
    for (int i = 0; i <= lines; ++i) {printf("----");} printf("\n");
}

void print_vect(float* vect, int lines){
    for (int l = 0; l < lines; ++l) {
        printf("| %.0f ", vect[l]);
        printf("|\n");
    }
    printf("\n");
}


/* Fills a Vector with random stuff */
void fill_vect(float* vect, int lines){
    srand48(time(NULL));
    for (int l = 0; l < lines; ++l) {
        vect[l] = (float)drand48();
    }
}

/* Fills a Matrix with random stuff */
void fill_mat(float** mat, int lines, int cols){
    srand48(time(NULL));

    for (int l = 0; l < lines; ++l) {
        for (int c = 0; c < cols; ++c) {
            mat[l][c] = (float)drand48();
        }
    }
}
