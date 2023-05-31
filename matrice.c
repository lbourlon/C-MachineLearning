#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* malloc_vect(int rows){
    float* vect = calloc(rows, sizeof(float));
    return vect;
}

float** malloc_mat(int rows, int cols){
  // Allocation de matrice de taille I x J 
  float** mat = malloc(rows * sizeof(float *));
  for(int r = 0; r < rows; ++r) mat[r] = malloc(cols * sizeof(float));
  
  return mat;
}

void free_mat(float** mat, int rows){
    for(int r = 0; r < rows; r++) free(mat[r]);
    free(mat);
}

void print_vect(float* vect, int rows)
{
    for (int r = 0; r < rows; ++r) {
        printf("| %.6f ", vect[r]);
        printf("|\n");
    }
    printf("\n");
}

void print_mat(float** mat, int rows, int cols){
    printf("mat[%d][%d] = \n", rows, cols);

    for (int r = 0; r < rows; r++) {
        printf(" |");
        for (int c = 0; c < cols; c++) {
            printf(" %.2f", mat[r][c]);
        }
        printf(" |\n");
    }
    printf("\n");
}

void M_times_a_plus_b(float** Mat, float* a, float* b, float* result, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        result[r] = 0;
        for (int c = 0; c < cols; c++) {
            result[r] += Mat[r][c] * a[c]; 
        }

        result[r] = result[r] + b[r];
    }
}

float* multiply_mat_vect(float** mat, float* in_vect, int rows, int cols)
{
    float* out_vect = calloc(rows, sizeof(float));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out_vect[r] += mat[r][c] * in_vect[c];
        }
    }

    return out_vect;
}



/* Fills a Vector with random stuff */
void fill_vect(float* vect, int rows){
    for (int r = 0; r < rows; ++r) {
        vect[r] = drand48();
    }
}

/* Fills a Matrix with random stuff */
void fill_mat(float** mat, int rows, int cols){
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat[r][c] = drand48();
        }
    }
}
