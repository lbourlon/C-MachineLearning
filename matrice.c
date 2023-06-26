#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <math.h>

double* malloc_vect(int rows){
    double* vect = calloc(rows, sizeof(double));
    return vect;
}

double** malloc_mat(int rows, int cols){
    double** mat = malloc(rows * sizeof(double *));
    for(int r = 0; r < rows; ++r) mat[r] = malloc(cols * sizeof(double));
    return mat;
}

void free_mat(double** mat, int rows){
    for(int r = 0; r < rows; r++) free(mat[r]);
    free(mat);
}

void print_vect(double* vect, int rows)
{
    for (int r = 0; r < rows; ++r) {
        printf("| %.6f ", vect[r]);
        printf("|\n");
    }
    printf("\n");
}

void print_mat(double** mat, int rows, int cols){
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

void copy_matA_to_matB(int rows, int cols, double matA[rows][cols], double** matB)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            matB[r][c] = matA[r][c]; 
        }
    }
}
void M_times_a_plus_b(double** Mat, double* a, double* b, double* result, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        result[r] = 0;
        for (int c = 0; c < cols; c++) {
            result[r] += Mat[r][c] * a[c]; 
        }

        result[r] = result[r] + b[r];
    }
}

double* multiply_mat_vect(double** mat, double* in_vect, int rows, int cols)
{
    double* out_vect = calloc(rows, sizeof(double));

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out_vect[r] += mat[r][c] * in_vect[c];
        }
    }

    return out_vect;
}

double vect_norm(double* vect, size_t n){
    double norm = 0;
    for (size_t x = 0; x <n ; x++) {
        norm += sqrtf(vect[x] * vect[x]);
    }

    return norm;
}


/* Fills a Vector with random stuff */
void fill_vect(double* vect, int rows){
    for (int r = 0; r < rows; ++r) {
        vect[r] = drand48();
    }
}

/* Fills a Matrix with random stuff */
void fill_mat(double** mat, int rows, int cols){
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat[r][c] = drand48();
        }
    }
}
