// Memory Things
#include <stddef.h>
double* malloc_vect(int rows);
void free_mat(double** mat, int rows);
double** malloc_mat(int rows, int cols);

// Algebra
void fill_vect(double* vect, int rows);
void fill_mat(double** mat, int rows, int cols);

double vect_norm(double* vect, size_t n);
double* multiply_mat_vect(double** mat, double* in_vect, int rows, int cols);

/* Matrix M (double)
 * vector a (double)
 * vector b (double)
 * vector result = M*a+b
 *  
 */
void M_times_a_plus_b(double** Mat, double* a, double* b, double* result, int rows, int cols);

void copy_matA_to_matB(int rows, int cols, double matA[rows][cols], double** matB);

// Print things
void print_mat(double** matrice, int rows, int cols);
void print_vect(double* vect, int rows);
