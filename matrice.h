// Memory Things
float* malloc_vect(int rows);
void free_mat(float** mat, int rows);
float** malloc_mat(int rows, int cols);

// Algebra
void fill_vect(float* vect, int rows);
void fill_mat(float** mat, int rows, int cols);

float* multiply_mat_vect(float** mat, float* in_vect, int rows, int cols);

/* Matrix M (float)
 * vector a (float)
 * vector b (float)
 * vector result = M*a+b
 *  
 */
void M_times_a_plus_b(float** Mat, float* a, float* b, float* result, int rows, int cols);

// Print things
void print_mat(float** matrice, int rows, int cols);
void print_vect(float* vect, int rows);
