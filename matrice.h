// Memory Things
float* malloc_vect(int lines);
void free_mat(float** mat, int lines);
float** malloc_mat(int lines, int cols);

// Algebra
void fill_vect(float* vect, int lines);
void fill_mat(float** mat, int lines, int cols);

float* multiply_mat_vect(float** mat, float* in_vect, int lines, int cols);

/* Matrix M (float)
 * vector a (float)
 * vector b (float)
 * vector result = M*a+b
 *  
 */
void M_times_a_plus_b(float** Mat, float* a, float* b, float* result, int lines, int cols);

// Print things
void print_mat(float** matrice, int lines, int cols);
void print_vect(float* vect, int lines);
