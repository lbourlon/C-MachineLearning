// Memory Things
float* malloc_vect(int lines);
void free_mat(float** mat, int lines);
float** malloc_mat(int lines, int cols);

// Algebra
float* multiply_mat_vect(float** mat, float* in_vect, int lines, int cols);
void fill_vect(float* vect, int lines);
void fill_mat(float** mat, int lines, int cols);

// Print things
void print_mat(float** matrice, int lines, int cols);
void print_vect(float* vect, int lines);
