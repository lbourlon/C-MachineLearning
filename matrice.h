// Memory Things
float* malloc_vect(int lines);
void free_mat(float** mat, int lines);
float** malloc_mat(int lines, int cols);

// Algebra
void multiply_mat_vect(float** mat, float* in_vect, float* out_vect, int lines, int cols);
void fill_mat(float** mat, int I, int J);

// Print things
void print_mat(float** matrice, int lines, int cols);
void print_vect(float* vect, int lines);
