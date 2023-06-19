
#include <stdint.h>

void print_img(float* img, int label);
uint8_t* parse_labels(const char* labels_path, size_t batch_size, size_t batch_offset);
float** parse_images(const char* images_path, int batch_size, int batch_offset);

