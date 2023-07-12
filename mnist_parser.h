#include <stddef.h>
#include <stdint.h>


typedef struct image_t{
    double* image_content;
    uint8_t image_label;
} image;

void print_img(double* img, int label);

void shuffle_imgs_and_lables(uint8_t* labels, double** images, int size);

void parse_labels_and_images(double*** images, uint8_t** labels, const char* images_path, const char* labels_path, size_t batch_size, size_t batch_offset);
void free_labels_and_images(double** images, uint8_t* labels, int tot_images);
