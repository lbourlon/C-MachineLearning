#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "matrice.h"

typedef struct images_t{
    // maybe do this later
    double* content;
    uint8_t label;
} image;

uint32_t reverse(uint32_t in)
{
    uint32_t out = 0;
    out  = in >> 24;
    out |= (in & 0x00ff0000) >> 8;
    out |= (in & 0x0000ff00) << 8;
    out |= in << 24;

    return out;
}

void shuffle_imgs_and_lables(uint8_t* labels, double** images, int size)
{
    uint8_t temp_lbl;
    double* temp_img;
    for(int i = size - 1; i > 0; i--)
    {
        int j = (rand() % i) + 1;

        temp_lbl  = labels[j];
        labels[j] = labels[i];
        labels[i] = temp_lbl;

        temp_img  = images[j];
        images[j] = images[i];
        images[i] = temp_img;
    }
}


uint8_t* parse_labels(const char* labels_path, size_t batch_size, size_t batch_offset)
{
    uint32_t* header = calloc(16, sizeof(uint32_t));
    size_t header_size = 8;

    int fd = open(labels_path, O_RDONLY);
    if (fd == -1) perror("open");

    int out_r = pread(fd, header, header_size, 0);
    if (out_r == -1) perror("read");

    // uint8_t magic_number     = reverse(header[0]);
    uint32_t nb_labels = reverse(header[1]);

    // printf("Reading [%zu to %zu] out of %d labels\n", batch_offset, batch_offset + batch_size, nb_labels);

    if(batch_size + batch_offset >= nb_labels){
        printf("Trying to read outside of file!\n");
        exit(-1);
    }

    uint8_t* labels = calloc(batch_size, sizeof(uint8_t));

    out_r = pread(fd, labels, batch_size, header_size + batch_offset);
    if (out_r == -1) perror("read");

    free(header);
    close(fd);

    return labels;
}

double** parse_images(const char* images_path, size_t batch_size, size_t batch_offset)
{
    int header_size = 16;
    uint32_t* header = calloc(header_size, sizeof(uint32_t));

    int fd = open(images_path, O_RDONLY);
    if (fd == -1) perror("open");

    int out_r = pread(fd, header, header_size, 0);
    if (out_r == -1) perror("read");

    // uint32_t magic_number = reverse(header[0]);
    uint32_t nb_images = reverse(header[1]);
    uint32_t nb_rows = reverse(header[2]);
    uint32_t nb_cols = reverse(header[3]);

    // printf("Reading [%zu to %zu] out of %d images\n", batch_offset, batch_offset + batch_size, nb_images);

    if(batch_size + batch_offset >= nb_images){
        printf("Trying to read outside of file!\n");
        exit(-1);
    }
    
    uint32_t nb_pixels = nb_rows * nb_cols;
    uint32_t pixels_to_read = nb_pixels * batch_size;

    // printf("magic num : 0x%0.8x\n", magic_number);
    // printf("pixels in file : %d\n", pixels_to_read);
    // printf("pixels per image %d x %d = %d\n", nb_rows, nb_cols, nb_pixels);

    uint8_t* images_concat = malloc(pixels_to_read * sizeof(uint8_t*));

    int out_i = pread(fd, images_concat, pixels_to_read, header_size + batch_offset * nb_pixels);
    if (out_i == -1) perror("read");

    // makes it easier later, don't worry about it
    double** images = malloc(batch_size * sizeof(double*));
    for (size_t i = 0; i < batch_size; i++) {
        images[i] = malloc(nb_pixels * sizeof(double));

        for (size_t p = 0; p < nb_pixels; p++) {
            images[i][p] = (images_concat[nb_pixels * i + p] / 255.0);
        }
    }

    free(images_concat);
    free(header);
    close(fd);

    return images;
}

void parse_labels_and_images(double*** images, uint8_t** labels, const char* images_path, const char* labels_path, size_t batch_size, size_t batch_offset){
    *images = parse_images(images_path, batch_size, batch_offset);
    *labels = parse_labels(labels_path, batch_size, batch_offset);
}

void free_labels_and_images(double** images, uint8_t* labels, int tot_images){
    free(labels);
    for(int i = 0; i < tot_images; i++) free(images[i]);
    free(images);
}

// I assume its of size 28, 28
void print_img(double* img, int label){
    printf("\nHere's your : %d\n", label);
    char ascii_map[] = " .-+=*#@";

    for (int i = 0; i < 784; i++)
    {
        int e = (int)(img[i] * 8.0);
        putchar(ascii_map[e]);

        if(i % 28 == 0)
            printf("\n");
    }
    printf("\n");
}

