#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <util.h>

void print_matrix(float *matrix, int nrows, int ncols) {
    if (!matrix)
        return;

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            int idx = i * ncols + j;
            printf("%.2f ", matrix[idx]);
        }
        printf("\n");
    }
}

void populate_matrix(float *matrix, int nrows, int ncols) {
    if (!matrix)
        return;

    srand(time(NULL));
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            int idx = i * ncols + j;
            float num = rand() % 100; // Geneate a random 2 digit number
            matrix[idx] = num;
        }
    }
}

bool compare_matrices(float *mat1, float *mat2, int nrows, int ncols) {
    if (!mat1 || !mat2)
        return false;

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            int idx = i * ncols + j;
            if (mat1[idx] != mat2[idx]) {
                return false;
            }
        }
    }
    return true;
}

void populate_standard_matrix(float *matrix) {
    matrix[0] = 1;
    matrix[1] = 1;
    matrix[2] = 1;
    matrix[3] = 2;
    matrix[4] = 2;
    matrix[5] = 2;
    matrix[6] = 3;
    matrix[7] = 3;
    matrix[8] = 3;
}
