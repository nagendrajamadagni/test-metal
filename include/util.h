#ifndef __UTIL__
#define __UTIL__

void print_matrix(float *matrix, int nrows, int ncols);

void populate_matrix(float *matrix, int nrows, int ncols);

bool compare_matrices(float *mat1, float *mat2, int nrows, int ncols);

void populate_standard_matrix(float *matrix);

#endif
