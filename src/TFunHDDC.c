#include "TFunHDDC.h"

void matrix_mult(double *mat1, double *mat2, int m, int n, int col, double *mult_mat){
  double total = 0;
  int i, j, k;

  for (k = 0; k < col; k++){
    for (i = 0; i < m; i++){
      for (j = 0; j < n; j++){
        total += mat1[RC2IDX(i, j, m)] * mat2[RC2IDX(j, k, n)];
      }
      mult_mat[RC2IDX(i, k, m)] = total;
      total = 0;
    }
  }
}

void transpose(double *mat, int *row, int *col, double *rv) {
  int i, j;

  for (i = 0; i < *row; i++) {
    for (j = 0; j < *col; j++) {
      rv[j + i*(*col)] = mat[i +j*(*row)];
    }
    /* to help understand rv[j + i*(*col)] = mat[i +j*(*row)], consider:
     row = 2, col = 4:
     j + i*col = {0, 1, 2, 3, 4, 5, 6, 7}
     i + j*row = {0, 2, 4, 6, 1, 3, 5, 7}
     row = 4, col = 2:
     j + i*col = {0, 1, 2, 3, 4, 5, 6, 7}
     i + j*row = {0, 4, 1, 5, 2, 6, 3, 7} */
  }
}

void row_sums(double * mat, int row, int col, double * sums_mat) {
  long double total = 0;
  int i, j;
  for (i = 0; i < row; i++) {
    for (j = 0; j < col; j++) {
      total += mat[RC2IDX(i, j, row)];
    }
    sums_mat[i] = total;
    total = 0;
  }
}
