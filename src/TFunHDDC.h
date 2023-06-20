#ifndef _TFUNHDDC_H_
#define _TFUNHDDC_H_

/* CODE USED FROM TEIGEN */
/*
 * Authors: Andrews, J. Wickins, J. Boers, N. McNicholas, P.
 * Date Taken: 2023-01-01
 * Original Source: teigen (unchanged)
 * Address: https://github.com/cran/teigen
 *
 */

/* convert matrix indexing to vector indexing
 r: row number, c: column number, nr: number of rows in matrix */
#define RC2IDX(r,c,nr) ((r) + ((nr)*(c)))

/* transpose computes the transpose the matrix given a matrix 'mat'
 with 'row' rows and 'col' columns and dimensions removed (i.e., the
 vector form of a matrix) and stores the result in 'rv' */
void transpose(double *mat, int *row, int *col, double *rv);

/* matrix_mult multiplies matrix 'mat1' with 'm' rows by 'n' columns with
 matrix 'mat2' with 'n' rows and 'col' columns. Result is stored in the
 'mult_mat' array. */
void matrix_mult(double *mat1, double *mat2, int m, int n, int col, double *mult_mat);

/* END OF CODE USED FROM TEIGEN */

void row_sums(double * mat, int row, int col, double * sums_mat);

#endif /* _TFUNHDDC_H_ */
