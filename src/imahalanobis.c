#include "imahalanobis.h"

void C_imahalanobis(double * x, double * muk, double * wk,
                    double * Qk, double * aki,
                    int * pp, int * pN, int * pdi, double * res)
{
  // Rprintf("Entered imahalanobis\n");
  int p = *pp;
  int N = *pN;
  int di = *pdi;
  int i,j;

  // allocate storage
  double * Qi = (double*)malloc(sizeof(double)*(p*di));
  double * xQi = (double*)malloc(sizeof(double)*(N*di));
  double * proj = (double*)malloc(sizeof(double)*(N*di));
  // Rprintf("Allocated storage\n");

  // Rprintf("x[1] = %f\n", x[0]);
  // X <- x - matrix(muk, N, p, byrow=TRUE)
  for(i = 0; i < N; i++) {
    for(j = 0; j < p; j++) {
      x[RC2IDX(i,j,N)] = x[RC2IDX(i,j,N)] - muk[j];
    }
  }
  // x[i] = x[i] - muk[i % (p+1)];
  // Rprintf("Subtracted the mean %f\n", x[0]);

  // Qi <- wk %*% Qk
  matrix_mult(wk, Qk, p, p, di, Qi);
  // Rprintf("Calculated Qi %f\n", Qi[0]);

  // proj <- (X %*% Qi) %*% aki
  matrix_mult(x, Qi, N, p, di, xQi);
  // Rprintf("Calculated xQi %f\n", xQi[0]);
  matrix_mult(xQi, aki, N, di, di, proj);
  // Rprintf("Calculated proj %f\n", proj[0]);

  // res_old <- rowSums(proj ^ 2)
  for(i = 0; i < N*di; i++) proj[i] = proj[i]*proj[i];
  row_sums(proj, N, di, res);
  // Rprintf("Calculated row sums\n");

  // free allocated memory
  free(Qi);
  free(xQi);
  free(proj);
  // Rprintf("Free memory\n");

}
