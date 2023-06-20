#include <R_ext/Applic.h>
#include <stdio.h>
#include <stdlib.h>
#include <R.h>
#include <math.h>
#include <Rmath.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include "TFunHDDC.h"

#ifndef _IMAHALANOBIS_H_
#define _IMAHALANOBIS_H_

/* C_imahalanobis
 Input:  (double*) x    [N*p] -> Input data matrix
 (double*) muk  [1*p] -> Means of input data
 (double*) wk   [p*p] -> Weights matrix
 (double*) Qk   [p*a] -> Qk matrix
 (double*) aki  [a*a] -> Diagonal matrix of ak values
 (int*)    pp         -> Number of input measurements
 (int*)    pN         -> Number of input rows
 (int*)    pdi        -> Value of d[i] (a in code)
 (double*) res  [N]   -> Pointer to results storage
 Purpose: Calculate mahalanobis distance of each entity in x given wk, Qk, and
  aki. Version made to be optimized and called by R.
 Output: None, put results into res.
 */
void C_imahalanobis(double * x, double * muk, double * wk, double * Qk,
                    double * aki, int * pp, int * pN, int * pdi,
                    double * res);

#endif /* _IMAHALANOBIS_H_ */
