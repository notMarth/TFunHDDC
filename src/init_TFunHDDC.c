#include "imahalanobis.h"
#include <R.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

static const R_CMethodDef CEntries[] = {
  {".C_imahalanobis", (DL_FUNC) &C_imahalanobis, 9},
  // {"C_etamax",  (DL_FUNC) &C_etamax,  13},
  {NULL, NULL, 0}
};

void R_init_TFunHDDC(DllInfo * dll)
{
  R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
