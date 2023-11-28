library(TFunHDDC)

a = fitNOxBenchmark()$fd

W = W <- inprod(a$basis,a$basis)
W[W < 1e-15] <- 0
#Construction of the triangular matrix of Choleski
W_m <- chol(W)
dety=det(W)
Wlist<-list(W=W,
            W_m=W_m,
            dety=dety)
