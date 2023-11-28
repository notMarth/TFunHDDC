library(TFunHDDC)

a = fitNOxBenchmark()$fd
fdobj = list('1'=a, '2'=a)

x = t(fdobj[[1]]$coefs);
for (i in 2:length(fdobj)) x <- cbind(x,t(fdobj[[i]]$coefs))
p <- ncol(x)
for ( i in 1:length(fdobj) ){
  name <- paste('W_var', i, sep = '')
  #Constructing the matrix with inner products of the basis functions
  W_fdobj <- inprod(fdobj[[i]]$basis, fdobj[[i]]$basis)
  assign(name, W_fdobj)
}

#Add 0 ? left and right of  W before constructing the matrix phi
prow <- dim(W_fdobj)[[1]]
pcol <- length(fdobj) * prow
W1 <- cbind( W_fdobj,
             matrix( 0, nrow = prow, ncol = ( pcol - ncol(W_fdobj) ) ) )
W_list <- list()
for ( i in 2:( length(fdobj) ) ){
  W2 <- cbind( matrix( 0, nrow = prow, ncol = (i - 1) * ncol(W_fdobj) ),
               get( paste('W_var', i, sep = '') ),
               matrix( 0, nrow = prow, ncol = ( pcol - i * ncol(W_fdobj) ) ) )
  W_list[[i - 1]] <- W2
}

#Constructing the matrix phi
W_tot <- rbind(W1,W_list[[1]])
if (length(fdobj) > 2){
  for( i in 2:(length(fdobj) - 1) ){
    W_tot <- rbind(W_tot, W_list[[i]])
  }
}
W_tot[W_tot < 1e-15] <- 0
#Construction of the triangular matrix of Choleski
W_m <- chol(W_tot) 
dety<-det(W_tot)
Wlist<-list(W=W_tot,
            W_m=W_m,
            dety=dety
)