.T_initmypca.fd1 <- function(fdobj,Wlist, Ti){
  # 
  if (inherits(fdobj, 'list')){ #the multivariate case
    #save the mean before centering
    mean_fd <- list()
    for ( i in 1:length(fdobj) ){
      mean_fd[[i]] <- fdobj[[i]]
    }
    
    #Making the matrix of coeffs
    coef <- t(fdobj[[1]]$coefs)
    for ( i in 2:length(fdobj) ){
      coef <- cbind( coef, t(fdobj[[i]]$coefs) )
    }
    
    wtcov <- cov.wt(coef, wt = Ti, method = "ML")
    mat_cov <- wtcov$cov
    coefmean <- wtcov$center
    
    n_lead <- 0
    n_var <- dim(fdobj[[1]]$coefs)[1]
    fdobj[[1]]$coefs <- sweep(fdobj[[1]]$coefs, 1,
                              coefmean[(n_lead + 1):(n_var + n_lead)])
    mean_fd[[1]]$coefs <- as.matrix(data.frame(mean = coefmean[(n_lead + 1):(n_var + n_lead)]))
    for ( i in 2:length(fdobj) ){
      n_lead <- n_lead + n_var
      n_var <- dim(fdobj[[i]]$coefs)[1]
      fdobj[[i]]$coefs <- sweep(fdobj[[i]]$coefs, 1, coefmean[(n_lead+1):(n_var+n_lead)])
      mean_fd[[i]]$coefs <- as.matrix(data.frame(mean = coefmean[(n_lead + 1):(n_var + n_lead)]))
    }
    #covariance matrix
    cov = Wlist$W_m %*% mat_cov %*% t(Wlist$W_m)
    #eigenvalues and eigenfunctions
    valeurs <- Eigen(cov)
    valeurs_propres <- valeurs$values
    vecteurs_propres <- valeurs$vectors
    bj <- solve(Wlist$W_m) %*% vecteurs_propres
    fonctionspropres <- fdobj[[1]]
    fonctionspropres$coefs <- bj
    scores <- coef %*% Wlist$W %*% bj
    
    varprop <- valeurs_propres / sum(valeurs_propres)
    
    ipcafd <- list(valeurs_propres = valeurs_propres,
                   harmonic = fonctionspropres,
                   scores = scores,
                   covariance = cov,
                   U = bj,
                   varprop = varprop,
                   meanfd = mean_fd,
                   mux = coefmean)
    
    
  } else if (!inherits(fdobj, 'list')) {
    #calculation of the mean for each group
    mean_fd <- fdobj
    coef <- t(fdobj$coefs)
    wtcov <- cov.wt(coef, wt = Ti, method = "ML")
    coefmean <- wtcov$center
    #covariance matrix
    mat_cov <- wtcov$cov
    #Centering of the functional objects by group
    fdobj$coefs <- sweep(fdobj$coefs, 1, coefmean) # sweep subtracts coefmean from fdobj$coefs, i.e fdobj$coefs_i-coefmean_i
    mean_fd$coefs <- as.matrix( data.frame(mean = coefmean) )
    cov <- Wlist$W_m %*% mat_cov %*% t(Wlist$W_m) 
    #eigenvalues and eigenfunctions
    #--------------------------------------------
    valeurs <- Eigen(cov)
    valeurs_propres <- valeurs$values
    vecteurs_propres <- valeurs$vectors
    fonctionspropres <- fdobj
    bj <- solve(Wlist$W_m) %*% vecteurs_propres
    fonctionspropres$coefs <- bj
    #calculations of scores from the formula for pca.fd
    scores <- inprod(fdobj, fonctionspropres)
    
    varprop <- valeurs_propres / sum(valeurs_propres)
    
    ipcafd <- list(valeurs_propres = valeurs_propres,
                   harmonic = fonctionspropres,
                   scores = scores,
                   covariance = cov,
                   U = bj,
                   meanfd = mean_fd,
                   mux = coefmean)
    #-------------------------------------------
  }
  class(ipcafd) <- "ipca.fd"
  return(ipcafd)
} # end of .T_initmypca.fd1

a = fitNOxBenchmark()$fd
a <- list('1' = a, '2' = a)

for(i in 1:ncol(t)){
  print(i)
  print(.T_initmypca.fd1(a, Wlist, t[,i])$mux)
}