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


.T_hdclassif_dim_choice <- function(ev, n, method, threshold, graph, noise.ctrl, d_set){
  # Selection of the intrinsic dimension
  # browser()
  N <- sum(n)
  prop <- n/N
  K = ifelse(is.matrix(ev), nrow(ev), 1)
  
  # browser()
  
  if(is.matrix(ev) && K>1){
    p <- ncol(ev)
    if(method=="cattell"){
      dev <- abs(apply(ev, 1, diff))
      max_dev <- apply(dev, 2, max, na.rm=TRUE)
      dev <- dev/rep(max_dev, each=p-1)
      #print(dev)
      print(ev[, -1])
      d <- apply((dev>threshold)*(1:(p-1))*t(ev[, -1]>noise.ctrl), 2, which.max)
      print((dev>threshold)*(1:(p-1))*t(ev[, -1]>noise.ctrl))
      
      if(graph){
        # return user settings to original
        oldpar <- par(no.readonly = TRUE)
        on.exit(par(oldpar))
        op = par(mfrow=c(K*(K<=3)+2*(K==4)+3*(K>4 && K<=9)+4*(K>9), 1+floor(K/4)-1*(K==12)+1*(K==7)))
        for(i in 1:K){
          sub1 <- paste("Class #", i, ",  d", i, "=", d[i], sep="")
          Nmax <- max(which(ev[i, ]>noise.ctrl))-1
          plot(dev[1:(min(d[i]+5, Nmax)), i], type="l", col="blue", main=paste("Cattell's Scree-Test\n", sub1, sep=""), ylab=paste("threshold =", threshold), xlab="Dimension", ylim=c(0, 1.05))
          abline(h=threshold, lty=3)
          points(d[i], dev[d[i], i], col='red')
        }
        par(op)
      }
    } else if(method=="bic"){
      
      d <- rep(0, K)
      if(graph) op = par(mfrow=c(K*(K<=3)+2*(K==4)+3*(K>4 && K<=9)+4*(K>9), 1*(1+floor(K/4)-1*(K==12)+1*(K==7))))
      
      for (i in 1:K) {
        B <- c()
        Nmax <- max(which(ev[i, ]>noise.ctrl))-1
        p2 <- sum(!is.na(ev[i, ]))
        Bmax <- -Inf
        for (kdim in 1:Nmax){
          if ((d[i]!=0 & kdim>d[i]+10)) break
          a <- sum(ev[i, 1:kdim])/kdim
          b <- sum(ev[i, (kdim+1):p2])/(p2-kdim)
          if (b<0 | a<0){
            B[kdim] <- -Inf
          } else {
            L2 <- -1/2*(kdim*log(a)+(p2-kdim)*log(b)-2*log(prop[i])+p2*(1+1/2*log(2*pi))) * n[i]
            B[kdim] <- 2*L2 - (p2+kdim*(p2-(kdim+1)/2)+1) * log(n[i])
          }
          
          if ( B[kdim]>Bmax ){
            Bmax <- B[kdim]
            d[i] <- kdim
          }
        }
        
        if(graph){
          plot(B, type='l', col=4, main=paste("class #", i, ",  d=", d[i], sep=''), ylab='BIC', xlab="Dimension")
          points(d[i], B[d[i]], col=2)
        }
      }
      if(graph) par(op)
    } else if(method=="grid"){
      d <- d_set
    }
  } else{
    ev <- as.vector(ev)
    p <- length(ev)
    
    if(method=="cattell"){
      dvp <- abs(diff(ev))
      Nmax <- max(which(ev>noise.ctrl))-1
      if (p==2) d <- 1
      else d <- max(which(dvp[1:Nmax]>=threshold*max(dvp[1:Nmax])))
      diff_max <- max(dvp[1:Nmax])
      
      if(graph){
        plot(dvp[1:(min(d+5, p-1))]/diff_max, type="l", col="blue", main=paste("Cattell's Scree-Test\nd=", d, sep=''), ylab=paste("threshold =", threshold, sep=' '), xlab='Dimension', ylim=c(0, 1.05))
        abline(h=threshold, lty=3)
        points(d, dvp[d]/diff_max, col='red')
      }
    } else if(method=="bic"){
      d <- 0
      Nmax <- max(which(ev>noise.ctrl))-1
      B <- c()
      Bmax <- -Inf
      for (kdim in 1:Nmax){
        if (d!=0 && kdim>d+10) break
        a <- sum(ev[1:kdim])/kdim
        b <- sum(ev[(kdim+1):p])/(p-kdim)
        if (b<=0 | a<=0) B[kdim] <- -Inf
        else{
          L2 <- -1/2*(kdim*log(a)+(p-kdim)*log(b)+p*(1+1/2*log(2*pi)))*N
          B[kdim] <- 2*L2 - (p+kdim*(p-(kdim+1)/2)+1)*log(N)
        }
        if ( B[kdim]>Bmax ){
          Bmax <- B[kdim]
          d <- kdim
        }
      }
      
      if(graph){
        plot(B, type='l', col=4, main=paste("BIC criterion\nd=", d, sep=''), ylab='BIC', xlab="Dimension")
        points(d, B[d], col=2)
      }
    }
  }
  return(d)
}

noise = 1e-8
graph = F
threshold = 0.1
d_set = c(2,2,2,2)
K = 3

a = fitNOxBenchmark()$fd

n = colSums(t)
p = nrow(a$coefs)
K = dim(t)[2]
ev = matrix(0, K, p)

for(i in 1:K){
  donnees = .T_initmypca.fd1(a, Wlist, t[, i])
  ev[i, ] = donnees$valeurs_propres
  
}

d = .T_hdclassif_dim_choice(ev, n, 'cattell', threshold, graph, noise, d_set)
print(d)