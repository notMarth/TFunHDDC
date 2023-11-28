library(TFunHDDC)
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
      #print(ev[, -1])
      d <- apply((dev>threshold)*(1:(p-1))*t(ev[, -1]>noise.ctrl), 2, which.max)
      #print((dev>threshold)*(1:(p-1))*t(ev[, -1]>noise.ctrl))
      
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

.T_funhddt_init  <- function(fdobj,Wlist, K, t, nux, model, threshold, method,
                             noise.ctrl, com_dim, d_max, d_set){ # this is the init step
  if (!inherits(fdobj, 'list')) { x <- t(fdobj$coefs) #THIS IS the univariate CASE
  } else {x = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) x <- cbind(x,t(fdobj[[i]]$coefs))}
  # x is the coefficient in the fdobject
  
  N <- nrow(x)
  p <- ncol(x)
  prop <- c()
  n <- colSums(t)
  prop <- n / N  # prop is pi as a vector needed for ai
  
  mu <- matrix(NA, K, p)
  
  
  ind <- apply(t > 0, 2, which)
  n_bis <- c()
  for(i in 1:K) n_bis[i] <- length(ind[[i]])
  
  #
  #Calculation on Var/Covar matrices and mean mu vector
  #
  
  # we keep track of the trace (== sum of eigenvalues) to compute the b
  traceVect <- c()
  
  
  ev <- matrix(0, K, p)
  Q <- vector(mode='list', length=K) #A@5 this is matrix q_k
  fpcaobj <- list()
  for (i in 1:K){
    donnees <- .T_initmypca.fd1(fdobj,Wlist, t[, i])
    
    
    mu[i, ] <- donnees$mux
    traceVect[i] <- sum( diag(donnees$valeurs_propres) )
    ev[i, ] <- donnees$valeurs_propres
    Q[[i]] <- donnees$U
    fpcaobj[[i]] <- donnees
  }
  
  
  #Intrinsic dimensions selection
  
  if (model%in%c("AJBQD", "ABQD")){
    d <- rep(com_dim, length=K)
  } else if ( model%in%c("AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD") ){
    dmax <- min(apply((ev>noise.ctrl)*rep(1:ncol(ev), each=K), 1, which.max))-1
    if(com_dim>dmax) com_dim <- max(dmax, 1)
    d <- rep(com_dim, length=K)
  } else {
    d <- .T_hdclassif_dim_choice(ev, n, method, threshold, FALSE, noise.ctrl, d_set)
  }
  
  #Setup of the Qi matrices
  # 
  Q1 <- Q
  for(i in 1:K) Q[[i]] <- matrix(Q[[i]][, 1:d[i]], p, d[i])
  
  
  #Calculation of the remaining parameters of the selected model
  
  # PARAMETER a
  # 
  ai <- matrix(NA, K, max(d))
  if ( model %in% c('AKJBKQKDK', 'AKJBQKDK', 'AKJBKQKD', 'AKJBQKD') ){
    for (i in 1:K) ai[i, 1:d[i]] <- ev[i, 1:d[i]]
  } else if ( model%in%c('AKBKQKDK', 'AKBQKDK' , 'AKBKQKD', 'AKBQKD') ){
    for (i in 1:K) ai[i, ] <- rep(sum(ev[i, 1:d[i]])/d[i], length=max(d))
  } else if(model=="AJBQD"){
    for (i in 1:K) ai[i, ] <- ev[1:d[1]]
  } else if(model=="ABQD") {
    ai[] <- sum(ev[1:d[1]])/d[1]
  } else {
    a <- 0
    eps <- sum(prop * d)
    for (i in 1:K) a <- a + sum(ev[i, 1:d[i]]) * prop[i]
    ai <- matrix(a / eps, K, max(d))
  }
  
  # PARAMETER b
  # 
  bi <- c()
  denom <- min(N, p)
  if ( model %in% c('AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK', 'AKJBKQKD', 'AKBKQKD', 'ABKQKD') ){
    for(i in 1:K){
      remainEV = traceVect[i] - sum(ev[ i, 1:d[i] ])
      # bi[i] <- sum(ev[i, (d[i]+1):min(N, p)])/(p-d[i])
      bi[i] <- remainEV / (p - d[i]) #pour moi c'est p au lieu de denom
    }
  } else {
    b <- 0
    eps <- sum(prop * d)
    for(i in 1:K){
      remainEV = traceVect[i] - sum(ev[ i, 1:d[i] ])
      # b <- b + sum(ev[i, (d[i]+1):min(N, p)])*prop[i]
      b <- b + remainEV * prop[i]
    }
    bi[1:K] <- b / (min(N, p) - eps)
  }
  
  
  
  list(model = model,
       K = K,
       d = d,
       a = ai,
       b = bi,
       mu = mu,
       prop = prop,
       ev = ev,
       Q = Q,
       fpcaobj = fpcaobj,
       Q1 = Q1
  )
  
  
} # end of .T_funhddt_init

.T_repmat <- function(v,n,p){ #A@5 WHAT IS THIS???????????????????????
  if (p==1){M = cbind(rep(1,n)) %*% v} #A@5 a matrix of column of v
  else { M = matrix(rep(v,n),n,(length(v)*p),byrow=T)} # removed cat("!");
  M
}

.T_mypcat.fd1 <- function(fdobj,Wlist, Ti, CorI){
  if (inherits(fdobj, 'list')){ # multivariate CASE
    #saving the mean before contering
    mean_fd <- list()
    for (i in 1:length(fdobj)){
      mean_fd[[i]] <- fdobj[[i]]
    }
    
    #centering the functional objects
    for ( i in 1:length(fdobj) ){
      coefmean <- apply(t( as.matrix(CorI) %*%
                             matrix(1, 1, nrow(fdobj[[i]]$coefs) ) ) *
                          fdobj[[i]]$coefs, 1, sum) / sum(CorI)
      fdobj[[i]]$coefs <- sweep(fdobj[[i]]$coefs, 1, coefmean)
      mean_fd[[i]]$coefs <- as.matrix( data.frame(mean = coefmean) )
    }
    #Constructing the matrix of coefficents
    coef <- t(fdobj[[1]]$coefs)
    for ( i in 2:length(fdobj) ){
      coef <- cbind( coef, t(fdobj[[i]]$coefs) )
    }
    #covariance matrix
    mat_cov <- crossprod( t( .T_repmat(sqrt(CorI), n = dim(t(coef))[[1]],p=1) *
                               t(coef) ) ) / sum(Ti)
    cov <- Wlist$W_m %*% mat_cov %*% t(Wlist$W_m)
    #eigenvalues and eigenfunctions
    valeurs <- Eigen(cov)
    valeurs_propres <- valeurs$values
    vecteurs_propres <- valeurs$vectors
    
    bj <- solve(Wlist$W_m) %*% vecteurs_propres
    fonctionspropres <- fdobj[[1]]
    fonctionspropres$coefs <- bj
    scores <- coef %*% Wlist$W %*% bj
    
    varprop <- valeurs_propres / sum(valeurs_propres)
    
    pcafd <- list(valeurs_propres = valeurs_propres,
                  harmonic = fonctionspropres,
                  scores = scores,
                  covariance = cov,
                  U = bj,
                  varprop = varprop,
                  meanfd = mean_fd
    )
    
  }else if (!inherits(fdobj, 'list')) { #univariate CASE
    #Calculation of means by group
    mean_fd <- fdobj
    #Centering the functional objects by group
    coefmean <- apply(t( as.matrix(CorI) %*% matrix( 1, 1, nrow(fdobj$coefs) ) )
                      * fdobj$coefs, 1, sum) / sum(CorI) 
    fdobj$coefs <- sweep(fdobj$coefs, 1, coefmean) 
    mean_fd$coefs <- as.matrix( data.frame(mean = coefmean) )
    coef <- t(fdobj$coefs)
    #print(coef[[1]])
    #covariance matrix 
    mat_cov <- crossprod( t( .T_repmat(sqrt(CorI),
                                       n = dim( t(coef) )[[1]],p=1) * t(coef) ) ) / sum(Ti)
    rep = t( .T_repmat(sqrt(CorI), n = dim( t(coef) )[[1]],p=1) * t(coef) )
    #print(rep)
    cov <- Wlist$W_m %*% mat_cov %*% t(Wlist$W_m) #A@5 this is the last formula on page 7
    #eigenvalues and eigenfunctions
    #--------------------------------------------
    valeurs <- Eigen(cov)
    valeurs_propres <- valeurs$values
    vecteurs_propres <- valeurs$vectors
    fonctionspropres <- fdobj
    bj <- solve(Wlist$W_m) %*% vecteurs_propres
    fonctionspropres$coefs <- bj
    #calculations of scores by the formula for pca.fd
    scores <- inprod(fdobj, fonctionspropres)
    
    varprop <- valeurs_propres / sum(valeurs_propres)
    
    pcafd <- list(valeurs_propres = valeurs_propres,
                  harmonic = fonctionspropres,
                  scores = scores,
                  covariance = cov,
                  U = bj,
                  meanfd = mean_fd
    )
    #-------------------------------------------
  }
  class(pcafd) <- "pca.fd"
  return(pcafd)
} # end of .T_mypcat.fd1

.T_tyxf7 <- function(dfconstr, nux, n, t, tw, K, p, N){
  if(dfconstr=="no"){
    dfoldg <- nux
    for(i in 1:K){
      constn <- 1 + (1/n[i]) * sum(t[, i] * ( log(tw[, i]) - tw[, i]) ) +
        digamma( (dfoldg[i] + p) / 2 ) - log( (dfoldg[i] + p) / 2 )
      
      nux[i] <- uniroot( function(v) log(v / 2) - digamma(v / 2) + constn,
                         lower = 0.0001, upper = 1000, tol = 0.00001)$root
      if(nux[i] > 200){
        nux[i] <- 200
      }
      if(nux[i] < 2){
        nux[i] <- 2
      }
    }
  }
  else
  {
    dfoldg <- nux[1]
    constn <- 1 + (1/N) * sum( t * (log(tw) - tw) ) +
      digamma( (dfoldg + p) / 2 ) - log( (dfoldg + p) / 2 )
    
    
    dfsamenewg <- uniroot( function(v) log(v / 2) - digamma(v / 2) + constn,
                           lower = 0.0001, upper = 1000, tol = 0.01)$root
    if(dfsamenewg > 200){
      dfsamenewg <- 200
    }
    if(dfsamenewg < 2){
      dfsamenewg <- 2
    }
    nux <- c( rep(dfsamenewg, K) )
  }
  return(nux)
} # end of .T_tyxf7

.T_tyxf8 <- function(dfconstr, nux, n, t, tw, K, p, N){
  if(dfconstr == "no"){
    dfoldg <- nux
    for(i in 1:K){
      constn <- 1 + (1 / n[i]) * sum( t[, i] * (log(tw[, i]) - tw[,i]) ) +
        digamma( (dfoldg[i] + p) / 2 ) -
        log( (dfoldg[i]+p) / 2 )
      
      constn <- -constn
      nux[i] <- (-exp(constn) + 2 * ( exp(constn)) *
                   ( exp( digamma(dfoldg[i] / 2)) -
                       ( (dfoldg[i]/2) - (1/2) ) ) ) / ( 1 - exp(constn) )
      if(nux[i] > 200){
        nux[i] <- 200
      }
      if(nux[i] < 2){
        nux[i] <- 2
      }
    }
  } else {
    dfoldg <- nux[1]
    constn <- 1 + (1 / N) * sum( t * (log(tw) - tw) ) +
      digamma( (dfoldg + p) / 2 ) - log( (dfoldg + p) / 2 )
    constn <- -constn
    dfsamenewg <- (-exp(constn) + 2 * ( exp(constn)) *
                     (exp( digamma(dfoldg / 2)) -
                        ( (dfoldg /2) - (1 / 2) ) ) ) / ( 1 - exp(constn) )
    if(dfsamenewg > 200){
      dfsamenewg <- 200
    }
    if(dfsamenewg < 2){
      dfsamenewg <- 2
    }
    nux <- c( rep(dfsamenewg, K) )
  }
  return(nux)
} # end of .T_tyxf8


.T_funhddt_m_step1  <- function(fdobj,Wlist, K, t,tw, nux, dfupdate, dfconstr, model, threshold, method, noise.ctrl, com_dim, d_max, d_set){ #A@5 this is the M step
  if (!inherits(fdobj, 'list')) { x <- t(fdobj$coefs) # THIS IS the univariate CASE
  } else {x = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) x = cbind(x,t(fdobj[[i]]$coefs))}
  # x is the coefficient in the fdobject
  
  N <- nrow(x)
  p <- ncol(x)
  prop <- c()
  n <- colSums(t)
  prop <- n / N  # props is pi as a vector
  mu <- matrix(NA, K, p)
  mu1 <- matrix(NA, K, p)
  corX <- matrix(0, N, K)
  corX <- t * tw
  for (i in 1:K) {
    
    mu[i, ] <- apply(t( as.matrix(corX[, i]) %*% matrix(1, 1, p) ) * t(x),
                     1, sum) / sum(corX[,i])
    mu1[i, ] <- colSums(corX[, i] * x) / sum(corX[, i])
  }
  
  ind <- apply(t>0, 2, which)
  n_bis <- c()
  for(i in 1:K) n_bis[i] <- length(ind[[i]])
  #calculation of the degrees of freedom
  if(dfupdate=="approx"){
    testing <- try(jk861 <- .T_tyxf8(dfconstr, nux, n, t, tw, K, p, N),
                   silent = TRUE)
    if( all( is.finite(testing) ) ){
      nux <- jk861
    }
    # else{break}
  }
  else{
    if(dfupdate=="numeric"){
      testing <- try(jk861 <- .T_tyxf7(dfconstr, nux, n, t, tw, K, p, N),
                     silent = TRUE)
      if( all( is.finite(testing) ) ){
        nux <- jk861
      }
      # else{break}
    }
  }
  
  
  #
  #Calculation on Var/Covar matrices
  #
  
  # we keep track of the trace (== sum of eigenvalues) to compute the b
  traceVect <- c()
  
  
  ev <- matrix(0, K, p)
  Q <- vector(mode='list', length=K) #A@5 this is matrix q_k
  fpcaobj = list()
  for (i in 1:K){
    donnees <- .T_mypcat.fd1(fdobj,Wlist, t[, i], corX[, i])
    #
    #-----------------------------------
    traceVect[i] <- sum( diag(donnees$valeurs_propres) )
    ev[i, ] <- donnees$valeurs_propres
    Q[[i]] <- donnees$U
    fpcaobj[[i]] <- donnees
  }
  
  #Intrinsic dimensions selection
  
  if ( model %in% c("AJBQD", "ABQD") ){
    d <- rep(com_dim, length=K)
  } else if ( model%in%c("AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD") ){
    dmax <- min(apply((ev>noise.ctrl)*rep(1:ncol(ev), each=K), 1, which.max))-1
    if(com_dim>dmax) com_dim <- max(dmax, 1)
    d <- rep(com_dim, length=K)
  } else {
    d <- .T_hdclassif_dim_choice(ev, n, method, threshold, FALSE, noise.ctrl, d_set)
  }
  
  #Setup of the Qi matrices
  Q1 <- Q
  for(i in 1:K) Q[[i]] <- matrix(Q[[i]][, 1:d[i]], p, d[i])
  
  #Calculation of the remaining parameters of the selected model
  
  # PARAMETER a
  
  ai <- matrix(NA, K, max(d))
  if ( model%in%c('AKJBKQKDK', 'AKJBQKDK', 'AKJBKQKD', 'AKJBQKD') ){
    for (i in 1:K) ai[i, 1:d[i]] <- ev[i, 1:d[i]]
  } else if ( model%in%c('AKBKQKDK', 'AKBQKDK' , 'AKBKQKD', 'AKBQKD') ){
    for (i in 1:K) ai[i, ] <- rep(sum(ev[i, 1:d[i]])/d[i], length=max(d))
  } else if(model=="AJBQD"){
    for (i in 1:K) ai[i, ] <- ev[1:d[1]]
  } else if(model=="ABQD") {
    ai[] <- sum(ev[1:d[1]])/d[1]
  } else {
    a <- 0
    eps <- sum(prop*d)
    for (i in 1:K) a <- a + sum(ev[i, 1:d[i]])*prop[i]
    ai <- matrix(a/eps, K, max(d))
  }
  
  # PARAMETER b
  bi <- c()
  denom <- min(N, p)
  if ( model%in%c('AKJBKQKDK', 'AKBKQKDK', 'ABKQKDK', 'AKJBKQKD', 'AKBKQKD', 'ABKQKD') ){
    for(i in 1:K){
      remainEV <- traceVect[i] - sum(ev[i, 1:d[i]])
      # bi[i] <- sum(ev[i, (d[i]+1):min(N, p)])/(p-d[i])
      bi[i] <- remainEV/(p-d[i]) #pour moi c'est p au lieu de denom
    }
  } else  {
    b <- 0
    eps <- sum(prop*d)
    for(i in 1:K){
      remainEV = traceVect[i] - sum(ev[i, 1:d[i]])
      # b <- b + sum(ev[i, (d[i]+1):min(N, p)])*prop[i]
      b <- b + remainEV*prop[i]
    }
    bi[1:K] <- b / (min(N, p) - eps)
  }
  
  
  #---------------------------------------
  
  list(model = model,
       K = K,
       d = d,
       a = ai,
       b = bi,
       mu = mu,
       prop = prop,
       nux = nux,
       ev = ev,
       Q = Q,
       fpcaobj = fpcaobj,
       Q1 = Q1)
  
  
} # end of .T_funhddt_m_step1


noise = 1e-8
graph = F
threshold = 0.1
d_set = c(2,2,2,2)
K = 3
d_max = 100
df_start = 50

a = fitNOxBenchmark()$fd

n = colSums(t)
p = nrow(a$coefs)
K = dim(t)[2]
ev = matrix(0, K, p)
nux = rep(df_start, K)

models = c("AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "ABQKDK")
dfupdate = c("approx", "numeric")
dfconstr = c("yes", "no")
method = c("cattell", "bic")

for(i in models){
  
  for(j in method){
    initx = .T_funhddt_init(a, Wlist, K, t, nux, i, threshold, j, noise, NULL, d_max, d_set)
    tw = TFunHDDC:::.T_funhddt_twinit(a, Wlist, initx, nux)
    
    for(k in dfupdate){
      for(q in dfconstr){
        m = .T_funhddt_m_step1(a, Wlist, K, t, tw, nux, k, q, i, threshold, j, noise, NULL, d_max, d_set)
        print(i)
        print(m$nux)
        temp = list(a=m$a, b=m$b, d=m$d, nux=m$nux, prop=m$prop, mu=m$mu, Q=m$Q, Q1=m$Q1, model=i, method=j, dfupdate=k, dfconstr=q)
        
        filename = paste0('R',i, j, k, q, ".csv")
        write.csv(temp, filename)
      }
    }
    
  }
  
}


