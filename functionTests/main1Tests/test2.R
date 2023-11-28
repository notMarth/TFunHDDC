library(TFunHDDC)

.T_funhddc_main1 <- function(fdobj,Wlist, K, dfstart=dfstart,dfupdate=dfupdate,
                             dfconstr=dfconstr,model, itermax=200,threshold,
                             method, eps=1e-6, init, init.vector, mini.nb,
                             min.individuals, noise.ctrl, com_dim=NULL,
                             kmeans.control, d_max, d_set=c(2,2,2,2),known=NULL, ...){
  
  ModelNames <- c("AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", "ABQKDK", "AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD")
  if (!inherits(fdobj, 'list')) {DATA <- t(fdobj$coefs)}
  else {DATA = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) DATA <- cbind(DATA,t(fdobj[[i]]$coefs))}
  p <- ncol(DATA)
  N <- nrow(DATA)
  com_ev <- NULL
  
  # We set d_max to a proper value
  d_max <- min(N, p, d_max)
  #################################clasification######################
  ########################################################
  # /*
  # * Authors: Andrews, J. Wickins, J. Boers, N. McNicholas, P.
  # * Date Taken: 2023-01-01
  # * Original Source: teigen (modified)
  # * Comment: Known and training are treated as in teigen, matchtab and matchit
  # *          are also treated like in teigen
  # * Address: https://github.com/cran/teigen
  # *
  # */
  if(is.null(known)){
    #this is the  clustering case
    clas <- 0
    kno <- NULL
    testindex <- NULL
  }
  else
    #various sorts of classification
  {
    if(length(known)!=N){
      return("Known classifications vector not given, or not the same length as the number of samples (see help file)")
    }else
    {
      
      if(!anyNA(known)){
        #Gs <- length(unique(known))
        warning("No NAs in 'known' vector supplied, all values have known classification (parameter estimation only)")
        testindex <- 1:N
        kno <- rep(1, N)
        unkno <- (kno-1)*(-1)
        K <- length(unique(known))
        init.vector <- as.numeric(known)
        init <-"vector"
      }
      else{
        training <- which(!is.na(known))
        testindex <- training
        kno <- vector(mode="numeric", length=N)
        kno[testindex] <- 1
        unkno <- (kno-1)*(-1)##it is 0 for training and 1 for the rest
      }
    }
    
    clas <- 1
    
  }
  ############################################
  #################################################
  
  if (K > 1){
    t <- matrix(0, N, K)
    tw <- matrix(0, N, K)
    ##############################################
    #initialization for t
    if(init == "vector"){
      if(clas>0){
        cn1=length(unique(known[testindex]))
        matchtab=matrix(-1,K,K)
        matchtab[1:cn1,1:K] <- table(known,init.vector)
        rownames(matchtab)<-c(rownames(table(known,init.vector)),which(!(1:K %in% unique( known[testindex]))))
        matchit<-rep(0,1,K)
        while(max(matchtab)>0){
          
          ij<-as.integer(which(matchtab==max(matchtab), arr.ind=T)[1,2])
          ik<-which.max(matchtab[,ij])
          matchit[ij]<-as.integer(rownames(as.matrix(ik)))
          matchtab[,ij]<-rep(-1,1,K)
          matchtab[as.integer(ik),]<-rep(-1,1,K)
          
        }
        matchit[which(matchit==0)]=which(!(1:K %in% unique(matchit)))
        initnew <- init.vector
        for(i in 1:K){
          initnew[init.vector==i] <- matchit[i]
        }
        init.vector <- initnew
      }
      
      
      for (i in 1:K) t[which(init.vector == i), i] <- 1
      print(t)
    }
    else {if (init == "kmeans") {
      kmc <- kmeans.control
      cluster <- kmeans(DATA, K, iter.max = kmc$iter.max, nstart = kmc$nstart,
                        algorithm = kmc$algorithm, trace = kmc$trace)$cluster
      if(clas>0)
      {
        cn1=length(unique(known[testindex]))
        matchtab=matrix(-1,K,K)
        matchtab[1:cn1,1:K] <- table(known,cluster)
        rownames(matchtab)<-c(rownames(table(known,cluster)),which(!(1:K %in% unique( known[testindex]))))
        matchit<-rep(0,1,K)
        while(max(matchtab)>0){
          
          ij<-as.integer(which(matchtab==max(matchtab), arr.ind=T)[1,2])
          ik<-which.max(matchtab[,ij])
          matchit[ij]<-as.integer(rownames(as.matrix(ik)))
          matchtab[,ij]<-rep(-1,1,K)
          matchtab[as.integer(ik),]<-rep(-1,1,K)
          
        }
        matchit[which(matchit==0)]=which(!(1:K %in% unique(matchit)))
        knew <- cluster
        for(i in 1:K){
          knew[cluster==i] <- matchit[i]
        }
        cluster <- knew
        
      }
      
      
      for (i in 1:K)
        t[which(cluster == i), i] <- 1
      print(t)
      #A@5 initialize t with the result of k-means
    } else
    {if (init=="tkmeans"){ # TRIMMED COMMENT
      kmc <- kmeans.control
      
      # added default alpha
      cluster <- tkmeans(DATA, k = K, iter.max = kmc$iter.max,
                         nstart = kmc$nstart, trace = kmc$trace,
                         alpha = kmc$alpha)$cluster
      for (i in 1:K)
        t[which(cluster == i), i] <- 1
      # random assignment for trimmed values
      rand_assign <- sample(1:K, sum(cluster == 0), replace = T)
      ind <- which(cluster == 0)
      for ( j in 1:length(rand_assign) )
        t[ ind[j], rand_assign[j] ] <- 1
      if(clas>0)
      {
        cluster <- max.col(t)
        cn1=length(unique(known[testindex]))
        matchtab=matrix(-1,K,K)
        matchtab[1:cn1,1:K] <- table(known,cluster)
        rownames(matchtab)<-c(rownames(table(known,cluster)),which(!(1:K %in% unique( known[testindex]))))
        matchit<-rep(0,1,K)
        while(max(matchtab)>0){
          
          ij<-as.integer(which(matchtab==max(matchtab), arr.ind=T)[1,2])
          ik<-which.max(matchtab[,ij])
          matchit[ij]<-as.integer(rownames(as.matrix(ik)))
          matchtab[,ij]<-rep(-1,1,K)
          matchtab[as.integer(ik),]<-rep(-1,1,K)
          
        }
        matchit[which(matchit==0)]=which(!(1:K %in% unique(matchit)))
        knew <- cluster
        for(i in 1:K){
          knew[cluster==i] <- matchit[i]
        }
        cluster <- knew
        for (i in 1:K)
          t[which(cluster == i), i] <- 1
      }
      
    }
      
      
      else {if (init=="mini-em"){
        
        prms_best <- 1
        for (i in 1:mini.nb[1]){
          prms <- .T_funhddc_main1(fdobj,Wlist, K, known=known,dfstart = dfstart, dfupdate = dfupdate,
                                   dfconstr = dfconstr, model = model,
                                   threshold = threshold, method =  method,
                                   itermax = mini.nb[2],
                                   init.vector = 0,
                                   init = 'random', mini.nb = mini.nb,
                                   min.individuals = min.individuals,
                                   noise.ctrl = noise.ctrl,
                                   kmeans.control = kmeans.control,
                                   com_dim = com_dim, d_max = d_max, d_set = d_set)
          if(length(prms) != 1){
            if (length(prms_best) == 1) prms_best <- prms
            else if (prms_best$loglik[length(prms_best$loglik)] <
                     prms$loglik[length(prms$loglik)]) prms_best <- prms
          }
        }
        
        if (length(prms_best) == 1) return(1)
        t <- prms_best$posterior
        if(clas>0)
        {
          cluster <- max.col(t)
          cn1=length(unique(known[testindex]))
          matchtab=matrix(-1,K,K)
          matchtab[1:cn1,1:K] <- table(known,cluster)
          rownames(matchtab)<-c(rownames(table(known,cluster)),which(!(1:K %in% unique( known[testindex]))))
          matchit<-rep(0,1,K)
          while(max(matchtab)>0){
            
            ij<-as.integer(which(matchtab==max(matchtab), arr.ind=T)[1,2])
            ik<-which.max(matchtab[,ij])
            matchit[ij]<-as.integer(rownames(as.matrix(ik)))
            matchtab[,ij]<-rep(-1,1,K)
            matchtab[as.integer(ik),]<-rep(-1,1,K)
            
          }
          matchit[which(matchit==0)]=which(!(1:K %in% unique(matchit)))
          knew <- cluster
          for(i in 1:K){
            knew[cluster==i] <- matchit[i]
          }
          cluster <- knew
          for (i in 1:K)
            t[which(cluster == i), i] <- 1
        }
        
      }
        
        
        else {if (init=="random"){ #INIT IS RANDOM
          
          t <- t( rmultinom( N, 1, rep(1 / K, K) ) ) # some multinomial
          compteur <- 1
          while(min( colSums(t) ) < 1 && (compteur <- compteur + 1) < 5)
            t <- t( rmultinom( N, 1, rep(1 / K, K) ) )
          if(min( colSums(t) ) < 1)
            return("Random initialization failed (n too small)")
          if(clas>0)
          {
            cluster <- max.col(t)
            cn1=length(unique(known[testindex]))
            matchtab=matrix(-1,K,K)
            matchtab[1:cn1,1:K] <- table(known,cluster)
            rownames(matchtab)<-c(rownames(table(known,cluster)),which(!(1:K %in% unique( known[testindex]))))
            matchit<-rep(0,1,K)
            while(max(matchtab)>0){
              
              ij<-as.integer(which(matchtab==max(matchtab), arr.ind=T)[1,2])
              ik<-which.max(matchtab[,ij])
              matchit[ij]<-as.integer(rownames(as.matrix(ik)))
              matchtab[,ij]<-rep(-1,1,K)
              matchtab[as.integer(ik),]<-rep(-1,1,K)
              
            }
            matchit[which(matchit==0)]=which(!(1:K %in% unique(matchit)))
            knew <- cluster
            for(i in 1:K){
              knew[cluster==i] <- matchit[i]
            }
            cluster <- knew
            for (i in 1:K)
              t[which(cluster == i), i] <- 1
          }
          
        }
          
        }}}}}
  else {
    t <- matrix(1, N, 1) # K IS 1
    tw <- matrix(1, N, 1)
  }
  if(clas>0){
    t <- unkno*t
    for(i in 1:N){
      if(kno[i]==1){
        t[i, known[i]] <- 1
      }
    }
  }
  
  nux <- c( rep.int(dfstart, K) )
  initx <- .T_funhddt_init(fdobj,Wlist, K,t,nux, model, threshold, method, noise.ctrl,
                           com_dim, d_max, d_set)
  tw <- .T_funhddt_twinit(fdobj,Wlist, initx,nux)
  
  likely <- c()
  I <- 0
  test <- Inf
  while ( (I <- I + 1) <= itermax && test >= eps ){
    # loops here until itermax or test <eps
    # Error catching
    if (K > 1){
      if( any( is.na(t) ) ) return("unknown error: NA in t_ik")
      
      if( any(colSums(t > 1 / K) < min.individuals) )
        return("pop<min.individuals")
    }
    
    m <- .T_funhddt_m_step1(fdobj,Wlist, K, t, tw, nux, dfupdate, dfconstr, model,
                            threshold, method, noise.ctrl, com_dim, d_max, d_set) # mstep is applied here
    if(I == 1) {print(m$d)}
    nux <- m$nux
    
    to <- .T_funhddt_e_step1(fdobj,Wlist, m,clas, known, kno) # E-step is applied here
    L <- to$L #  this s the likelihood  
    t <- to$t# this is the new t
    tw <- to$tw
    
    
    #likelihood contrib=L
    likely[I] <- L
    
    if (I == 2) test <- abs(likely[I] - likely[I - 1])
    else if (I > 2)
    {
      lal <- (likely[I] - likely[I - 1]) / (likely[I - 1] - likely[I - 2])
      lbl <- likely[I - 1] + (likely[I] - likely[I - 1]) / (1.0 - lal)
      test <- abs(lbl - likely[I - 1])
    }
    
  }
  
  
  
  # a
  print(m$a)
  if ( model%in%c("AKBKQKDK", "AKBQKDK", "AKBKQKD", "AKBQKD") ) {
    a <- matrix(m$a[, 1], 1, m$K, dimnames=list(c("Ak:"), 1:m$K))
  } else if(model=="AJBQD") {
    a <- matrix(m$a[1, ], 1, m$d[1], dimnames=list(c('Aj:'), paste('a', 1:m$d[1], sep='')))
  } else if ( model%in%c("ABKQKDK", "ABQKDK", "ABKQKD", "ABQKD", "ABQD") ) {
    a <- matrix(m$a[1], dimnames=list(c('A:'), c('')))
  } else a <- matrix(m$a, m$K, max(m$d), dimnames=list('Class'=1:m$K, paste('a', 1:max(m$d), sep='')))
  
  # b
  if ( model%in%c("AKJBQKDK", "AKBQKDK", "ABQKDK", "AKJBQKD", "AKBQKD", "ABQKD",
                  "AJBQD", "ABQD") ) {
    b <- matrix(m$b[1], dimnames=list(c('B:'), c('')))
  } else b <- matrix(m$b, 1, m$K, dimnames=list(c("Bk:"), 1:m$K))
  
  # d, mu, prop
  d <- matrix(m$d, 1, m$K, dimnames=list(c('dim:'), "Intrinsic dimensions of the classes:"=1:m$K))
  mu <- matrix(m$mu, m$K, p, dimnames=list('Class'=1:m$K, 'Posterior group means:'=paste('V', 1:p, sep='')))
  prop <- matrix(m$prop, 1, m$K, dimnames=list(c(''), 'Posterior probabilities of groups'=1:m$K))
  nux <- matrix(m$nux, 1, m$K, dimnames=list(c(''), 'Degrees of freedom of t-distributions'=1:m$K))
  
  # Other elements
  complexity <- .T_hdc_getComplexityt(m, p, dfconstr)
  class(b) <- class(a) <- class(d) <- class(prop) <- class(mu) <- 'hd' #A@5 alpha and eta will nedd class too
  cls <- max.col(t)#@ here the cluster is found
  
  ## 
  converged = test < eps
  
  
  params <- list()
  params = c(params,list(
    Wlist=Wlist,
    model = model,
    K = K,
    d = d,
    a = a,
    b = b,
    mu = mu,
    prop = prop,
    nux = nux,
    ev = m$ev,
    Q = m$Q,
    Q1 = m$Q1,
    fpca = m$fpcaobj,
    loglik = likely[length(likely)],
    loglik_all = likely,
    posterior = t,
    class = cls,
    com_ev = com_ev,
    N = N,
    complexity = complexity,
    threshold = threshold,
    d_select = method,
    converged = converged))
  
  
  
  if(clas>0){
    params[["index"]] <- testindex
  }
  # We compute the BIC / ICL
  bic_icl <- .T_hdclassift_bic(params, p, dfconstr)
  params$BIC <- bic_icl$bic
  params$ICL <- bic_icl$icl
  
  # We set the class
  
  class(params) <- 'tfunHDDC'
  return(params)
} # end of .T_funhddc_main1

#########################################################
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

###################################################
############################################################
.T_funhddt_twinit  <- function(fdobj,Wlist, par,nux){ # this is the E step
  if (!inherits(fdobj, 'list')) {x <- t(fdobj$coefs)}  # THIS IS the univariate CASE
  else {x = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) x = cbind(x,t(fdobj[[i]]$coefs))} #NOT THIS
  p <- ncol(x)
  N <- nrow(x)
  K <- par$K
  a <- par$a
  b <- par$b
  mu <- par$mu
  d <- par$d
  Q <- par$Q
  Q1 <- par$Q1 
  W <- matrix(0, N, K)
  b[b<1e-6] <- 1e-6
  
  #############################aici am ajuns
  #################################################
  
  mah_pen<-matrix(0, N, K)
  
  
  for (i in 1:K) { 
    
    Qk <- Q1[[i]]
    
    aki <- sqrt( diag( c( 1 / a[ i, 1:d[i] ], rep(1 / b[i], p - d[i]) ) ) )
    muki <- mu[i,]
    
    Wki <- Wlist$W_m
    mah_pen[, i] <- .T_imahalanobis(x, muki, Wki ,Qk, aki)
    W[,i] <- (nux[i] + p) / (nux[i] + mah_pen[, i])
  }
  return(W)
} # end of .T_funhddt_twinit

######################################################################
############################################################
.T_funhddt_e_step1  <- function(fdobj,Wlist, par,clas=0,known=NULL, kno=NULL){ # this is the E step
  if (!inherits(fdobj, 'list')) {x <- t(fdobj$coefs)}  # THIS IS the univariate CASE
  else {x = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) x = cbind(x,t(fdobj[[i]]$coefs))} #NOT THIS
  p <- ncol(x)
  N <- nrow(x)
  K <- par$K
  nux <- par$nux
  a <- par$a
  b <- par$b
  mu <- par$mu
  d <- par$d
  prop <- par$prop
  Q <- par$Q
  Q1 <- par$Q1
  b[b<1e-6] <- 1e-6
  
  if(clas>0){
    unkno <- (kno-1)*(-1)
  }
  ##################################################
  #########################################################
  ###########################################
  t <- matrix(0, N, K)
  tw <- matrix(0, N, K)
  mah_pen <- matrix(0, N, K)
  K_pen <- matrix(0, N, K)
  num <- matrix(0, N, K)
  ft <- matrix(0, N, K)
  
  s <- rep(0, K)
  
  for (i in 1:K) { 
    s[i] <- sum( log(a[i, 1:d[i]]) ) 
    
    Qk <- Q1[[i]]
    aki <- sqrt( diag( c( 1 / a[i, 1:d[i]], rep(1 / b[i], p - d[i]) ) ) )
    muki <- mu[i, ]
    
    Wki <- Wlist$W_m
    dety<-Wlist$dety
    
    mah_pen[, i] <- .T_imahalanobis(x, muki, Wki ,Qk, aki)
    
    tw[, i] <- (nux[i] + p) / (nux[i] + mah_pen[, i])
    K_pen[, i] <- log(prop[i]) +
      lgamma( (nux[i] + p) / 2) - (1 / 2) * (s[i] + (p - d[i]) * log(b[i]) -log(dety)) -
      ( ( p / 2) * ( log(pi) + log(nux[i]) ) +
          lgamma(nux[i] / 2) + ( (nux[i] + p) / 2 ) *
          (log(1 + mah_pen[, i] / nux[i]) ) )
    
  }
  ft <- exp(K_pen)
  #ft_den used for likelihood
  ft_den <- rowSums(ft)
  kcon <- -apply(K_pen,1,max)
  K_pen <- K_pen + kcon
  num <- exp(K_pen)
  t <- num / rowSums(num)
  
  
  L1 <- sum( log(ft_den) )
  L <- sum(log( rowSums( exp(K_pen) ) ) - kcon)
  
  trow <- numeric(N)
  tcol <- numeric(K)
  trow <- rowSums(t)
  tcol <- colSums(t)
  if( any(tcol < p) ) { 
    t <- (t + 0.0000001) / ( trow + (K * 0.0000001) )
    
  }
  if(clas>0){
    t <- unkno*t
    for(i in 1:N){
      if(kno[i]==1){
        t[i, known[i]] <- 1
      }
    }
  }
  # if(any(is.nan(t))){
  #   break
  # }
  list(t = t,
       tw = tw,
       L = L)
} # end of .T_funhddt_e_step1

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
  
  print(bi)
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
#############################################
##################################################
#degrees of freedom functions modified yxf7 and yxf8 functions
# /*
# * Authors: Andrews, J. Wickins, J. Boers, N. McNicholas, P.
# * Date Taken: 2023-01-01
# * Original Source: teigen (modified)
# * Address: https://github.com/cran/teigen
# *
# */
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
#######################################################
############################################################
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
    #covariance matrix 
    mat_cov <- crossprod( t( .T_repmat(sqrt(CorI),
                                       n = dim( t(coef) )[[1]],p=1) * t(coef) ) ) / sum(Ti)
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


.T_hddc_ari <- function(x, y){
  #This function is drawn from the mclust package
  x <- as.vector(x)
  y <- as.vector(y)
  tab <- table(x, y)
  if ( all( dim(tab) == c(1, 1) ) ) return(1)
  a <- sum( choose(tab, 2) )
  b <- sum( choose(rowSums(tab), 2) ) - a
  c <- sum( choose(colSums(tab), 2) ) - a
  d <- choose(sum(tab), 2) - a - b - c
  ARI <- (a - (a + b) * (a + c)/(a + b + c + d))/((a + b + a + c)/2 - (a + b) * (a + c)/(a + b + c + d))
  return(ARI)
} # end of .T_hddc_ari



####
#### CONTROLS ####
####

.T_hddc_control = function(call){
  
  prefix = "HDDC: "
  .T_myCallAlerts(call, "data", "list,fd", 3, TRUE, prefix)
  .T_myCallAlerts(call, "K", "integerVectorGE1", 3, FALSE, prefix) # 
  .T_myCallAlerts(call, "known", "intNAVectorGE1", 3, FALSE, prefix) # 
  .T_myCallAlerts(call, "dfstart", "singleIntegerGE2", 3, FALSE, prefix)
  .T_myCallAlerts(call, "dfupdate", "singleCharacter", 3, FALSE, prefix)
  .T_myCallAlerts(call, "dfconstr", "singleCharacter", 3, FALSE, prefix)
  .T_myCallAlerts(call, "model", "characterVector", 3, FALSE, prefix)
  .T_myCallAlerts(call, "threshold", "numericVectorGE0LE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "criterion", "character", 3, FALSE, prefix)
  .T_myCallAlerts(call, "com_dim", "singleIntegerGE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "itermax", "singleIntegerGE2", 3, FALSE, prefix) # IAIN set to 2 iteration min
  .T_myCallAlerts(call, "eps", "singleNumericGE0", 3, FALSE, prefix)
  .T_myCallAlerts(call, "graph", "singleLogical", 3, FALSE, prefix)
  .T_myCallAlerts(call, "d_select", "singleCharacter", 3, FALSE, prefix)
  .T_myCallAlerts(call, "init", "singleCharacter", 3, FALSE, prefix)
  .T_myCallAlerts(call, "show", "singleLogical", 3, FALSE, prefix)
  .T_myCallAlerts(call, "mini.nb", "integerVectorGE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "min.individuals", "singleIntegerGE2", 3, FALSE, prefix)
  .T_myCallAlerts(call, "noise.ctrl", "singleNumericGE0", 3, FALSE, prefix)
  .T_myCallAlerts(call, "mc.cores", "singleIntegerGE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "nb.rep", "singleIntegerGE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "keepAllRes", "singleLogical", 3, FALSE, prefix)
  .T_myCallAlerts(call, "d_max", "singleIntegerGE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "d_range", "integerVectorGE1", 3, FALSE, prefix)
  .T_myCallAlerts(call, "verbose", "singleLogical", 3, FALSE, prefix)
  
  
  ####
  #### SPECIFIC controls
  ####
  
  # Getting some elements
  data = eval.parent(call[["data"]], 2)
  K = eval.parent(call[["K"]], 2)
  known = eval.parent(call[["known"]], 2)
  init = eval.parent(call[["init"]], 2)
  criterion = eval.parent(call[["criterion"]], 2)
  d_select = eval.parent(call[["d_select"]], 2)
  data_length = 0
  N = 0
  d_range = eval.parent(call[["d_range"]], 2)
  
  if (is.fd(data)){
    # No NA in the data:
    if (any(is.na(data$coefs))) stop("HDDC: NA values in the data are not supported. Please remove them beforehand.", call. = FALSE)
    if (any(!is.numeric(data$coefs))) stop("HDDC: Only numeric values are supported in fdata object. Please check coefficients.", call. = FALSE)
    if (any(!is.finite(data$coefs))) stop("HDDC: Only finite values are supported in fdata object. Please check coefficients.", call. = FALSE)
    # Size of the data
    if(any(K>2*NROW(data$coefs))) stop("HDDC: The number of observations must be at least twice the number of clusters.", call. = FALSE)
    
    data_length <- nrow(data$coefs)
    N <- ncol(data$coefs)
    
  }else{
    # No NA in the data:
    for (i in 1:length(data)){
      if(!is.fd(data[[i]])) stop("HDDC: All dimensions of data must be fd object.", call. = FALSE)
      if (any(is.na(data[[i]]$coefs))) stop("HDDC: NA values in the data are not supported. Please remove them beforehand.", call. = FALSE)
      if (any(!is.numeric(data[[i]]$coefs))) stop("HDDC: Only numeric values are supported in fdata object. Please check coefficients.", call. = FALSE)
      if (any(!is.finite(data[[i]]$coefs))) stop("HDDC: Only finite values are supported in fdata object. Please check coefficients.", call. = FALSE)
      data_length <- data_length + nrow(data[[i]]$coefs)
    }
    # Size of the data
    if(any(K>2*NROW(data[[1]]$coefs))) stop("HDDC: The number of observations must be at least twice the number of clusters ", call. = FALSE)
    N <- ncol(data[[1]]$coefs)
  }
  
  # Initialization Controls
  if(!is.null(init)){
    
    # we get the value of the initialization
    init = .T_myAlerts(init, "init", "singleCharacterMatch.arg", "HDDC: ", c('random', 'kmeans', 'tkmeans', 'mini-em', 'vector'))
    
    # Custom initialization => controls and setup
    if(init == "vector"){
      fdobj = data
      if (!inherits(fdobj, 'list')) {x = t(fdobj$coefs)}
      else {x = t(fdobj[[1]]$coefs); for (i in 2:length(fdobj)) x = cbind(x,t(fdobj[[i]]$coefs))}
      .T_myCallAlerts(call, "init.vector", "(integer,factor)Vector", 3, FALSE, prefix)
      
      init.vector = eval.parent(call[["init.vector"]], 2)
      
      if(is.null(init.vector)) stop("HDDC: When init='vector', the argument 'init.vector' should be provided.", call. = FALSE)
      
      if(length(unique(K))>1) stop("HDDC: Several number of classes K cannot be estimated when init='vector'.", call. = FALSE)
      
      init.vector <- unclass(init.vector)
      if(K!=max(init.vector)) stop("HDDC: The number of class K, and the number of classes in the initialization vector are different", call. = FALSE)
      
      if( length(init.vector)!=nrow(x) ) stop("HDDC: The size of the initialization vector is different of the size of the data", call. = FALSE)
    }
    
    # The param init A@5 is this even implemented????
    if (init=='param' && nrow(data)<ncol(data)){
      stop("HDDC: The 'param' initialization can't be done when N<p", call. = FALSE)
    }
    
    # The mini.em init
    if (init=='mini-em'){
      
      mini.nb = eval.parent(call[["mini.nb"]], 2)
      
      if(!is.null(mini.nb) && length(mini.nb)!=2){
        stop("HDDC: The parameter mini.nb must be a vector of length 2 with integers\n", call. = FALSE)
      }
      
    }
  }
  if(!is.null(d_select) && d_select == 'grid') {
    if(!is.null(d_range) && (max(d_range) > data_length)) stop("HDDC: Intrinsic dimension 'd' cannot be larger than number of input parameters. Please set a lower max.", call. = FALSE)
  }
  
  if(!is.null(known)) {
    if(all(is.na(known))) stop("HDDC: declared 'known' must have known values from each class (not all NA).", call. = F)
    if(length(known) != N) stop("HDDC: 'known' length must match the number of observations from data (known may include NA for observations where groups are unknown).", call. = F)
    if(length(K) > 1) stop("HDDC: only one group count can be used since known must have values for each group.", call. = F)
    if(length(unique(known[!is.na(known)])) > K) stop("HDDC: at most K different classes may be passed to 'known'.", call. = F)
    if(max(known, na.rm = T) > K) stop("HDDC: group numbers must come from integers up to K (ie. for K = 3 integers are from 1, 2, 3).", call. = F)
  }
}

.T_default_kmeans_control = function(control){
  
  .T_myAlerts(control,"kmeans.control","list","kmeans controls: ")
  
  #
  # Default values of the control parameters
  #
  
  myDefault = list()
  myDefault$iter.max = 10
  myDefault$nstart = 1
  myDefault$algorithm = c("Hartigan-Wong", "Lloyd", "Forgy","MacQueen")
  myDefault$trace = FALSE
  myDefault$alpha = 0.2
  
  #
  # Types of each arg
  #
  
  myTypes = c("singleIntegerGE1", "singleIntegerGE1", "match.arg",
              "singleLogical", "singleIntegerGE1")
  
  #
  # Recreation of the kmeans controls + Alerts
  #
  
  control = .T_matchTypeAndSetDefault(control, myDefault,
                                      myTypes, "kmeans list of controls: ")
  
  return(control)
}

#=================================#
# This file contains all the
# "control" functions
#=================================#

# Possible elements of myAlerts:
#
# /* BEGIN FROM FUNHDDC (UNMODIFIED) */
# /*
# * Authors: Bouveyron, C. Jacques, J.
# * Date Taken: 2022-01-01
# * Original Source: funHDDC (unmodified)
# * Address: https://github.com/cran/funHDDC
# *
# */

.T_myCallAlerts = function(call, name, myType, nParents=1, mustBeThere=FALSE, prefix=""){
  # This function basically calls the function myAlerts, but the arguments are different
  
  if( name %in% names(call) ){
    # we check the element exists => to provide a fine error
    what = call[[name]]
    val = try(eval.parent(what, nParents), silent = TRUE)
    # browser()
    if( "try-error" %in% class(val) ){
      if( inherits(what, 'name') ){
        # it means the variable was not found
        stop(prefix,"For argument '",name,"': object '",what,"' not found.", call. = FALSE)
      } else {
        stop(prefix,"For argument '",name,"': expression ",as.character(as.expression(what))," could not be evaluated.", call. = FALSE)
      }
      
    } else {
      a = .T_myAlerts(val, name, myType, prefix)
      return(a)
    }
  } else if(mustBeThere) {
    stop(prefix, "The argument '", name, "' must be provided.", call. = FALSE)
  }
}

.T_myAlerts = function(x, name, myType, prefix="", charVec){
  # Format of my types:
  #   - single => must be of lenght one
  #   - Vector => must be a vector
  #   - Matrix => must be a matrix
  #   - GE/GT/LE/LT: greater/lower than a given value
  #   - predefinedType => eg: numeric, integer, etc
  #   - match.arg => very specific => should match the charVec
  # If there is a parenthesis => the class must be of specified types:
  # ex: "(list, data.frame)" must be a list of a data.frame
  
  ignore.case = TRUE
  
  firstMsg = paste0(prefix,"The argument '",name,"' ")
  
  # simple function to extract a pattern
  # ex: if my type is VectorIntegerGE1 => myExtract("GE[[:digit:]]+","VectorIntegerGE1") => 1
  myExtract = function(expr, text, trim=2){
    start = gregexpr(expr,text)[[1]] + trim
    length = attr(start,"match.length") - trim
    res = substr(text,start,start+length-1)
    as.numeric(res)
  }
  
  #
  # General types handling
  #
  
  loType = tolower(myType)
  
  if(grepl("single",loType)){
    if(length(x)!=1) stop(firstMsg,"must be of length one.", call. = FALSE)
  }
  
  if(grepl("vector",loType) && !grepl("factor",loType)){
    if(!is.vector(x)) stop(firstMsg,"must be a vector.", call. = FALSE)
    if(is.list(x)) stop(firstMsg,"must be a vector (and not a list).", call. = FALSE)
  }
  
  res = .T_checkTheTypes(loType, x)
  if(!res$OK) stop(firstMsg,res$message, call. = FALSE)
  
  # GE: greater or equal // GT: greater than // LE: lower or equal // LT: lower than
  if(grepl("ge[[:digit:]]+",loType)){
    n = myExtract("ge[[:digit:]]+", loType)
    if( !all(x>=n, na.rm = T) ) stop(firstMsg,"must be greater than, or equal to, ", n,".", call. = FALSE)
  }
  if(grepl("gt[[:digit:]]+",loType)){
    n = myExtract("gt[[:digit:]]+", loType)
    if( !all(x>n, na.rm = T) ) stop(firstMsg,"must be strictly greater than ", n,".", call. = FALSE)
  }
  if(grepl("le[[:digit:]]+",loType)){
    n = myExtract("le[[:digit:]]+", loType)
    if( !all(x<=n, na.rm = T) ) stop(firstMsg,"must be lower than, or equal to, ",n,".", call. = FALSE)
  }
  if(grepl("lt[[:digit:]]+",loType)){
    n = myExtract("lt[[:digit:]]+", loType)
    if( !all(x<n, na.rm = T) ) stop(firstMsg,"must be strictly lower than ", n,".", call. = FALSE)
  }
  
  #
  # Specific Types Handling
  #
  
  if(grepl("match.arg",loType)){
    if(ignore.case){
      x = toupper(x)
      newCharVec = toupper(charVec)
    } else {
      newCharVec = charVec
    }
    
    if( is.na(pmatch(x, newCharVec)) ){
      n = length(charVec)
      if(n == 1){
        msg = paste0("'",charVec,"'")
      } else {
        msg = paste0("'", paste0(charVec[1:(n-1)], collapse="', '"), "' or '",charVec[n],"'")
      }
      stop(firstMsg, "must be one of:\n", msg, ".", call. = FALSE)
    } else {
      qui = pmatch(x, newCharVec)
      return(charVec[qui])
    }
  }
}

.T_matchTypeAndSetDefault = function(myList, myDefault, myTypes, prefix){
  # This  function:
  #   i) check that all the elements of the list are valid
  #   ii) put the default values if some values are missing 
  #   iii) Gives error messages if some types are wrong 
  # This function obliges  myList to be valid (as given by myDefault)
  
  # 1) check that the names of the list are valid
  if(is.null(myList)) myList = list()
  list_names = names(myList)
  
  if(length(list_names)!=length(myList) || any(list_names=="")){
    stop(prefix,"The elements of the list should be named.", call. = FALSE)
  }
  
  obj_names = names(myDefault)
  
  isHere = pmatch(list_names,obj_names)
  
  if(anyNA(isHere)){
    if(sum(is.na(isHere))==1) stop(prefix, "The following argument is not defined: ",paste(list_names[is.na(isHere)],sep=", "), call. = FALSE)
    else stop(prefix, "The following arguments are not defined: ",paste(list_names[is.na(isHere)],sep=", "), call. = FALSE)
  }
  
  # 2) We set the default values and run Warnings
  res = list()
  for(i in 1:length(obj_names)){
    obj = obj_names[i]
    qui = which(isHere==i) # qui vaut le numero de l'objet dans myList
    type = myTypes[i] # we extract the type => to control for "match.arg" type
    if(length(qui)==0){
      # we set to the default if it's missing
      if(type == "match.arg") {
        res[[obj]] = myDefault[[i]][1]
      } else {
        res[[obj]] = myDefault[[i]]
      }
    } else {
      # we check the integrity of the value
      val = myList[[qui]]
      if(type == "match.arg"){
        # If the value is to be a match.arg => we use our controls and not
        # directly the one of the function match.arg()
        charVec = myDefault[[i]]
        .T_myAlerts(val, obj, "singleCharacterMatch.arg", prefix, charVec)
        val = match.arg(val, charVec)
      } else {
        .T_myAlerts(val, obj, type, prefix)
      }
      
      res[[obj]] = val
    }
  }
  
  return(res)
}



.T_checkTheTypes = function(str, x){
  # This function takes in a character string describing the types of the
  # element x => it can be of several types
  
  # types that are controlled for:
  allTypes = c("numeric", "integer", "character", "logical", "list", "data.frame", "matrix", "factor", "intna")
  
  OK = FALSE
  message = c()
  
  for(type in allTypes){
    
    if(grepl(type, str)){
      # we add the type of the control
      if(type == "intna") {
        message = c(message, "integer or NA")
      } else {
        message = c(message, type)
      }
      
      if(type == "numeric"){
        if(!OK & is.numeric(x)){
          OK = TRUE
        }
      } else if(type == "integer"){
        if(is.numeric(x) && (is.integer(x) || (all(is.finite(x)) && all(x%%1==0)))){ # IAIN added non finite condition
          OK = TRUE
        }
      } else if(type == "intna"){
        if(is.numeric(x[!is.na(x)]) && (is.integer(x[!is.na(x)]) || (all(is.finite(x[!is.na(x)])) && all(x[!is.na(x)]%%1==0)))){ # IAIN added non finite condition
          OK = TRUE
        }
      } else if(type == "character"){
        if(is.character(x)){
          OK = TRUE
        }
      } else if(type == "logical"){
        if(is.logical(x)){
          OK = TRUE
        }
      } else if(type == "list"){
        if(is.list(x)){
          OK = TRUE
        }
      } else if(type == "data.frame"){
        if(is.data.frame(x)){
          OK=TRUE
        }
      } else if(type == "matrix"){
        if(is.matrix(x)){
          OK = TRUE
        }
      } else if(type == "factor"){
        if(is.factor(x)){
          OK = TRUE
        }
      }
      
    }
    
    if(OK) break
  }
  
  if(length(message) == 0) OK = TRUE #ie there is no type to be searched
  else if(length(message) >= 3){
    n = length(message)
    message = paste0("must be of type: ",  paste0(message[1:(n-1)], collapse = ", "), " or ", message[n], ".")
  } else {
    message = paste0("must be of type: ",  paste0(message, collapse = " or "), ".")
  }
  
  
  return(list(OK=OK, message=message))
}




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
      d <- apply((dev>threshold)*(1:(p-1))*t(ev[, -1]>noise.ctrl), 2, which.max)
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

.T_hdclassift_bic <- function(par, p, dfconstr,data=NULL){
  model <- par$model
  K <- par$K
  d <- par$d
  b <- par$b
  a <- par$a
  mu <- par$mu
  N <- par$N
  prop <- par$prop
  mux<-par$mux
  if(length(b)==1){
    #update of b to set it as variable dimension models
    eps <- sum(prop*d)
    n_max <-  ncol(par$ev)
    b <- b*(n_max-eps)/(p-eps)
    b <- rep(b, length=K)
  }
  if (length(a)==1) a <- matrix(a, K, max(d))
  else if (length(a)==K) a <- matrix(a, K, max(d))
  else if (model=='AJBQD') a <- matrix(a, K, d[1], byrow=TRUE)
  
  if(min(a, na.rm=TRUE)<=0 | any(b<0)) return(-Inf)
  
  if (is.null(par$loglik)){
    som_a <- c()
    
    for (i in 1:K) som_a[i] <- sum(log(a[i, 1:d[i]]))
    L <- -1/2*sum(prop * (som_a + (p-d)*log(b) - 2*log(prop)+ p*(1+log(2*pi))))*N
  }
  else  L <- par$loglik[length(par$loglik)]
  
  # add K or 1 parameters for nux in ro
  if (dfconstr=="no")
    ro <- K*(p+1)+K-1
  else
    ro <- K*p+K
  tot <- sum(d*(p-(d+1)/2))
  D <- sum(d)
  d <- d[1]
  to <- d*(p-(d+1)/2)
  if (model=='AKJBKQKDK') m <- ro+tot+D+K
  else if (model=='AKBKQKDK') m <- ro+tot+2*K
  else if (model=='ABKQKDK') m <- ro+tot+K+1
  else if (model=='AKJBQKDK') m <- ro+tot+D+1
  else if (model=='AKBQKDK') m <- ro+tot+K+1
  else if (model=='ABQKDK') m <- ro+tot+2
  bic <- -(-2*L+m*log(N))
  
  #calcul ICL
  t = par$posterior
  
  Z = ((t - apply(t, 1, max))==0) + 0
  icl = bic - 2*sum(Z*log(t+1e-15))
  
  
  return(list(bic = bic, icl = icl))
}

.T_hdc_getComplexityt = function(par, p, dfconstr){
  model <- par$model
  K <- par$K
  d <- par$d
  b <- par$b
  a <- par$a
  mu <- par$mu
  
  prop <- par$prop
  #@ add in ro K parameters for nuk or add 1 parameter if degrees equal
  if (dfconstr=="no")
    ro <- K*(p+1)+K-1
  else
    ro <- K*p+K
  tot <- sum(d*(p-(d+1)/2))
  D <- sum(d)
  d <- d[1]
  to <- d*(p-(d+1)/2)
  if (model=='AKJBKQKDK') m <- ro+tot+D+K
  else if (model=='AKBKQKDK') m <- ro+tot+2*K
  else if (model=='ABKQKDK') m <- ro+tot+K+1
  else if (model=='AKJBQKDK') m <- ro+tot+D+1
  else if (model=='AKBQKDK') m <- ro+tot+K+1
  else if (model=='ABQKDK') m <- ro+tot+2
  
  return(m)
}

.T_hdc_getTheModel = function(model, all2models = FALSE){
  # Function used to get the models from number or names
  
  model_in = model
  
  if(!is.vector(model)) stop("The argument 'model' must be a vector.")
  
  if(anyNA(model)) stop("The argument 'model' must not contain any NA.")
  
  ModelNames <- c("AKJBKQKDK", "AKBKQKDK", "ABKQKDK", "AKJBQKDK", "AKBQKDK", "ABQKDK", "AKJBKQKD", "AKBKQKD", "ABKQKD", "AKJBQKD", "AKBQKD", "ABQKD")
  
  model = toupper(model)
  
  if(length(model)==1 && model=="ALL"){
    if(all2models) model <- 1:14
    else return("ALL")
  }
  
  qui = which(model %in% 1:14)
  model[qui] = ModelNames[as.numeric(model[qui])]
  
  # We order the models properly
  qui = which(!model%in%ModelNames)
  if (length(qui)>0){
    if(length(qui)==1){
      msg = paste0("(e.g. ", model_in[qui], " is incorrect.)")
    } else {
      msg = paste0("(e.g. ", paste0(model_in[qui[1:2]], collapse=", or "), " are incorrect.)")
    }
    stop("Invalid model name ", msg)
  }
  
  # warning:
  if(max(table(model))>1) warning("The model vector, argument 'model', is made unique (repeated values are not tolerated).")
  
  mod_num <- c()
  for(i in 1:length(model)) mod_num[i] <- which(model[i]==ModelNames)
  mod_num <- sort(unique(mod_num))
  model <- ModelNames[mod_num]
  
  return(model)
}



####
#### Utilities ####
####


.T_addCommas = function(x) sapply(x, .T_addCommas_single )

.T_addCommas_single = function(x){
  # This function adds commas for clarity for very long values of likelihood 
  
  if(!is.finite(x)) return(as.character(x))
  
  s = sign(x)
  x = abs(x)
  
  decimal = x - floor(x)
  if(decimal>0) dec_string = substr(decimal, 2, 4)
  else dec_string = ""
  
  entier = as.character(floor(x))
  
  quoi = rev(strsplit(entier, "")[[1]])
  n = length(quoi)
  sol = c()
  for(i in 1:n){
    sol = c(sol, quoi[i])
    if(i%%3 == 0 && i!=n) sol = c(sol, ",")
  }
  
  res = paste0(ifelse(s==-1, "-", ""), paste0(rev(sol), collapse=""), dec_string)
  res
}


.T_repmat <- function(v,n,p){ #A@5 WHAT IS THIS???????????????????????
  if (p==1){M = cbind(rep(1,n)) %*% v} #A@5 a matrix of column of v
  else { M = matrix(rep(v,n),n,(length(v)*p),byrow=T)} # removed cat("!");
  M
}

.T_diago <- function(v){
  if (length(v)==1){ res = v }
  else { res = diag(v)}
  res
}

# /* END OF FROM FUNHDDC */

.T_imahalanobis <- function(x, muk, wk, Qk, aki) {
  # w = fpcaobj[[i]]$W
  # *************************************************************************** #
  # should be called as imahalanobis(x, mu[i,], w[i,], Q[[i]], a[i,], b[i])
  # return formula on top of page 9
  # *************************************************************************** #
  
  # Code modified to take vectors instead of matrices
  p <- ncol(x)
  N <- nrow(x)
  res <- rep(0, N)
  
  
  
  #X <- x - matrix(muk, N, p, byrow=TRUE)
  
  #  Qi <- wk %*% Qk
  
  
  # xQi <- (X %*% Qi)
  
  #proj <- (X %*% Qi) %*% aki
  #the R result- no C code
  # res_old <- rowSums(proj ^ 2)
  
  
  
  # Calling C_imahalanobis
  res <- .C(".C_imahalanobis", as.double(x), as.double(muk), as.double(wk),
            as.double(Qk), as.double(aki), as.integer(p),
            as.integer(N), as.integer(nrow(aki)), res = rep(0,N), PACKAGE="TFunHDDC")
  
  
  return(res$res)
  
} # end of .T_imahalanobis

data = fitNOxBenchmark()$fd
W = inprod(data$basis, data$basis)
W[W<1e-15] = 0
W_m = chol(W)
dety = det(W)
W_list = list(W = W, W_m = W_m, dety = dety)
noise = 1e-8
threshold = 0.1
d_set = c(2,2,2,2)
K = 3
D_max = 100
df_start=50
itermax =200
vec = rep(3, ncol(data$coefs))
vec[1:50] = 1
vec[51:100] = 2

res = .T_funhddc_main1(fdobj = data, Wlist=W_list, K=K, dfstart=df_start, dfupdate='numeric', dfconstr = 'yes', model='AKJBKQKDK', itermax=itermax, threshold=threshold, method='bic', eps=1e-6, init='vector', init.vector=vec, mini.nb=NULL, min.individuals=0, noise.ctrl=noise, com_dim=NULL, kmeans.control=NULL, d_max=D_max, d_set=d_set, known=NULL)

print(res$BIC)
print(res$ICL)
print(res$class)