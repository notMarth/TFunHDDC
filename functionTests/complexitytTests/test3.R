.T_hdc_getComplexityt = function(par, p, dfconstr){
  model <- par$model
  K <- par$K
  d <- par$d
  #b <- par$b
  #a <- par$a
  #mu <- par$mu
  
  #prop <- par$prop
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

par = list(model = "ABKQKDK", K = 3, d = c(3,2,2))
res1 = .T_hdc_getComplexityt(par, 3, 'no')
res2 = .T_hdc_getComplexityt(par, 3, 'yes')
print(res1)
print(res2)