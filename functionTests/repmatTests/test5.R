.T_repmat <- function(v,n,p){ #A@5 WHAT IS THIS???????????????????????
  if (p==1){M = cbind(rep(1,n)) %*% v} #A@5 a matrix of column of v
  else { M = matrix(rep(v,n),n,(length(v)*p),byrow=T)} # removed cat("!");
  M
}

v = matrix(c(1,2,3,4,5,6,7,8), 2, 4, byrow=T)

res = .T_repmat(v, 2, 3)
print(res)