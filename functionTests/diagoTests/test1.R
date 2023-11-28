.T_diago <- function(v){
  if (length(v)==1){ res = v }
  else { res = diag(v)}
  res
}

a = c(1,2,3)
b = c(1)
c = c(1,2,3,4,5,6,7)
d = c(1.2, 3.4, 5.6)
e = matrix(c(1,2,3,4,5,6), 2,3, byrow=T)
f = matrix(c(1,2,3,4,5,6,7,8,9), 3, 3, byrow=TRUE)

tests = list(a = a, b=b, c=c, d=d, e=e, f=f)

for(i in tests){
  print(.T_diago(i))
}