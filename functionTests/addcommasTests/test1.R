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

a = 123.45
b = 123
c = 123.456
d = 4000.45
e = 4000
f = 4000.456
g = 10000000000
h = 10000000000.123456789
i = -300000.1234
j = -300000
tests = c(a,b,c,d,e,f,g,h,i,j)

print(.T_addCommas(tests))