
.T_hddc_ari <- function(x, y){
  #This function is drawn from the mclust package
  x <- as.vector(x)
  y <- as.vector(y)
  tab <- table(x, y)
  print(tab)
  if ( all( dim(tab) == c(1, 1) ) ) return(1)
  a <- sum( choose(tab, 2) )
  print(a)
  b <- sum( choose(rowSums(tab), 2) ) - a
  print(b)
  c <- sum( choose(colSums(tab), 2) ) - a
  print(c)
  d <- choose(sum(tab), 2) - a - b - c
  print(d)
  ARI <- (a - (a + b) * (a + c)/(a + b + c + d))/((a + b + a + c)/2 - (a + b) * (a + c)/(a + b + c + d))
  return(ARI)
} # end of .T_hddc_ari

a = c(6.64e-55, 45.3e-32, 45.3e-32, 77.689e-69, 6.64e-55, 95.89e-60)
b = c(6.63e-45, 45.3e-31, 45.3e-32, 77.689e-69, 6.64e-55, 89.89e-89)

c = .T_hddc_ari(a,b)