
.T_hddc_ari <- function(x, y){
  #This function is drawn from the mclust package
  x <- as.vector(x)
  y <- as.vector(y)
  tab <- table(x, y)
  if ( all( dim(tab) == c(1, 1) ) ) return(1)
  a <- sum( choose(tab, 2) )
  print(a)
  b <- sum( choose(rowSums(tab), 2) ) - a
  c <- sum( choose(colSums(tab), 2) ) - a
  d <- choose(sum(tab), 2) - a - b - c
  ARI <- (a - (a + b) * (a + c)/(a + b + c + d))/((a + b + a + c)/2 - (a + b) * (a + c)/(a + b + c + d))
  return(ARI)
} # end of .T_hddc_ari

a = c(66, 3, 77, 53, 10, 4)
b = c(66, 3, 66, 52, 3, 3)

c = .T_hddc_ari(a,b)
