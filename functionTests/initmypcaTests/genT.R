set.seed(1009)
known = NULL
kno = NULL
testindex = NULL
N = 115
K=3
clas = 0

#Create T matrix using init=random
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
