
################################
# Randomness Index
################################

my.dis<-function(x1,y1,x2,y2){
  sqrt((x2-x1)^2+(y2-y1)^2)
}

my.flatten<-function(matrix,transverse){
  if (transverse==TRUE){
    as.vector(t(matrix))
  }
  else{
    as.vector(matrix)
    }
}

my.RI<-function(r,N){
  RI=2*mean(r)/sqrt(1/N)
}

#--------------------------------
# Regular 분포
#--------------------------------
# install.packages('pracma')
# install.packages('purrr')
library(pracma)
library(purrr)

x=y=linspace(0,1,n=10)
mesh=meshgrid(x,y)
plot(my.flatten(mesh$X,TRUE),my.flatten(mesh$Y,TRUE),pch=21, bg='green')

r=nn=d=numeric(length(x))


for (ii in 1:length(x)){
  for (jj in 1:length(x)){
    d[jj]=my.dis(x[ii],y[ii],x[jj],y[jj])
  }
  r[ii]=min(d[-ii]);  nn[ii]=which(d==min(d[-ii]))
}
N=length(my.flatten(mesh$X,transverse = TRUE))
print(my.RI(r,N))



#--------------------------------
# Random 분포
#--------------------------------

x=runif(100); y=runif(100);
plot(x,y,pch=21, bg='green')

r=nn=d=numeric(length(x))

for (ii in 1:length(x)){
  for (jj in 1:length(x)){
    d[jj]=my.dis(x[ii],y[ii],x[jj],y[jj])
  }
  r[ii]=min(d[-ii]);  nn[ii]=which(d==min(d[-ii]))
}

N=length(my.flatten(mesh$X,transverse = TRUE))
print(my.RI(r,N))


#--------------------------------
# Cluster 분포
#--------------------------------

x=rnorm(n=100, mean=.5, sd=.1)
y=rnorm(n=100, mean=.5, sd=.1)
plot(x,y,pch=21, bg='green')

r=nn=d=numeric(length(x))

for (ii in 1:length(x)){
  for (jj in 1:length(x)){
    d[jj]=my.dis(x[ii],y[ii],x[jj],y[jj])
  }
  r[ii]=min(d[-ii]);  nn[ii]=which(d==min(d[-ii]))
}

N=length(my.flatten(mesh$X,transverse = TRUE))
print(my.RI(r,N))


