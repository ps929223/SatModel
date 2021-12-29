
#--------------------------------
# Edge Effect
#--------------------------------

x=runif(120); y=runif(120);

plot(x,y,pch=21, bg='green')

dis<-function(x1,y1,x2,y2){
  sqrt((x2-x1)^2+(y2-y1)^2)
}

r=nn=d=numeric(120)


for (ii in 1:120){
  for (jj in 1:120){
    d[jj]=dis(x[ii],y[ii],x[jj],y[jj])
  }
  r[ii]=min(d[-ii]);  nn[ii]=which(d==min(d[-ii]))
}


for (ii in 1:100){
  lines(c(x[ii],x[nn[ii]]),c(y[ii],y[nn[ii]]), col='blue')
}
title(main='Lines Between Nearest Neighbor Points')
edge = pmin(x,y,1-y,1-x)
plot(x,y,pch=16)
id = which(edge<r)

points(x[id],y[id],col='red',cex=2,lwd=2)
title(main='Edge Effect')
mean(r)
mean(r[-id])

