################################
# Quadrat Analysis(방격분석)
# 균일분포난수 runif&rnorm
################################

library(spatstat)

#--------------------------------
# 자료입력 및 구성
#--------------------------------
x1 <-matrix(runif(50))
y1 <-matrix(runif(50))
xa <- ppp(x1,y1) # Planar point pattern # Quadrat(5,5)

x2 <-matrix(rnorm(50, mean=.5, sd=.15))
y2 <-matrix(rnorm(50, mean=.5, sd=.15))
xb <- ppp(x2,y2) # Planar point pattern # Quadrat(5,5)

#--------------------------------
# 분석
#--------------------------------

t1=quadrat.test(xa)
t2=quadrat.test(xb)

#--------------------------------
# 가시화
#--------------------------------


plot(t1, main='Quadrat Test: Random')
points(xa, col='red', pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t1$p.value,3))))
title(sub=ss, cex.sub=1.2)

plot(t2, main='Quadrat Test: One-clustered')
points(xb, col='red', pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t2$p.value,3))))
title(sub=ss, cex.sub=1.2)

