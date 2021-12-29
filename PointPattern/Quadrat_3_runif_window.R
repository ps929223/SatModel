################################
# Quadrat Analysis(방격분석)
# 균일분포난수 runif, window 변화
################################

library(spatstat)

#--------------------------------
# 자료입력 및 구성
#--------------------------------
x3 <-matrix(runif(50))
y3 <-matrix(runif(50))

ya <- ppp(x3,y3)
yb <- ya[owin(c(-0.2,1.2),c(-0.2,1.2))]
yc <- ppp(x3,y3,poly=list(x=c(-0.7,1.7,0.5),y=c(0,0,1.7)))


#--------------------------------
# 분석
#--------------------------------

t_ya=quadrat.test(ya)
print(t_ya)
t_yb=quadrat.test(yb)
print(t_yb)
t_yc=quadrat.test(yc)
print(t_yc)


#--------------------------------
# 가시화
#--------------------------------

plot(t_ya, main='Quadrat Test: Random')
points(ya, col='red', pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t_ya$p.value,3))))
title(sub=ss, cex.sub=1.2)

plot(t_yb, main='Quadrat Test: Random')
points(yb, col='red', pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t_yb$p.value,3))))
title(sub=ss, cex.sub=1.2)

plot(t_yc, main='Quadrat Test: One-clustered')
points(yc, col='red', pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t_yc$p.value,3))))
title(sub=ss, cex.sub=1.2)
