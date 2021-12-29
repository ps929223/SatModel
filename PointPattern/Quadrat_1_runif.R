
################################
# Quadrat Analysis(방격분석)
# 균일분포난수 runif
################################

library(spatstat)

#--------------------------------
# 자료입력 및 구성
#--------------------------------
xx <-matrix(runif(50))
yy <-matrix(runif(50))

aa <- ppp(xx,yy) # Planar point pattern # Quadrat(5,5)
print(aa)

aq <- quadratcount(aa, 4,3) # Quadrat(4,3)
print(aq)

plot(xx, yy, type='n', main='Spatial Point Distribution')
points(aa)
plot(aq)


#--------------------------------
# 분석
# CSR(complete spatial randomness)분포에 따른 자료(PPP)와
# 분석대상 자료 비교분석
#--------------------------------

t1=quadrat.test(aa) # Planar point pattern # Quadrat(5,5)
print(t1) # p>0.05 then Randomness

t2=quadrat.test(aq) # Quadrat(4,3)
print(t2)  # p>0.05 then Randomness



#--------------------------------
# 가시화
#--------------------------------

plot(t1,main='Quadrat Test: Default(5*5)')
points(aa,col='red',pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t1$p.value,3))))
title(sub=ss, cex.sub=1.2)

plot(t1,main='Quadrat Test: Arbitrary(4*3)')
points(aa,col='red',pch=19)
ss=eval(substitute(expression(p[chi^2]==z),list(z=signif(t2$p.value,3))))
title(sub=ss, cex.sub=1.2)


