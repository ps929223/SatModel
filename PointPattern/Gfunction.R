'''
공간의 점 분포에 대한 임의성 검정(G Function)
'''
# install.packages('spatstat')
# install.packages('RNetCDF')
library(spatstat)


## 예제 ##
# Obtain the bramble cane data
data("bramblecanes") # 검은 산딸기
plot(bramblecanes)
Gf <-Gest(bramblecanes, correction='border')
# 'border' 주변(edge)-교정된 추정값(Gd)
plot(Gf)

# 포락분석(envelope analysis)
Gf.env <- envelope(bramblecanes, Gest, correction='border')
plot(Gf.env)

# MAD Test
mad.test(bramblecanes, Gest)

# Dclf Test
dclf.test(bramblecanes, Gest)
