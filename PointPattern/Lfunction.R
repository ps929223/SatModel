'''
공간의 점 분포에 대한 임의성 검정(L Function)
'''
# install.packages('spatstat')
# install.packages('RNetCDF')
library(spatstat)


## 예제 ##
# Obtain the bramble cane data
data("bramblecanes") # 검은 산딸기
plot(bramblecanes)
Lf <-Lest(bramblecanes, correction='border')
# 'border' 주변(edge)-교정된 추정값(Kd)
plot(Lf)

# 포락분석(envelope analysis)
Lf.env <- envelope(bramblecanes, Lest, correction='border')
plot(Lf.env)

# MAD Test
mad.test(bramblecanes, Lest)

# Dclf Test
dclf.test(bramblecanes, Lest)
