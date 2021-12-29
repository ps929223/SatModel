'''
공간의 점 분포에 대한 임의성 검정(Ripley`s K Function)
'''
# install.packages('spatstat')
# install.packages('RNetCDF')
library(spatstat)

## 예제 ##
# Obtain the bramble cane data
data("bramblecanes") # 검은 산딸기
plot(bramblecanes)
Kf <-Kest(bramblecanes, correction='border')
# 'border' 주변(edge)-교정된 추정값(Kd)
plot(Kf)

# 포락분석(envelope analysis)
Kf.env <- envelope(bramblecanes, Kest, correction='border')
plot(Kf.env)

# MAD Test
mad.test(bramblecanes, Kest)

# Dclf Test
dclf.test(bramblecanes, Kest)
