################################
# Quadrat Analysis(방격분석)
# random, collective
################################

library(spatstat)

#--------------------------------
# Random
# 자료입력 및 구성
#--------------------------------

x2<-matrix(rnorm(50, mean=.5, sd=0.15))
y2<-matrix(rnorm(50, mean=.5, sd=0.15))

xt = cut(x2, seq(0,1,.1)) # 0.1 단위로 범위 나타내기
yt = cut(y2, seq(0,1,.1))
my.quadrat=table(xt,yt) # 0.1 단위로 x,y축 2D hist. == Quadrat

# 분석자료
count=as.vector(my.quadrat) # x,y축 방향 밀도값을 count로 반환 == 분포구하기기
print(count)
(cc=table(count)) # 밀도의 최소최대값 사이에서 Histogram

## 분산평균비: 푸아송분포 같다..
var(count)/mean(count)

## 푸아송분포: 이론값
exp1=dpois(0:5,lambda = 1)*100
hist(exp1)
## Chi-square 검정. 귀무가설: 관측된 데이터가 비교데이터(푸아송분포)와 다르지 않다.
ct=chisq.test(cc,p=exp1,rescale.p=T)
print(ct)

## 음이항분포
exp2=dnbinom(0:5,size=mean(count)^2/(var(count)-mean(count)), mu=mean(count))*100

## Chi-square 검정. 귀무가설: 관측된 데이터가 비교데이터(음이항분포)와 다르지 않다.
ct=chisq.test(cc,p=exp2,rescale.p=T)
print(ct)
