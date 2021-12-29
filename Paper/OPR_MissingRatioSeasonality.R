## 결측값의 계절성
# https://m.blog.naver.com/haiena21/221751555833


# 데이터 읽기
path_csv='E:/20_Product/GOCI1/CHL/MissingRatio/MR_GOCI1_CHL_DM3_2018-03-04_2021-03-26.csv'
DF=read.csv(file=path_csv, header=TRUE)


# 월평균 결측률 최대/최소인 월의 데이터 추출
idx_201812=grepl('2018-12', DF$Time) # 31개
idx_201808=grepl('2018-08', DF$Time) # 31개

MR_201812=DF$MR[idx_201812]
MR_201808=DF$MR[idx_201808]


# 요약
summary(MR_201812)
summary(MR_201808)


# 등분산 검정
var.test(MR_201808, MR_201812)
# data:  MR_201808 and MR_201812
# F = 0.053239, num df = 30, denom df = 30, p-value = 2.828e-12
# p<0.05이므로 등분산이라는 영가설을 기각하지 못함: 즉, 등분산임
# alternative hypothesis: true ratio of variances is not equal to 1
# 95 percent confidence interval:
#   0.02567048 0.11041496
# sample estimates:
#   ratio of variances 
# 0.05323913 

# t.test # 등분산인 경우
t.test(MR_201808, MR_201812, var.equal=T)

# t.test # 등분산 아닌 경우
t.test(MR_201808, MR_201812, var.equal=F)
