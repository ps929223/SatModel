
''' Outlier Removal '''

setwd('D:/20_Product/Buoy/Dokdo')

path_input_csv="Dokdo_recent.csv"
data <-read.csv(path_input_csv)

col_names=colnames(data)

TGT_Data <- data$S37Tmp01

TGT_Data[TGT_Data<5]=NaN
table(is.na(TGT_Data))

### forecast
# install.packages('forecast')
library(forecast)


## NaN이 있으면 idx 반환
for (ii in 1:length(TGT_Data)) ifelse(is.na(TGT_Data[ii]),print(ii), next)

## Outlier 값 반환
TGT_Data[tsoutliers(TGT_Data, iterate = 5)$index]=NaN
new_Data=na.interp(TGT_Data)

## 비교

new_Data=tsclean(TGT_Data)
tsdisplay(TGT_Data)
tsdisplay(new_Data)

DF <- data.frame(data$yymmddHHMMSS,new_Data)
name_var='S37Tmp01'
colnames(DF) <- c('Time',name_var)
write.csv(DF, file=paste0('Dokdo_recent_',name_var,'.csv'), row.names=FALSE)



### Thompson 통계량
tom = function(x){
  nn=length(x)
  mu=mean(x)
  ss=sd(x,1)
  t1=(x-mu)/ss
  (sqrt(nn-2)*t1)/(sqrt(nn-1-t1*t1))
  }

score=tom(TGT_Data)
tt1= qt(0.99,length(TGT_Data)-2)
tt2= qt(0.95,length(TGT_Data)-2)

new_Data=TGT_Data[score<tt2]
tsdisplay(new_Data)
