
##----------------------------------------------
## OPR논문용
## GOCI1 Chl-a의 공간의 점 분포 패턴 분석
##----------------------------------------------

# install.packages('rlist')
# install.packages('ncdf4')
library(rlist)
library(ncdf4)
library(spatstat)


boolean.integer <- function(arg1) {
  ## Boolean을 정수로 변경
  arg1 [arg1 == "true"] <- 1
  arg1 [arg1 == "false"] <- 0
  arg1 <- as.integer(arg1)
}


file_list.DM <-function(path_source_dir,pattern){
  ## 경로와 패턴을 정해주면 목록을 반환  
  file_list=list.files(path = path_source_dir, full.names = TRUE, 
                       recursive = TRUE, pattern=pattern)
  help(list.files)
  DM1_list= grep(pattern, file_list, value=TRUE)
  Dokdo=matrix(NaN,length(file_list),1)
  res100=matrix(NaN,length(file_list),1)
  
  ## 필터링 조건
  for (ii in 1:length(file_list)){
    res100[ii]=grepl('y86x88', file_list[ii], fixed = TRUE)
  }
  
  Mask=as.logical(res100)
  
  file_list=file_list[Mask]
}



get_date.fromPath <- function(path_file){
  #---------------------------
  #경로에서 날짜를 가져옴
  #---------------------------
  
  tp=strsplit(path_file,'/')[[1]][7]
  date=strsplit(tp,'_')[[1]][4]
  # temp_list <- unlist(strsplit(DMn_list,'/'))
  # temp_list[length(temp_list)]
}


get_ptspattern <- function(path_file){
  #---------------------------
  #경로를 주면 K,L,G function 계산하고 그림을 그려 냄
  #---------------------------
  
  # path_file="E:/20_Product/CHL_DM/CHL_DM3/2019/01/GOCI1_DM3_CHL_2019-01-01_y86x88.nc"
  nc <- nc_open(path_file)
  
  nc_vars <- names(nc$var)
  
  #set coordinate variable: lon, lat
  lat <- ncvar_get(nc,"lat")
  lon <- ncvar_get(nc, "lon")
  chl <-  ncvar_get(nc, 'chl')
  lm <-  ncvar_get(nc, 'lm')
  noa <-  ncvar_get(nc, 'NonObsArea')
  
  
  path_dir='E:/20_Product/GOCI1/CHL/PointDistribution'
  file_name=strsplit(path_file,'/')[[1]][7]
  beforeExtension=substr(file_name,1,nchar(file_name)-3)
  path_img=file.path(path_dir,paste0(beforeExtension,'.png'))
  
  png(path_img, height=800, width=800)
  
  
  par(mfrow=c(2,2), cex.lab=2, cex.axis=2, cex.main=3)
  
  # cond_target=((lm==0)&(noa==0))&(!is.na(chl)) # 관측영역
  
  cond_target=((lm==0)&(noa==0))&(is.na(chl)) # 결측영역만
  
  plot(lon[cond_target],lat[cond_target], pch=20, cex=1, 
       xlab='Lon', ylab='Lat', main='Black: Cloud')
  
  data=ppp(lon[cond_target], lat[cond_target],
           c(min(lon),max(lon)),c(min(lat),max(lat)))
  
  # Kf <-Kest(data, correction='border')
  # 'border' 주변(edge)-교정된 추정값(Kd)
  # plot(Kf, main='Kf')
  
  # 포락분석(envelope analysis)
  Kf.env <- envelope(data, Kest, correction='border')
  plot(Kf.env, main='KfEnv')
  # MAD Test
  # MAD=mad.test(data, Kest)
  # Dclf Test
  # DCLF=dclf.test(bramblecanes, Kest)
  
  # Lf <-Lest(data, correction='border')
  # 'border' 주변(edge)-교정된 추정값(Ld)
  # plot(Lf, main='Lf')
  
  # 포락분석(envelope analysis)
  Lf.env <- envelope(data, Lest, correction='border')
  plot(Lf.env, main='LfEnv')
  # MAD Test
  # MAD=mad.test(data, Kest)
  # Dclf Test
  # DCLF=dclf.test(bramblecanes, Kest)
  
  # 포락분석(envelope analysis)
  Gf.env <- envelope(data, Gest, correction='border')
  plot(Gf.env, main='GfEnv')
  
  dev.off()
}


  
## DM1 기준 파일 목록
DM1_list = file_list.DM(path_source_dir='E:/20_Product/CHL_DM/CHL_DM1',pattern='_y86x88.nc')
date_list = c()

## 날짜 목록 추출
# ii = 18
for (ii in 411:length(DM1_list)){
  date_list[ii]<-get_date.fromPath(unlist(DM1_list)[[ii]])
  
  ## 특정 날짜 기준 파일 목록
  DMn_list= file_list.DM(path_source_dir='E:/20_Product/CHL_DM',
                         pattern=paste0(date_list[ii],'_y86x88.nc'))

  # jj = 1
  for (jj in 1:length(DMn_list)){
    path_file=DMn_list[jj]
    get_ptspattern(path_file=path_file)
  }
}

