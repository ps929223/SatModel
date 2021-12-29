library(ncdf4) # package for netcdf manipulation
library(raster) # package for raster manipulation
library(rgdal) # package for geospatial analysis
library(ggplot2) # package for plotting
library(sp)
library(ape)
library(spdep)


wd=paste0('E:/CHL_DM')
setwd(wd)

days='3'
yyyy='2018'
mm='03'
dd='21'

p_DM1=paste0('E:/CHL_DM/',yyyy,'/',mm,'/GOCI1_DM','_CHL_',yyyy,'-',mm,'-',dd,'_res100.nc')
p_DM3=paste0('E:/CHL_DM3/',yyyy,'/',mm,'/GOCI1_DM3','_CHL_',yyyy,'-',mm,'-',dd,'_res100.nc')
p_DM5=paste0('E:/CHL_DM5/',yyyy,'/',mm,'/GOCI1_DM5','_CHL_',yyyy,'-',mm,'-',dd,'_res100.nc')

file_list=c(p_DM1, p_DM3, p_DM5)
# file_list=list.files(pattern = "2018-03-04*res100.nc", recursive = TRUE)

moranI.estimate <- list() # Moran's I
moranI.p <- list() # Moran's I
moranI.statistic <-list() # Moran's I

gearyC.estimate <- list() # Geary's C
gearyC.p<- list() # Geary's C
gearyC.statistic<- list() # Geary's C

getisG <- list() # getis-ord's g

filenames <- list()
# rm(moran.statistic)

for (ii in 1:length(file_list)){
  # ii=2
  print(file_list[ii])
  nc_data <- nc_open(file_list[ii])
  # nc_data <- nc_open('E:/CHL_DM3/2018/03/GOCI1_DM3_CHL_2018-03-21_res100.nc')
  chl.array <- ncvar_get(nc_data, 'chl')
  chl.flat <- as.vector(chl.array)
  lon.array <- ncvar_get(nc_data, 'lon')
  lon.flat <- as.vector(lon.array)
  lat.array <- ncvar_get(nc_data, 'lat')
  lat.flat <- as.vector(lat.array)
  lm.array <- ncvar_get(nc_data, 'lm')
  lm.flat <- as.vector(lm.array)
  head(chl.flat)
  # 
  DF=data.frame(lon.flat, lat.flat, chl.flat)
  
  ## Land masking
  cond=lm.flat==2 # 2: land
  sum(cond)
  DF=DF[!cond,]
  
  ## cond cloud
  cond=is.na(DF$chl.flat) # cloud 0, chl 1
  head(cond)
  DF$chl.flat[cond]=0 # cloud
  DF$chl.flat[!cond]=1 # chl

  
  ## 거리계산
   DF.dists <- as.matrix(dist(cbind(DF$lon.flat,DF$lat.flat)))
   DF.dists.inv <- 1/DF.dists
   diag(DF.dists.inv) <-0
   Moran.I(DF$chl.flat, DF.dists.inv, na.rm=TRUE)
  ## 거리에 따른 삼각망 이용
  # coord=cbind(DF$lon.flat,DF$lat.flat)
  
  # dd3nb=dnearneigh(coord,0,3)
  # help("dnearneigh")
  ## Moran I
  m=moran.test(DF$chl.flat, nb2listw(dd3nb, style='W'))
  moranI.estimate<-append(moranI.estimate, m$estimate)
  moranI.p<-append(moranI.p, m$p)
  moranI.statistic <- append(moranI.statistic, m$statistic)
  ## Geary C
  g=geary.test(DF$chl.flat, nb2listw(dd3nb, style='W'))
  gearyC.estimate<-append(gearyC.estimate, g$estimate)
  gearyC.p<-append(gearyC.p, g$p)
  gearyC.statistic <- append(gearyC.statistic, g$statistic)
  ## getis-ord's g
  go=globalG.test(DF$chl.flat, nb2listw(dd3nb, style='W'))
  getisG<-append(getisG, go)
  ## file_name
  filenames <- append(filenames, file_list[ii])
  }
