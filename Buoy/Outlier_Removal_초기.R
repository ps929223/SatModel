
library("KernSmooth")

path_wd='E:/20_Product/Buoy/Dokdo'
setwd(path_wd)


rdata=read.csv('Recent_2020-10-16_2021-11-18_thred.csv')
head(rdata)
dim(rdata)


WT <- as.vector(rdata$w_ZP) # º¤ÅÍº¯È¯
ndata <- dim(rdata)[1] # n-sample
didx <- 1:ndata # indices

plot(1:ndata, WT)
H = 0
L = 8
abline(h=c(L, H), col="red", lwd=2)


oidx <- which(WT > H | WT < L)
WT[oidx] <- rep(NA, length(oidx))
plot(WT)

smt1 <- lowess(didx, WT, f = 10000, iter=3)
smt2 <- ksmooth(didx, WT, bandwidth=5000.0)

midx <- which(is.na(WT)==TRUE)
smt3 <- locpoly(didx[-midx], WT[-midx], gridsize=ndata, bandwidth=2500)


plot(WT)
lines(smt3$x, smt3$y, col="red", lwd=3)

plot(WT - smt3$y)
sigma <- sd(WT-smt3$y, na.rm=T)
abline(h=c(0,2*sigma, -2*sigma), col="red", lwd=3)


rr <- WT - smt3$y
oidx <- which(rr > 2*sigma | rr < -2*sigma)
WT[oidx] <- rep(NA, length(oidx))

plot(WT)


midx <- which(is.na(WT)==TRUE)
smt3 <- locpoly(didx[-midx], WT[-midx], gridsize=ndata, bandwidth=500)
lines(smt3$x, smt3$y, col="red", lwd=3)

rr <- WT - smt3$y

plot(rr)


'''
Á¶È«¿¬ ±³¼ö´Ô ÄÚµå
'''


library("KernSmooth")

rdata <- read.csv("Recent_2020-10-16_2021-11-17_37S_T.csv")

head(rdata)
dim(rdata)

WT <- as.vector(rdata$X37S_T)
ndata <- dim(rdata)[1]
didx <- 1:ndata

plot(1:ndata, WT)
abline(h=c(5, 35), col="red", lwd=2)


oidx <- which(WT > 35 | WT < 5)
WT[oidx] <- rep(NA, length(oidx))
plot(WT)

smt1 <- lowess(didx, WT, f = 10000, iter=3)
smt2 <- ksmooth(didx, WT, bandwidth=5000.0)

midx <- which(is.na(WT)==TRUE)
smt3 <- locpoly(didx[-midx], WT[-midx], gridsize=ndata, bandwidth=2500)


plot(WT)
lines(smt3$x, smt3$y, col="red", lwd=3)

plot(WT - smt3$y)
sigma <- sd(WT-smt3$y, na.rm=T)
abline(h=c(0,2*sigma, -2*sigma), col="red", lwd=3)


rr <- WT - smt3$y
oidx <- which(rr > 2*sigma | rr < -2*sigma)
WT[oidx] <- rep(NA, length(oidx))

plot(WT)


midx <- which(is.na(WT)==TRUE)
smt3 <- locpoly(didx[-midx], WT[-midx], gridsize=ndata, bandwidth=500)
lines(smt3$x, smt3$y, col="red", lwd=3)

rr <- WT - smt3$y

plot(rr)