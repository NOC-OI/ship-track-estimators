# Script to get historical ships data
# Last modification by Joao Morado (15/05/2023)
# This script is meant to be run on the NOC Southampton servers

dir.in<-"/noc/mpoc/surface/eck/TRACK_CHECK/"
source("/noc/mpoc/surface/eck/TRACK_CHECK/plot4id.R")

#config.file<-"/noc/mpoc/surface/config.yml"
#config<-config::get(file = config.file)

flist<-list.files(paste0(dir.in,"DWD/proc/steamer"),patt="1920",full.names=T)
flist<-list.files(paste0(dir.in,"DWD/proc/sailing_vessel"),patt="1878",full.names=T)

sp<-lapply(flist,readRDS)
d<-do.call(rbind,sp)
d<-d[!is.na(d$date),]

sp<-split(d,d$primary.id)
sp<-sp[sapply(sp,nrow)>=30]

# Save as csv
lapply(sp, function(x) write.table( data.frame(x), 'sp_historical_ship_data.csv'  , append= T, sep=',' ))
lapply(d, function(x) write.table( data.frame(x), 'd_historical_ship_data.csv'  , append= T, sep=',' ))

#lapply(sp,plot4id)
