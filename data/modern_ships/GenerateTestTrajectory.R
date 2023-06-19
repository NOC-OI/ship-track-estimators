# Script to download data of modern ships
# Original version by Liz Kent (12/05/2023)
# Updated by by Joao Morado (15/05/2023)
# This script is meant to be run on the NOC Southampton servers

flist<-list.files("/scratch/general/ricorne",patt="2021",full.names=T)

sp<-lapply(flist,readRDS)
sp<-lapply(sp,function(X) X<-X[X$pt==5 & X$dups<=2,c("yr","mo","dy","hr","dck","id","lat","lon","w","d")])
df<-do.call(rbind,sp)

#df<-icoads.utils::add_date2(df)
#df<-df[!(df$id %in% c("MASKSTID","SHIP")),]

hourly.sub<-subset(df, id %in% c("KAOU","WDG7520","WDA7827","AMOUK05","WCE5063","WGAE","SJA4RSK"))

# Save to csv file
lapply(sp, function(x) write.table( data.frame(x), 'modern_ship_data.csv'  , append= T, sep=',' ))
