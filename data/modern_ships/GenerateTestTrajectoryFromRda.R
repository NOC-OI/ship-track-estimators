# Script to download data of modern ships
# Original version by Liz Kent (12/05/2023)
# Updated by Joao Morado (15/05/2023)
# This script is meant to be run on the NOC Southampton servers


flist<-list.files(".",patt="hourly",full.names=T)

sp<-lapply(flist,readRDS)

# Save to csv file
output_file<-"modern_ship_data.csv"
write.csv(sp,file=output_file)
#lapply(sp, function(x) write.table( data.frame(x), 'modern_ship_data.csv'  , append= T, sep=',' ))
