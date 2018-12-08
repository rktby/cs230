library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(lubridate)
library(broom)
library(zoo)
library(chron)
library(scales)
library(forecast)
library(proxy)
library(data.table)
library(magrittr)
library(smooth)
library(Mcomp)

########### Input Arguments ########### 
inDir = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Data'
opDSN = 'cr2c_opdata_TMP_PRESSURE_TEMP_WATER_COND_GAS_PH_DPI_LEVEL.csv'
outDir = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Data'
ltypes = c('PH','COD','TSS_VSS','ALKALINITY','TKN','AMMONIA','VFA','GASCOMP','SULFATE','BOD')
########### Input Arguments ########### 

# Read in HMI data
setwd(inDir)
hmiDat = 
  read.csv(opDSN, header = T) %>% 
  mutate(
    Date = as.Date(Time),
    Time = as.POSIXct(Time)
  )

# Read in COD data
labdata = 
  lapply(ltypes, function(ltype){
    ldata = 
      read.csv(paste0(ltype,'.csv'), header = T) %>%
      mutate(Date = as.Date(Date_Time))
    if ('Stage' %in% colnames(ldata) & 'Type' %in% colnames(ldata)){
      ldata_wide =
        group_by(ldata, Date, Stage, Type) %>%
        summarize(Value_ = mean(Value, na.rm = T)) %>%
        filter(!is.na(Value_)) %>%
        select(Date, Stage, Type, Value_) %>%
        dcast(Date ~ Stage + Type, value.var = 'Value_') 
    } else if ('Stage' %in% colnames(ldata)){
      ldata_wide =
        group_by(ldata, Date, Stage) %>%
        summarize(Value_ = mean(Value, na.rm = T)) %>%
        filter(!is.na(Value_)) %>%
        select(Date, Stage, Value_) %>%
        dcast(Date ~ Stage, value.var = 'Value_')
    } else {
      ldata_wide =
        group_by(ldata, Date, Type) %>%
        summarize(Value_ = mean(Value, na.rm = T)) %>%
        filter(!is.na(Value_)) %>%
        select(Date, Type, Value_) %>%
        dcast(Date ~ Type, value.var = 'Value_')         
    }
    value_colnames = colnames(ldata_wide)[2:length(colnames(ldata_wide))]
    value_colnames = gsub(' ','_', value_colnames)
    value_colnames = gsub(':','', value_colnames)
    names(ldata_wide) = c('Date',paste(ltype, value_colnames, sep = '_'))
    ldata_wide
  })

labdata = Reduce(function(...) merge(..., by = 'Date', all = T), labdata)

# Merge and sort by time (day,hour)
merged_data = 
  merge(hmiDat, labdata, by = 'Date', all = TRUE) %>% 
  arrange(Time) %>%
  mutate(
    hour = as.factor(hour(Time)),
    weekday = as.factor(weekdays(Date)),
    week = as.factor(week(Date)),
    month = as.factor(month(Date))
  ) 

# Output
setwd(outDir)
write.csv(merged_data, 'merged_data.csv', row.names = F)
