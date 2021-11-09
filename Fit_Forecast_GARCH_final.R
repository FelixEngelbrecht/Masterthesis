###################################################################################

############################# Master's Thesis #####################################
#Does sentiment from relevant news improve volatility forecast from GARCH models? -
#An application of natural language #processing

###################################################################################

library(xts)
library(zoo)
library(tidyverse)
library(ggthemes)
library(forecast)
library(tseries)
library(gridExtra)
library(rugarch)
library(ggplot)
library(lmtest)
library(rugarch)

################################################################################
#read in data
################################################################################

df = read.csv("Sent_SP500_1.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
close_rets = diff(log(df$close))
df2 = read.csv("merged_pol_Sent.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
df3 = read.csv("Sent_mva.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
df4 = read.csv("mva_std_Sent.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
#since the first four rows are nas´s, they replaced by the first four rows of the corresponding average
df3$pos_Sent_mva[1:4] = df3$pos_Sent[1:4]
df3$neg_Sent_mva[1:4] = df3$neg_Sent[1:4]
################################################################################
#check for arma model
################################################################################

fit1 = auto.arima(close_rets, trace=TRUE, test="kpss",  ic="aic")

AIC=matrix(NA , nrow = 3, ncol = 3)

################################################################################
#check which lag order for the variance model, fits the data best
#uo to GARCH(3,3)
################################################################################

for(i in 1:3){
  for (j in 1:3){
    garchspec_test = ugarchspec(mean.model = list(armaOrder = c(0,2)),
                                variance.model = list(model = "sGARCH", garchOrder = c(i,j)), 
                                distribution.model = "norm")
    garchfit_test = ugarchfit(data = close_rets, spec = garchspec_test)
    AIC[i,j] = infocriteria(garchfit_test)[1]
    
  }
}
AIC_best = min(AIC)
AIC_best

#result GARCH(2,1) fits the data the best, but to reduce computation time 
#and increase comparability with the literature, GARCH(1,1) is used
################################################################################
#fit models without external regressor (ARMA order (0,2))

################################################################################
#Number in second position:
#1 = sGARCH
#2 = eGARCH
#3 = GJR-GARCH
################################################################################
#Number in third position
#1 = norm
#2 = std
#3 = ged
#4 = snorm
#5 = sstd
#6 = sged
################################################################################

#sGARCH
#norm
garchspec0.1.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                        variance.model = list(model = "sGARCH"), 
                        distribution.model = "norm")
garchfit0.1.1 <- ugarchfit(data = close_rets, spec = garchspec0.1.1)
garchfit0.1.1

#std
garchspec0.1.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH"), 
                             distribution.model = "std")

garchfit0.1.2 <- ugarchfit(data = close_rets, spec = garchspec0.1.2)
garchfit0.1.2

#ged
garchspec0.1.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH"), 
                             distribution.model = "ged")

garchfit0.1.3 <- ugarchfit(data = close_rets, spec = garchspec0.1.3)
garchfit0.1.3

#snorm
garchspec0.1.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH"), 
                             distribution.model = "snorm")

garchfit0.1.4 <- ugarchfit(data = close_rets, spec = garchspec0.1.4)
garchfit0.1.4

#sstd
garchspec0.1.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH"), 
                             distribution.model = "sstd")

garchfit0.1.5 <- ugarchfit(data = close_rets, spec = garchspec0.1.5)
garchfit0.1.5

#sged
garchspec0.1.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH"), 
                             distribution.model = "sged")

garchfit0.1.6 <- ugarchfit(data = close_rets, spec = garchspec0.1.6)
garchfit0.1.6

################################################################################
#eGARCH
################################################################################
#norm
garchspec0.2.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH"), 
                             distribution.model = "norm")

garchfit0.2.1 <- ugarchfit(data = close_rets, spec = garchspec0.2.1)
garchfit0.2.1

#std
garchspec0.2.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH"), 
                             distribution.model = "std")

garchfit0.2.2 <- ugarchfit(data = close_rets, spec = garchspec0.2.2)
garchfit0.2.2

#ged
garchspec0.2.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH"), 
                             distribution.model = "ged")

garchfit0.2.3 <- ugarchfit(data = close_rets, spec = garchspec0.2.3)
garchfit0.2.3

#snorm
garchspec0.2.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH"), 
                             distribution.model = "snorm")

garchfit0.2.4 <- ugarchfit(data = close_rets, spec = garchspec0.2.4)
garchfit0.2.4

#sstd
garchspec0.2.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH"), 
                             distribution.model = "sstd")

garchfit0.2.5 <- ugarchfit(data = close_rets, spec = garchspec0.2.5)
garchfit0.2.5

#sged
garchspec0.2.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH"), 
                             distribution.model = "sged")

garchfit0.2.6 <- ugarchfit(data = close_rets, spec = garchspec0.2.6)
garchfit0.2.6

################################################################################
#gjrGARCH
################################################################################
#norm
garchspec0.3.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH"), 
                             distribution.model = "norm")

garchfit0.3.1 <- ugarchfit(data = close_rets, spec = garchspec0.3.1)
garchfit0.3.1

#std
garchspec0.3.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH"), 
                             distribution.model = "std")

garchfit0.3.2 <- ugarchfit(data = close_rets, spec = garchspec0.3.2)
garchfit0.3.2

#ged
garchspec0.3.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH"), 
                             distribution.model = "ged")

garchfit0.3.3 <- ugarchfit(data = close_rets, spec = garchspec0.3.3)
garchfit0.3.3

#snorm
garchspec0.3.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH"), 
                             distribution.model = "snorm")

garchfit0.3.4 <- ugarchfit(data = close_rets, spec = garchspec0.3.4)
garchfit0.3.4

#sstd
garchspec0.3.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH"), 
                             distribution.model = "sstd")

garchfit0.3.5 <- ugarchfit(data = close_rets, spec = garchspec0.3.5)
garchfit0.3.5

#sged
garchspec0.3.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH"), 
                             distribution.model = "sged")

garchfit0.3.6 <- ugarchfit(data = close_rets, spec = garchspec0.3.6)
garchfit0.3.6


################################################################################
#add external regressor 

################################################################################
#(1 in first position = average daily sentiment; 
#2=split into positive and negative;
#3=mva;
#4=std)
################################################################################


#sGARCH
#norm
garchspec1.1.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "norm")

garchfit1.1.1 <- ugarchfit(data = close_rets, spec = garchspec1.1.1)
garchfit1.1.1

#std
garchspec1.1.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH",
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "std")

garchfit1.1.2 <- ugarchfit(data = close_rets, spec = garchspec1.1.2)
garchfit1.1.2

#ged
garchspec1.1.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "ged")

garchfit1.1.3 <- ugarchfit(data = close_rets, spec = garchspec1.1.3)
garchfit1.1.3

#snorm
garchspec1.1.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "snorm")

garchfit1.1.4 <- ugarchfit(data = close_rets, spec = garchspec1.1.4)
garchfit1.1.4

#sstd
garchspec1.1.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "sstd")

garchfit1.1.5 <- ugarchfit(data = close_rets, spec = garchspec1.1.5)
garchfit1.1.5

#sged
garchspec1.1.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "sged")

garchfit1.1.6 <- ugarchfit(data = close_rets, spec = garchspec1.1.6)
garchfit1.1.6

################################################################################
#eGARCH
################################################################################
#norm
garchspec1.2.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "norm")

garchfit1.2.1 <- ugarchfit(data = close_rets, spec = garchspec1.2.1)
garchfit1.2.1

#std
garchspec1.2.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "std")

garchfit1.2.2 <- ugarchfit(data = close_rets, spec = garchspec1.2.2)
garchfit1.2.2

#ged
garchspec1.2.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "ged")

garchfit1.2.3 <- ugarchfit(data = close_rets, spec = garchspec1.2.3)
garchfit1.2.3

#snorm
garchspec1.2.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "snorm")

garchfit1.2.4 <- ugarchfit(data = close_rets, spec = garchspec1.2.4)
garchfit1.2.4

#sstd
garchspec1.2.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "sstd")

garchfit1.2.5 <- ugarchfit(data = close_rets, spec = garchspec1.2.5)
garchfit1.2.5

#sged
garchspec1.2.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "sged")

garchfit1.2.6 <- ugarchfit(data = close_rets, spec = garchspec1.2.6)
garchfit1.2.6

################################################################################
#gjrGARCH
################################################################################
#norm
garchspec1.3.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "norm")

garchfit1.3.1 <- ugarchfit(data = close_rets, spec = garchspec1.3.1)
garchfit1.3.1

#std
garchspec1.3.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "std")

garchfit1.3.2 <- ugarchfit(data = close_rets, spec = garchspec1.3.2)
garchfit1.3.2

#ged
garchspec1.3.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "ged")

garchfit1.3.3 <- ugarchfit(data = close_rets, spec = garchspec1.3.3)
garchfit1.3.3

#snorm
garchspec1.3.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "snorm")

garchfit1.3.4 <- ugarchfit(data = close_rets, spec = garchspec1.3.4)
garchfit1.3.4

#sstd
garchspec1.3.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "sstd")

garchfit1.3.5 <- ugarchfit(data = close_rets, spec = garchspec1.3.5)
garchfit1.3.5

#sged
garchspec1.3.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH",
                             external.regressors = matrix(cbind(df$sentiment_score), ncol=1)),
                             distribution.model = "sged")

garchfit1.3.6 <- ugarchfit(data = close_rets, spec = garchspec1.3.6)
garchfit1.3.6

################################################################################
#positive and negative sentiment seperated
################################################################################
#sGARCH
#norm
garchspec2.1.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "norm")

garchfit2.1.1 <- ugarchfit(data = close_rets, spec = garchspec2.1.1)
garchfit2.1.1

#std
garchspec2.1.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH",
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "std")

garchfit2.1.2 <- ugarchfit(data = close_rets, spec = garchspec2.1.2)
garchfit2.1.2

#ged
garchspec2.1.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "ged")

garchfit2.1.3 <- ugarchfit(data = close_rets, spec = garchspec2.1.3)
garchfit2.1.3

#snorm
garchspec2.1.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "snorm")

garchfit2.1.4 <- ugarchfit(data = close_rets, spec = garchspec2.1.4)
garchfit2.1.4

#sstd
garchspec2.1.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "sstd")

garchfit2.1.5 <- ugarchfit(data = close_rets, spec = garchspec2.1.5)
garchfit1.1.5

#sged
garchspec2.1.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "sged")

garchfit2.1.6 <- ugarchfit(data = close_rets, spec = garchspec2.1.6)
garchfit2.1.6

################################################################################
#eGARCH
################################################################################
#norm
garchspec2.2.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "norm")

garchfit2.2.1 <- ugarchfit(data = close_rets, spec = garchspec2.2.1)
garchfit2.2.1

#std
garchspec2.2.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "std")

garchfit2.2.2 <- ugarchfit(data = close_rets, spec = garchspec2.2.2)
garchfit2.2.2

#ged
garchspec2.2.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "ged")

garchfit2.2.3 <- ugarchfit(data = close_rets, spec = garchspec2.2.3)
garchfit2.2.3

#snorm
garchspec2.2.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "snorm")

garchfit2.2.4 <- ugarchfit(data = close_rets, spec = garchspec2.2.4)
garchfit2.2.4

#sstd
garchspec2.2.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "sstd")

garchfit2.2.5 <- ugarchfit(data = close_rets, spec = garchspec2.2.5)
garchfit2.2.5

#sged
garchspec2.2.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "sged")

garchfit2.2.6 <- ugarchfit(data = close_rets, spec = garchspec2.2.6)
garchfit2.2.6

################################################################################
#gjrGARCH
################################################################################
#norm
garchspec2.3.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "norm")

garchfit2.3.1 <- ugarchfit(data = close_rets, spec = garchspec2.3.1)
garchfit2.3.1

#std
garchspec2.3.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "std")

garchfit2.3.2 <- ugarchfit(data = close_rets, spec = garchspec2.3.2)
garchfit2.3.2

#ged
garchspec2.3.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "ged")

garchfit2.3.3 <- ugarchfit(data = close_rets, spec = garchspec2.3.3)
garchfit2.3.3

#snorm
garchspec2.3.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "snorm")

garchfit2.3.4 <- ugarchfit(data = close_rets, spec = garchspec2.3.4)
garchfit2.3.4

#sstd
garchspec2.3.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "sstd")

garchfit2.3.5 <- ugarchfit(data = close_rets, spec = garchspec2.3.5)
garchfit2.3.5

#sged
garchspec2.3.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH",
                                                   external.regressors = matrix(cbind(df2$pos_Sent, df2$neg_Sent), ncol=2)),
                             distribution.model = "sged")

garchfit2.3.6 <- ugarchfit(data = close_rets, spec = garchspec2.3.6)
garchfit2.3.6


################################################################################
#positive and negative sentiment seperated (moving average)
################################################################################
#sGARCH
#norm
garchspec3.1.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "norm")

garchfit3.1.1 <- ugarchfit(data = close_rets, spec = garchspec3.1.1)
garchfit3.1.1

#std
garchspec3.1.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH",
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "std")

garchfit3.1.2 <- ugarchfit(data = close_rets, spec = garchspec3.1.2)
garchfit3.1.2

#ged
garchspec3.1.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "ged")

garchfit3.1.3 <- ugarchfit(data = close_rets, spec = garchspec3.1.3)
garchfit3.1.3

#snorm
garchspec3.1.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "snorm")

garchfit3.1.4 <- ugarchfit(data = close_rets, spec = garchspec3.1.4)
garchfit3.1.4

#sstd
garchspec3.1.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "sstd")

garchfit3.1.5 <- ugarchfit(data = close_rets, spec = garchspec3.1.5)
garchfit1.1.5

#sged
garchspec3.1.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "sged")

garchfit3.1.6 <- ugarchfit(data = close_rets, spec = garchspec3.1.6)
garchfit3.1.6

################################################################################
#eGARCH
################################################################################
#norm
garchspec3.2.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "norm")

garchfit3.2.1 <- ugarchfit(data = close_rets, spec = garchspec3.2.1)
garchfit3.2.1

#std
garchspec3.2.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "std")

garchfit3.2.2 <- ugarchfit(data = close_rets, spec = garchspec3.2.2)
garchfit3.2.2

#ged
garchspec3.2.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "ged")

garchfit3.2.3 <- ugarchfit(data = close_rets, spec = garchspec3.2.3)
garchfit3.2.3

#snorm
garchspec3.2.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "snorm")

garchfit3.2.4 <- ugarchfit(data = close_rets, spec = garchspec3.2.4)
garchfit3.2.4

#sstd
garchspec3.2.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "sstd")

garchfit3.2.5 <- ugarchfit(data = close_rets, spec = garchspec3.2.5)
garchfit3.2.5

#sged
garchspec3.2.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "sged")

garchfit3.2.6 <- ugarchfit(data = close_rets, spec = garchspec3.2.6)
garchfit3.2.6

################################################################################
#gjrGARCH
################################################################################
#norm
garchspec3.3.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "norm")

garchfit3.3.1 <- ugarchfit(data = close_rets, spec = garchspec3.3.1)
garchfit3.3.1

#std
garchspec3.3.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "std")

garchfit3.3.2 <- ugarchfit(data = close_rets, spec = garchspec3.3.2)
garchfit3.3.2

#ged
garchspec3.3.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "ged")

garchfit3.3.3 <- ugarchfit(data = close_rets, spec = garchspec3.3.3)
garchfit3.3.3

#snorm
garchspec3.3.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "snorm")

garchfit3.3.4 <- ugarchfit(data = close_rets, spec = garchspec3.3.4)
garchfit3.3.4

#sstd
garchspec3.3.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "sstd")

garchfit3.3.5 <- ugarchfit(data = close_rets, spec = garchspec3.3.5)
garchfit3.3.5

#sged
garchspec3.3.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH",
                                                   external.regressors = matrix(cbind(df3$pos_Sent_mva, df3$neg_Sent_mva), ncol=2)),
                             distribution.model = "sged")

garchfit3.3.6 <- ugarchfit(data = close_rets, spec = garchspec3.3.6)
garchfit3.3.6

################################################################################
#Only sentiment more than one std from mean
################################################################################
#sGARCH
#norm
garchspec4.1.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "norm")

garchfit4.1.1 <- ugarchfit(data = close_rets, spec = garchspec4.1.1)
garchfit4.1.1

#std
garchspec4.1.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH",
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "std")

garchfit4.1.2 <- ugarchfit(data = close_rets, spec = garchspec4.1.2)
garchfit4.1.2

#ged
garchspec4.1.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "ged")

garchfit4.1.3 <- ugarchfit(data = close_rets, spec = garchspec4.1.3)
garchfit4.1.3

#snorm
garchspec4.1.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "snorm")

garchfit4.1.4 <- ugarchfit(data = close_rets, spec = garchspec4.1.4)
garchfit4.1.4

#sstd
garchspec4.1.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "sstd")

garchfit4.1.5 <- ugarchfit(data = close_rets, spec = garchspec4.1.5)
garchfit1.1.5

#sged
garchspec4.1.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "sGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "sged")

garchfit4.1.6 <- ugarchfit(data = close_rets, spec = garchspec4.1.6)
garchfit4.1.6

################################################################################
#eGARCH
################################################################################
#norm
garchspec4.2.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "norm")

garchfit4.2.1 <- ugarchfit(data = close_rets, spec = garchspec4.2.1)
garchfit4.2.1

#std
garchspec4.2.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "std")

garchfit4.2.2 <- ugarchfit(data = close_rets, spec = garchspec4.2.2)
garchfit4.2.2

#ged
garchspec4.2.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "ged")

garchfit4.2.3 <- ugarchfit(data = close_rets, spec = garchspec4.2.3)
garchfit4.2.3

#snorm
garchspec4.2.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "snorm")

garchfit4.2.4 <- ugarchfit(data = close_rets, spec = garchspec4.2.4)
garchfit4.2.4

#sstd
garchspec4.2.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "sstd")

garchfit4.2.5 <- ugarchfit(data = close_rets, spec = garchspec4.2.5)
garchfit4.2.5

#sged
garchspec4.2.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "eGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "sged")

garchfit4.2.6 <- ugarchfit(data = close_rets, spec = garchspec4.2.6)
garchfit4.2.6

################################################################################
#gjrGARCH
################################################################################
#norm
garchspec4.3.1 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "norm")

garchfit4.3.1 <- ugarchfit(data = close_rets, spec = garchspec4.3.1)
garchfit4.3.1

#std
garchspec4.3.2 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "std")

garchfit4.3.2 <- ugarchfit(data = close_rets, spec = garchspec4.3.2)
garchfit4.3.2

#ged
garchspec4.3.3 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "ged")

garchfit4.3.3 <- ugarchfit(data = close_rets, spec = garchspec4.3.3)
garchfit4.3.3

#snorm
garchspec4.3.4 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "snorm")

garchfit4.3.4 <- ugarchfit(data = close_rets, spec = garchspec4.3.4)
garchfit4.3.4

#sstd
garchspec4.3.5 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH", 
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "sstd")

garchfit4.3.5 <- ugarchfit(data = close_rets, spec = garchspec4.3.5)
garchfit4.3.5

#sged
garchspec4.3.6 <- ugarchspec(mean.model = list(armaOrder = c(0,2)),
                             variance.model = list(model = "gjrGARCH",
                                                   external.regressors = matrix(cbind(df4$pos_Sent_x, df4$neg_Sent_x), ncol=2)),
                             distribution.model = "sged")

garchfit4.3.6 <- ugarchfit(data = close_rets, spec = garchspec4.3.6)
garchfit4.3.6

################################################################################
#One step ahead forecast using expanding windows (start: 400; refit every 50)
################################################################################
# no external regressor
################################################################################
#sGARCH
#norm
model.fit_exp0.1.1 = ugarchroll(garchspec0.1.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                           refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                           keep.coef = TRUE)
report(model.fit_exp0.1.1, type = "fpm")

#std
model.fit_exp0.1.2 = ugarchroll(garchspec0.1.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.2, type = "fpm")

#ged
model.fit_exp0.1.3 = ugarchroll(garchspec0.1.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.3, type = "fpm")

#snorm
model.fit_exp0.1.4 = ugarchroll(garchspec0.1.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.4, type = "fpm")

#sstd
model.fit_exp0.1.5 = ugarchroll(garchspec0.1.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.5, type = "fpm")

#sged
model.fit_exp0.1.6 = ugarchroll(garchspec0.1.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.6, type = "fpm")

#eGARCH
#norm
model.fit_exp0.2.1 = ugarchroll(garchspec0.2.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.1, type = "fpm")

#std
model.fit_exp0.2.2 = ugarchroll(garchspec0.2.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.2, type = "fpm")

#ged
model.fit_exp0.2.3 = ugarchroll(garchspec0.2.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.3, type = "fpm")

#snorm
model.fit_exp0.2.4 = ugarchroll(garchspec0.2.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.4, type = "fpm")

#sstd
model.fit_exp0.2.5 = ugarchroll(garchspec0.2.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.5, type = "fpm")

#sged
model.fit_exp0.2.6 = ugarchroll(garchspec0.2.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.6, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp0.3.1 = ugarchroll(garchspec0.3.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.1, type = "fpm")

#std
model.fit_exp0.3.2 = ugarchroll(garchspec0.3.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.2, type = "fpm")

#ged
model.fit_exp0.3.3 = ugarchroll(garchspec0.3.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.3, type = "fpm")

#snorm
model.fit_exp0.3.4 = ugarchroll(garchspec0.3.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.4, type = "fpm")

#sstd
model.fit_exp0.3.5 = ugarchroll(garchspec0.3.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.5, type = "fpm")

#sged
model.fit_exp0.3.6 = ugarchroll(garchspec0.3.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.6, type = "fpm")

################################################################################
# average daily sentiment 
################################################################################
#sGARCH
#norm
model.fit_exp1.1.1 = ugarchroll(garchspec1.1.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.1, type = "fpm")

#std
model.fit_exp1.1.2 = ugarchroll(garchspec1.1.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.2, type = "fpm")

#ged
model.fit_exp1.1.3 = ugarchroll(garchspec1.1.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.3, type = "fpm")

#snorm
model.fit_exp1.1.4 = ugarchroll(garchspec1.1.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.4, type = "fpm")

#sstd
model.fit_exp1.1.5 = ugarchroll(garchspec1.1.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.5, type = "fpm")

#sged
model.fit_exp1.1.6 = ugarchroll(garchspec1.1.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.6, type = "fpm")

#eGARCH
#norm
model.fit_exp1.2.1 = ugarchroll(garchspec1.2.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.1, type = "fpm")

#std
model.fit_exp1.2.2 = ugarchroll(garchspec1.2.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.2, type = "fpm")

#ged
model.fit_exp1.2.3 = ugarchroll(garchspec1.2.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.3, type = "fpm")

#snorm
model.fit_exp1.2.4 = ugarchroll(garchspec1.2.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.4, type = "fpm")

#sstd
model.fit_exp1.2.5 = ugarchroll(garchspec1.2.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.5, type = "fpm")

#sged
model.fit_exp1.2.6 = ugarchroll(garchspec1.2.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.6, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp1.3.1 = ugarchroll(garchspec1.3.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.1, type = "fpm")

#std
model.fit_exp1.3.2 = ugarchroll(garchspec1.3.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.2, type = "fpm")

#ged
model.fit_exp1.3.3 = ugarchroll(garchspec1.3.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.3, type = "fpm")

#snorm
model.fit_exp1.3.4 = ugarchroll(garchspec1.3.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.4, type = "fpm")

#sstd
model.fit_exp1.3.5 = ugarchroll(garchspec1.3.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.5, type = "fpm")

#sged
model.fit_exp1.3.6 = ugarchroll(garchspec1.3.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.6, type = "fpm")

################################################################################
# average daily sentiment split into positive and negative
################################################################################
#sGARCH
#norm
model.fit_exp2.1.1 = ugarchroll(garchspec2.1.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.1, type = "fpm")

#std
model.fit_exp2.1.2 = ugarchroll(garchspec2.1.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.2, type = "fpm")

#ged
model.fit_exp2.1.3 = ugarchroll(garchspec2.1.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.3, type = "fpm")

#snorm
model.fit_exp2.1.4 = ugarchroll(garchspec2.1.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.4, type = "fpm")

#sstd
model.fit_exp2.1.5 = ugarchroll(garchspec2.1.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.5, type = "fpm")

#sged
model.fit_exp2.1.6 = ugarchroll(garchspec2.1.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.6, type = "fpm")

#eGARCH
#norm
model.fit_exp2.2.1 = ugarchroll(garchspec2.2.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.1, type = "fpm")

#std
model.fit_exp2.2.2 = ugarchroll(garchspec2.2.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.2, type = "fpm")

#ged
model.fit_exp2.2.3 = ugarchroll(garchspec2.2.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.3, type = "fpm")

#snorm
model.fit_exp2.2.4 = ugarchroll(garchspec2.2.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.4, type = "fpm")

#sstd
model.fit_exp2.2.5 = ugarchroll(garchspec2.2.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.5, type = "fpm")

#sged
model.fit_exp2.2.6 = ugarchroll(garchspec2.2.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.6, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp2.3.1 = ugarchroll(garchspec2.3.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.1, type = "fpm")

#std
model.fit_exp2.3.2 = ugarchroll(garchspec2.3.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.2, type = "fpm")

#ged
model.fit_exp2.3.3 = ugarchroll(garchspec2.3.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.3, type = "fpm")

#snorm
model.fit_exp2.3.4 = ugarchroll(garchspec2.3.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.4, type = "fpm")

#sstd
model.fit_exp2.3.5 = ugarchroll(garchspec2.3.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.5, type = "fpm")

#sged
model.fit_exp2.3.6 = ugarchroll(garchspec2.3.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.6, type = "fpm")

################################################################################
# average daily sentiment split into positive and negative (MVA)
################################################################################
#sGARCH
#norm
model.fit_exp3.1.1 = ugarchroll(garchspec3.1.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.1, type = "fpm")

#std
model.fit_exp3.1.2 = ugarchroll(garchspec3.1.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.2, type = "fpm")

#ged
model.fit_exp3.1.3 = ugarchroll(garchspec3.1.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.3, type = "fpm")

#snorm
model.fit_exp3.1.4 = ugarchroll(garchspec3.1.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.4, type = "fpm")

#sstd
model.fit_exp3.1.5 = ugarchroll(garchspec3.1.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.5, type = "fpm")

#sged
model.fit_exp3.1.6 = ugarchroll(garchspec3.1.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.6, type = "fpm")

#eGARCH
#norm
model.fit_exp3.2.1 = ugarchroll(garchspec3.2.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.1, type = "fpm")

#std
model.fit_exp3.2.2 = ugarchroll(garchspec3.2.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.2, type = "fpm")

#ged
model.fit_exp3.2.3 = ugarchroll(garchspec3.2.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.3, type = "fpm")

#snorm
model.fit_exp3.2.4 = ugarchroll(garchspec3.2.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.4, type = "fpm")

#sstd
model.fit_exp3.2.5 = ugarchroll(garchspec3.2.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.5, type = "fpm")

#sged
model.fit_exp3.2.6 = ugarchroll(garchspec3.2.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.6, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp3.3.1 = ugarchroll(garchspec3.3.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.1, type = "fpm")

#std
model.fit_exp3.3.2 = ugarchroll(garchspec3.3.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.2, type = "fpm")

#ged
model.fit_exp3.3.3 = ugarchroll(garchspec3.3.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.3, type = "fpm")

#snorm
model.fit_exp3.3.4 = ugarchroll(garchspec3.3.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.4, type = "fpm")

#sstd
model.fit_exp3.3.5 = ugarchroll(garchspec3.3.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.5, type = "fpm")

#sged
model.fit_exp3.3.6 = ugarchroll(garchspec3.3.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.6, type = "fpm")

################################################################################
# Sentiment mean + std
################################################################################
#sGARCH
#norm
model.fit_exp4.1.1 = ugarchroll(garchspec4.1.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.1.1, type = "fpm")

#std
model.fit_exp4.1.2 = ugarchroll(garchspec4.1.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.1.2, type = "fpm")

#ged
model.fit_exp4.1.3 = ugarchroll(garchspec4.1.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.1.3, type = "fpm")

#snorm
model.fit_exp4.1.4 = ugarchroll(garchspec4.1.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.1.4, type = "fpm")

#sstd
model.fit_exp4.1.5 = ugarchroll(garchspec4.1.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.1.5, type = "fpm")

#sged
model.fit_exp4.1.6 = ugarchroll(garchspec4.1.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.1.6, type = "fpm")

#eGARCH
#norm
model.fit_exp4.2.1 = ugarchroll(garchspec4.2.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.2.1, type = "fpm")

#std
model.fit_exp4.2.2 = ugarchroll(garchspec4.2.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.2.2, type = "fpm")

#ged
model.fit_exp4.2.3 = ugarchroll(garchspec4.2.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.2.3, type = "fpm")

#snorm
model.fit_exp4.2.4 = ugarchroll(garchspec4.2.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.2.4, type = "fpm")

#sstd
model.fit_exp4.2.5 = ugarchroll(garchspec4.2.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.2.5, type = "fpm")

#sged
model.fit_exp4.2.6 = ugarchroll(garchspec4.2.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.2.6, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp4.3.1 = ugarchroll(garchspec4.3.1, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.3.1, type = "fpm")

#std
model.fit_exp4.3.2 = ugarchroll(garchspec4.3.2, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.3.2, type = "fpm")

#ged
model.fit_exp4.3.3 = ugarchroll(garchspec4.3.3, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.3.3, type = "fpm")

#snorm
model.fit_exp4.3.4 = ugarchroll(garchspec4.3.4, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.3.4, type = "fpm")

#sstd
model.fit_exp4.3.5 = ugarchroll(garchspec4.3.5, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.3.5, type = "fpm")

#sged
model.fit_exp4.3.6 = ugarchroll(garchspec4.3.6, close_rets, n.ahead = 1, n.start = 400, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp4.3.6, type = "fpm")


################################################################################
#One step ahead forecast using expanding windows (start: 888 half); refit every 50)
################################################################################
# no external regressor
################################################################################
#sGARCH
#norm
model.fit_exp0.1.1_1 = ugarchroll(garchspec0.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.1_, type = "fpm")

#std
model.fit_exp0.1.2_1 = ugarchroll(garchspec0.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.2_1, type = "fpm")

#ged
model.fit_exp0.1.3_1 = ugarchroll(garchspec0.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.3_1, type = "fpm")

#snorm
model.fit_exp0.1.4_1 = ugarchroll(garchspec0.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.4_1, type = "fpm")

#sstd
model.fit_exp0.1.5_1 = ugarchroll(garchspec0.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.5_1, type = "fpm")

#sged
model.fit_exp0.1.6_1 = ugarchroll(garchspec0.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.1.6_1, type = "fpm")

#eGARCH
#norm
model.fit_exp0.2.1_1 = ugarchroll(garchspec0.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.1_1, type = "fpm")

#std
model.fit_exp0.2.2_1 = ugarchroll(garchspec0.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.2_1, type = "fpm")

#ged
model.fit_exp0.2.3_1 = ugarchroll(garchspec0.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.3_1, type = "fpm")

#snorm
model.fit_exp0.2.4_1 = ugarchroll(garchspec0.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.4_1, type = "fpm")

#sstd
model.fit_exp0.2.5_1 = ugarchroll(garchspec0.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.5_1, type = "fpm")

#sged
model.fit_exp0.2.6_1 = ugarchroll(garchspec0.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.2.6_1, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp0.3.1_1 = ugarchroll(garchspec0.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.1_1, type = "fpm")

#std
model.fit_exp0.3.2_1 = ugarchroll(garchspec0.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.2_1, type = "fpm")

#ged
model.fit_exp0.3.3_1 = ugarchroll(garchspec0.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.3_1, type = "fpm")

#snorm
model.fit_exp0.3.4_1 = ugarchroll(garchspec0.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.4_1, type = "fpm")

#sstd
model.fit_exp0.3.5_1 = ugarchroll(garchspec0.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.5_1, type = "fpm")

#sged
model.fit_exp0.3.6_1 = ugarchroll(garchspec0.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp0.3.6_1, type = "fpm")

################################################################################
# average daily sentiment 
################################################################################
#sGARCH
#norm
model.fit_exp1.1.1_1 = ugarchroll(garchspec1.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.1_1, type = "fpm")

#std
model.fit_exp1.1.2_1 = ugarchroll(garchspec1.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.2_1, type = "fpm")

#ged
model.fit_exp1.1.3_1 = ugarchroll(garchspec1.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.3_1, type = "fpm")

#snorm
model.fit_exp1.1.4_1 = ugarchroll(garchspec1.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.4_1, type = "fpm")

#sstd
model.fit_exp1.1.5_1 = ugarchroll(garchspec1.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.5_1, type = "fpm")

#sged
model.fit_exp1.1.6_1 = ugarchroll(garchspec1.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.1.6_1, type = "fpm")

#eGARCH
#norm
model.fit_exp1.2.1_1 = ugarchroll(garchspec1.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.1_1, type = "fpm")

#std
model.fit_exp1.2.2_1 = ugarchroll(garchspec1.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.2_1, type = "fpm")

#ged
model.fit_exp1.2.3_1 = ugarchroll(garchspec1.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.3_1, type = "fpm")

#snorm
model.fit_exp1.2.4_1 = ugarchroll(garchspec1.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.4_1, type = "fpm")

#sstd
model.fit_exp1.2.5_1 = ugarchroll(garchspec1.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.5_1, type = "fpm")

#sged
model.fit_exp1.2.6_1 = ugarchroll(garchspec1.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.2.6_1, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp1.3.1_1 = ugarchroll(garchspec1.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.1_1, type = "fpm")

#std
model.fit_exp1.3.2_1 = ugarchroll(garchspec1.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.2_1, type = "fpm")

#ged
model.fit_exp1.3.3_1 = ugarchroll(garchspec1.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.3_1, type = "fpm")

#snorm
model.fit_exp1.3.4_1 = ugarchroll(garchspec1.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.4_1, type = "fpm")

#sstd
model.fit_exp1.3.5_1 = ugarchroll(garchspec1.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.5_1, type = "fpm")

#sged
model.fit_exp1.3.6_1 = ugarchroll(garchspec1.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp1.3.6_1, type = "fpm")

################################################################################
# average daily sentiment split into positive and negative
################################################################################
#sGARCH
#norm
model.fit_exp2.1.1_1 = ugarchroll(garchspec2.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.1_1, type = "fpm")

#std
model.fit_exp2.1.2_1 = ugarchroll(garchspec2.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.2_1, type = "fpm")

#ged
model.fit_exp2.1.3_1 = ugarchroll(garchspec2.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.3_1, type = "fpm")

#snorm
model.fit_exp2.1.4_1 = ugarchroll(garchspec2.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.4_1, type = "fpm")

#sstd
model.fit_exp2.1.5_1 = ugarchroll(garchspec2.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.5_1, type = "fpm")

#sged
model.fit_exp2.1.6_1 = ugarchroll(garchspec2.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.1.6_1, type = "fpm")

#eGARCH
#norm
model.fit_exp2.2.1_1 = ugarchroll(garchspec2.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.1_1, type = "fpm")

#std
model.fit_exp2.2.2_1 = ugarchroll(garchspec2.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.2_1, type = "fpm")

#ged
model.fit_exp2.2.3_1 = ugarchroll(garchspec2.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.3_1, type = "fpm")

#snorm
model.fit_exp2.2.4_1 = ugarchroll(garchspec2.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.4_1, type = "fpm")

#sstd
model.fit_exp2.2.5_1 = ugarchroll(garchspec2.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.5_1, type = "fpm")

#sged
model.fit_exp2.2.6_1 = ugarchroll(garchspec2.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.2.6_1, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp2.3.1_1 = ugarchroll(garchspec2.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.1_1, type = "fpm")

#std
model.fit_exp2.3.2_1 = ugarchroll(garchspec2.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.2_1, type = "fpm")

#ged
model.fit_exp2.3.3_1 = ugarchroll(garchspec2.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.3_1, type = "fpm")

#snorm
model.fit_exp2.3.4_1 = ugarchroll(garchspec2.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.4_1, type = "fpm")

#sstd
model.fit_exp2.3.5_1 = ugarchroll(garchspec2.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.5_1, type = "fpm")

#sged
model.fit_exp2.3.6_1 = ugarchroll(garchspec2.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp2.3.6_1, type = "fpm")

################################################################################
# average daily sentiment split into positive and negative (MVA)
################################################################################
#sGARCH
#norm
model.fit_exp3.1.1_1 = ugarchroll(garchspec3.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.1_1, type = "fpm")

#std
model.fit_exp3.1.2_1 = ugarchroll(garchspec3.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.2_1, type = "fpm")

#ged
model.fit_exp3.1.3_1 = ugarchroll(garchspec3.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.3_1, type = "fpm")

#snorm
model.fit_exp3.1.4_1 = ugarchroll(garchspec3.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.4_1, type = "fpm")

#sstd
model.fit_exp3.1.5_1 = ugarchroll(garchspec3.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.5_1, type = "fpm")

#sged
model.fit_exp3.1.6_1 = ugarchroll(garchspec3.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.1.6_1, type = "fpm")

#eGARCH
#norm
model.fit_exp3.2.1_1 = ugarchroll(garchspec3.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.1_1, type = "fpm")

#std
model.fit_exp3.2.2_1 = ugarchroll(garchspec3.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.2_1, type = "fpm")

#ged
model.fit_exp3.2.3_1 = ugarchroll(garchspec3.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.3_1, type = "fpm")

#snorm
model.fit_exp3.2.4_1 = ugarchroll(garchspec3.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.4_1, type = "fpm")

#sstd
model.fit_exp3.2.5_1 = ugarchroll(garchspec3.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.5_1, type = "fpm")

#sged
model.fit_exp3.2.6_1 = ugarchroll(garchspec3.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.2.6_1, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp3.3.1_1 = ugarchroll(garchspec3.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.1_1, type = "fpm")

#std
model.fit_exp3.3.2_1 = ugarchroll(garchspec3.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.2_1, type = "fpm")

#ged
model.fit_exp3.3.3_1 = ugarchroll(garchspec3.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.3_1, type = "fpm")

#snorm
model.fit_exp3.3.4_1 = ugarchroll(garchspec3.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.4_1, type = "fpm")

#sstd
model.fit_exp3.3.5_1 = ugarchroll(garchspec3.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.5_1, type = "fpm")

#sged
model.fit_exp3.3.6_1 = ugarchroll(garchspec3.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                keep.coef = TRUE)
report(model.fit_exp3.3.6_1, type = "fpm")

################################################################################
# Sentiment mean + std
################################################################################
#sGARCH
#norm
model.fit_exp4.1.1_1 = ugarchroll(garchspec4.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.1_1, type = "fpm")

#std
model.fit_exp4.1.2_1 = ugarchroll(garchspec4.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.2_1, type = "fpm")

#ged
model.fit_exp4.1.3_1 = ugarchroll(garchspec4.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.3_1, type = "fpm")

#snorm
model.fit_exp4.1.4_1 = ugarchroll(garchspec4.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.4_1, type = "fpm")

#sstd
model.fit_exp4.1.5_1 = ugarchroll(garchspec4.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.5_1, type = "fpm")

#sged
model.fit_exp4.1.6_1 = ugarchroll(garchspec4.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.6_1, type = "fpm")

#eGARCH
#norm
model.fit_exp4.2.1_1 = ugarchroll(garchspec4.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.1_1, type = "fpm")

#std
model.fit_exp4.2.2_1 = ugarchroll(garchspec4.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.2_1, type = "fpm")

#ged
model.fit_exp4.2.3_1 = ugarchroll(garchspec4.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.3_1, type = "fpm")

#snorm
model.fit_exp4.2.4_1 = ugarchroll(garchspec4.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.4_1, type = "fpm")

#sstd
model.fit_exp4.2.5_1 = ugarchroll(garchspec4.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.5_1, type = "fpm")

#sged
model.fit_exp4.2.6_1 = ugarchroll(garchspec4.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.6_1, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp4.3.1_1 = ugarchroll(garchspec4.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.1_1, type = "fpm")

#std
model.fit_exp4.3.2_1 = ugarchroll(garchspec4.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.2_1, type = "fpm")

#ged
model.fit_exp4.3.3_1 = ugarchroll(garchspec4.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.3_1, type = "fpm")

#snorm
model.fit_exp4.3.4_1 = ugarchroll(garchspec4.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.4_1, type = "fpm")

#sstd
model.fit_exp4.3.5_1 = ugarchroll(garchspec4.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.5_1, type = "fpm")

#sged
model.fit_exp4.3.6_1 = ugarchroll(garchspec4.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 50, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.6_1, type = "fpm")

################################################################################
#One step ahead forecast using expanding windows (start: 888 half); refit every 25)
################################################################################
# no external regressor
################################################################################
#sGARCH
#norm
model.fit_exp0.1.1_2 = ugarchroll(garchspec0.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.1.1_2, type = "fpm")

#std
model.fit_exp0.1.2_2 = ugarchroll(garchspec0.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.1.2_2, type = "fpm")

#ged
model.fit_exp0.1.3_2 = ugarchroll(garchspec0.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.1.3_2, type = "fpm")

#snorm
model.fit_exp0.1.4_2 = ugarchroll(garchspec0.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.1.4_2, type = "fpm")

#sstd
model.fit_exp0.1.5_2 = ugarchroll(garchspec0.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.1.5_2, type = "fpm")

#sged
model.fit_exp0.1.6_2 = ugarchroll(garchspec0.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.1.6_2, type = "fpm")

#eGARCH
#norm
model.fit_exp0.2.1_2 = ugarchroll(garchspec0.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.2.1_2, type = "fpm")

#std
model.fit_exp0.2.2_2 = ugarchroll(garchspec0.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.2.2_2, type = "fpm")

#ged
model.fit_exp0.2.3_2 = ugarchroll(garchspec0.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.2.3_2, type = "fpm")

#snorm
model.fit_exp0.2.4_2 = ugarchroll(garchspec0.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.2.4_2, type = "fpm")

#sstd
model.fit_exp0.2.5_2 = ugarchroll(garchspec0.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.2.5_2, type = "fpm")

#sged
model.fit_exp0.2.6_2 = ugarchroll(garchspec0.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.2.6_2, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp0.3.1_2 = ugarchroll(garchspec0.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.3.1_2, type = "fpm")

#std
model.fit_exp0.3.2_2 = ugarchroll(garchspec0.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.3.2_2, type = "fpm")

#ged
model.fit_exp0.3.3_2 = ugarchroll(garchspec0.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.3.3_2, type = "fpm")

#snorm
model.fit_exp0.3.4_2 = ugarchroll(garchspec0.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.3.4_2, type = "fpm")

#sstd
model.fit_exp0.3.5_2 = ugarchroll(garchspec0.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.3.5_2, type = "fpm")

#sged
model.fit_exp0.3.6_2 = ugarchroll(garchspec0.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp0.3.6_2, type = "fpm")

################################################################################
# average daily sentiment 
################################################################################
#sGARCH
#norm
model.fit_exp1.1.1_2 = ugarchroll(garchspec1.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.1.1_2, type = "fpm")

#std
model.fit_exp1.1.2_2 = ugarchroll(garchspec1.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.1.2_2, type = "fpm")

#ged
model.fit_exp1.1.3_2 = ugarchroll(garchspec1.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.1.3_2, type = "fpm")

#snorm
model.fit_exp1.1.4_2 = ugarchroll(garchspec1.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.1.4_2, type = "fpm")

#sstd
model.fit_exp1.1.5_2 = ugarchroll(garchspec1.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.1.5_2, type = "fpm")

#sged
model.fit_exp1.1.6_2 = ugarchroll(garchspec1.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.1.6_2, type = "fpm")

#eGARCH
#norm
model.fit_exp1.2.1_2 = ugarchroll(garchspec1.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.2.1_2, type = "fpm")

#std
model.fit_exp1.2.2_2 = ugarchroll(garchspec1.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.2.2_2, type = "fpm")

#ged
model.fit_exp1.2.3_2 = ugarchroll(garchspec1.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.2.3_2, type = "fpm")

#snorm
model.fit_exp1.2.4_2 = ugarchroll(garchspec1.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.2.4_2, type = "fpm")

#sstd
model.fit_exp1.2.5_2 = ugarchroll(garchspec1.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.2.5_2, type = "fpm")

#sged
model.fit_exp1.2.6_2 = ugarchroll(garchspec1.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.2.6_2, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp1.3.1_2 = ugarchroll(garchspec1.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.3.1_2, type = "fpm")

#std
model.fit_exp1.3.2_2 = ugarchroll(garchspec1.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.3.2_2, type = "fpm")

#ged
model.fit_exp1.3.3_2 = ugarchroll(garchspec1.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.3.3_2, type = "fpm")

#snorm
model.fit_exp1.3.4_2 = ugarchroll(garchspec1.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.3.4_2, type = "fpm")

#sstd
model.fit_exp1.3.5_2 = ugarchroll(garchspec1.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.3.5_2, type = "fpm")

#sged
model.fit_exp1.3.6_2 = ugarchroll(garchspec1.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp1.3.6_2, type = "fpm")

################################################################################
# average daily sentiment split into positive and negative
################################################################################
#sGARCH
#norm
model.fit_exp2.1.1_2 = ugarchroll(garchspec2.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.1.1_2, type = "fpm")

#std
model.fit_exp2.1.2_2 = ugarchroll(garchspec2.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.1.2_2, type = "fpm")

#ged
model.fit_exp2.1.3_2 = ugarchroll(garchspec2.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.1.3_2, type = "fpm")

#snorm
model.fit_exp2.1.4_2 = ugarchroll(garchspec2.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.1.4_2, type = "fpm")

#sstd
model.fit_exp2.1.5_2 = ugarchroll(garchspec2.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.1.5_2, type = "fpm")

#sged
model.fit_exp2.1.6_2 = ugarchroll(garchspec2.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.1.6_2, type = "fpm")

#eGARCH
#norm
model.fit_exp2.2.1_2 = ugarchroll(garchspec2.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.2.1_2, type = "fpm")

#std
model.fit_exp2.2.2_2 = ugarchroll(garchspec2.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.2.2_2, type = "fpm")

#ged
model.fit_exp2.2.3_2 = ugarchroll(garchspec2.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.2.3_2, type = "fpm")

#snorm
model.fit_exp2.2.4_2 = ugarchroll(garchspec2.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.2.4_2, type = "fpm")

#sstd
model.fit_exp2.2.5_2 = ugarchroll(garchspec2.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.2.5_2, type = "fpm")

#sged
model.fit_exp2.2.6_2 = ugarchroll(garchspec2.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.2.6_2, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp2.3.1_2 = ugarchroll(garchspec2.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.3.1_2, type = "fpm")

#std
model.fit_exp2.3.2_2 = ugarchroll(garchspec2.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.3.2_2, type = "fpm")

#ged
model.fit_exp2.3.3_2 = ugarchroll(garchspec2.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.3.3_2, type = "fpm")

#snorm
model.fit_exp2.3.4_2 = ugarchroll(garchspec2.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.3.4_2, type = "fpm")

#sstd
model.fit_exp2.3.5_2 = ugarchroll(garchspec2.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.3.5_2, type = "fpm")

#sged
model.fit_exp2.3.6_2 = ugarchroll(garchspec2.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp2.3.6_2, type = "fpm")

################################################################################
# average daily sentiment split into positive and negative (MVA)
################################################################################
#sGARCH
#norm
model.fit_exp3.1.1_2 = ugarchroll(garchspec3.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.1.1_2, type = "fpm")

#std
model.fit_exp3.1.2_2 = ugarchroll(garchspec3.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.1.2_2, type = "fpm")

#ged
model.fit_exp3.1.3_2 = ugarchroll(garchspec3.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.1.3_2, type = "fpm")

#snorm
model.fit_exp3.1.4_2 = ugarchroll(garchspec3.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.1.4_2, type = "fpm")

#sstd
model.fit_exp3.1.5_2 = ugarchroll(garchspec3.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.1.5_2, type = "fpm")

#sged
model.fit_exp3.1.6_2 = ugarchroll(garchspec3.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.1.6_2, type = "fpm")

#eGARCH
#norm
model.fit_exp3.2.1_2 = ugarchroll(garchspec3.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.2.1_2, type = "fpm")

#std
model.fit_exp3.2.2_2 = ugarchroll(garchspec3.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.2.2_2, type = "fpm")

#ged
model.fit_exp3.2.3_2 = ugarchroll(garchspec3.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.2.3_2, type = "fpm")

#snorm
model.fit_exp3.2.4_2 = ugarchroll(garchspec3.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.2.4_2, type = "fpm")

#sstd
model.fit_exp3.2.5_2 = ugarchroll(garchspec3.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.2.5_2, type = "fpm")

#sged
model.fit_exp3.2.6_2 = ugarchroll(garchspec3.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.2.6_2, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp3.3.1_2 = ugarchroll(garchspec3.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.3.1_2, type = "fpm")

#std
model.fit_exp3.3.2_2 = ugarchroll(garchspec3.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.3.2_2, type = "fpm")

#ged
model.fit_exp3.3.3_2 = ugarchroll(garchspec3.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.3.3_2, type = "fpm")

#snorm
model.fit_exp3.3.4_2 = ugarchroll(garchspec3.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.3.4_2, type = "fpm")

#sstd
model.fit_exp3.3.5_2 = ugarchroll(garchspec3.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.3.5_2, type = "fpm")

#sged
model.fit_exp3.3.6_2 = ugarchroll(garchspec3.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp3.3.6_2, type = "fpm")

################################################################################
# Sentiment_mean + std
################################################################################
#sGARCH
#norm
model.fit_exp4.1.1_2 = ugarchroll(garchspec4.1.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.1_2, type = "fpm")

#std
model.fit_exp4.1.2_2 = ugarchroll(garchspec4.1.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.2_2, type = "fpm")

#ged
model.fit_exp4.1.3_2 = ugarchroll(garchspec4.1.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.3_2, type = "fpm")

#snorm
model.fit_exp4.1.4_2 = ugarchroll(garchspec4.1.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.4_2, type = "fpm")

#sstd
model.fit_exp4.1.5_2 = ugarchroll(garchspec4.1.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.5_2, type = "fpm")

#sged
model.fit_exp4.1.6_2 = ugarchroll(garchspec4.1.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.1.6_2, type = "fpm")

#eGARCH
#norm
model.fit_exp4.2.1_2 = ugarchroll(garchspec4.2.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.1_2, type = "fpm")

#std
model.fit_exp4.2.2_2 = ugarchroll(garchspec4.2.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.2_2, type = "fpm")

#ged
model.fit_exp4.2.3_2 = ugarchroll(garchspec4.2.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.3_2, type = "fpm")

#snorm
model.fit_exp4.2.4_2 = ugarchroll(garchspec4.2.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.4_2, type = "fpm")

#sstd
model.fit_exp4.2.5_2 = ugarchroll(garchspec4.2.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.5_2, type = "fpm")

#sged
model.fit_exp4.2.6_2 = ugarchroll(garchspec4.2.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.2.6_2, type = "fpm")

#GJR-GARCH
#norm
model.fit_exp4.3.1_2 = ugarchroll(garchspec4.3.1, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.1_2, type = "fpm")

#std
model.fit_exp4.3.2_2 = ugarchroll(garchspec4.3.2, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.2_2, type = "fpm")

#ged
model.fit_exp4.3.3_2 = ugarchroll(garchspec4.3.3, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.3_2, type = "fpm")

#snorm
model.fit_exp4.3.4_2 = ugarchroll(garchspec4.3.4, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.4_2, type = "fpm")

#sstd
model.fit_exp4.3.5_2 = ugarchroll(garchspec4.3.5, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.5_2, type = "fpm")

#sged
model.fit_exp4.3.6_2 = ugarchroll(garchspec4.3.6, close_rets, n.ahead = 1, n.start = 888, refit.every = 25, 
                                  refit.window = "expanding", solver = "hybrid", calculate.VaR = FALSE, 
                                  keep.coef = TRUE)
report(model.fit_exp4.3.6_2, type = "fpm")
