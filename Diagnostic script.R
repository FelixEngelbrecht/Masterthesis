###################################################################################

############################# Master's Thesis #####################################
#Does sentiment from relevant news improve volatility forecast from GARCH models? -
#An application of natural language #processing


###################################################################################

#if packagaes are not installed, it needs to be done beforehand

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
library(psych)
citation("rugarch")

df = read.csv("Sent_SP500_1.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
df2 = read.csv("merged_pol_Sent.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
df3 = read.csv("Sent_mva.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
df4 = read.csv("mva_std_Sent.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)

#plot ACF
autocorrelation = acf(df$close, plot = FALSE)
plot(autocorrelation, main = "")

#check for stationarity
adf.test(df$close)

#plot of the time series of close prices
df$Date = as.Date(df$Date, "%Y-%m-%d")
date=as.Date(df$Date, "%Y-%m-%d")
plot(df$Date, df$close, type ="l", xlab = "Year", ylab = "Price")

#construct close returns to stationarize the time series
close_rets = diff(log(df$close))

#diagnostics of the time series log returns
plot(date[-1], close_rets, type ="l", xlab = "Year", ylab = "log return")
autocorrelation_rets = acf(close_rets, plot = FALSE)
plot(autocorrelation_rets, main = "")
adf.test(close_rets)

#summary statistics of close returns (for accuracy it is calculated one by one)
mean(close_rets)
sd(close_rets)
median(close_rets)
min(close_rets)
max(close_rets)
skew(close_rets)
kurtosi(close_rets)

#check for best ARIMA-model

fit1 = auto.arima(close_rets, trace=TRUE, test="kpss",  ic="aic")
adf.test(fit1$residuals^2)
acf(fit1$residuals^2)
Box.test(fit1$residuals^2,lag=30, type="Ljung-Box") 

#################################################################################
#plot the sentiment time series
#################################################################################

#daily average sentiment
Sent= df[,2]
plot.ts(Sent)
plot(date, Sent, type ="l", xlab = "Year", ylab = "Sentiment")

#positive and negative average sentiment
pos_sent = df2$pos_Sent
plot(date, pos_sent, type ="l", xlab = "Year", ylab = "Sentiment")

neg_sent = df2$neg_Sent
plot(date, neg_sent, type ="l", xlab = "Year", ylab = "Sentiment")

#positive and negative average sentiment (moving average)
pos_mva_sent = df3$pos_Sent_mva
plot(date, pos_mva_sent, type ="l", xlab = "Year", ylab = "Sentiment")

neg_mva_sent = df3$neg_Sent_mva
plot(date, neg_mva_sent, type ="l", xlab = "Year", ylab = "Sentiment")

#positive and negative average sentiment exceed on std from mean
pos_std_sent = df4$pos_Sent_x
plot(date, pos_std_sent, type ="l", xlab = "Year", ylab = "Sentiment")

neg_std_sent = df4$neg_Sent_x
plot(date, neg_std_sent, type ="l", xlab = "Year", ylab = "Sentiment")

#################################################################################
#ADF-Tests of sentiment time series
#################################################################################

#daily average sentiment
adf.test(Sent)

#positive and negative average sentiment
adf.test(pos_sent)
adf.test(neg_sent)

#positive and negative average sentiment (moving average)
#replace nas to perform adf test
pos_mva_sent[1:4] = df3$pos_Sent[1:4]
neg_mva_sent[1:4] = df3$neg_Sent[1:4]
adf.test(pos_mva_sent)
adf.test(neg_mva_sent)

#positive and negative average sentiment exceed on std from mean
adf.test(pos_std_sent)
adf.test(neg_std_sent)

