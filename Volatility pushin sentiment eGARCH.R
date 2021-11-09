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
citation("rugarch")

df = read.csv("Sent_mva.csv", header = TRUE, sep = ",", dec = ".", check.names = TRUE)
df = na.omit(df)

fit2 = auto.arima(pos_Sent_mva, trace=TRUE, test="kpss",  ic="aic")
adf.test(fit2$residuals^2)
Box.test(fit2$residuals^2,lag=30, type="Ljung-Box")

fit3 = auto.arima(neg_Sent_mva, trace=TRUE, test="kpss",  ic="aic")
adf.test(fit3$residuals^2)
Box.test(fit3$residuals^2,lag=30, type="Ljung-Box") 
Box.test(pos_Sent_mva, lag = 30, type = "Ljung-Box")
Box.test(neg_Sent_mva, lag = 30, type = "Ljung-Box")

# Specify a standard GARCH model with constant mean
garchspec11 <- ugarchspec(mean.model = list(armaOrder = c(0,1)),
                        variance.model = list(model = "eGARCH"), 
                        distribution.model = "norm")

# Estimate the model
garchfit11 <- ugarchfit(data = close_rets, spec = garchspec11)
garchfit11
# Use the method sigma to retrieve the estimated volatilities 
garchvol11 <- sigma(garchfit11)

# Plot the volatility
plot(garchvol11)

#get the lagged sentiment

lagged_pos_Sent_mva = pos_Sent_mva[-1]
lagged_neg_Sent_mva = neg_Sent_mva[-1]
garchvol_ext11 = garchvol11[-1775]

###################################################################################
#fit garch for pushing hypothesis
###################################################################################
#norm
model.spec_ex11 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext11), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "norm")

fit_ex11 =ugarchfit(spec=model.spec_ex11,data = lagged_pos_Sent_mva)
fit_ex11.1 =ugarchfit(spec=model.spec_ex11,data = lagged_neg_Sent_mva)
fit_ex11
fit_ex11.1
##################################################################################
#std
model.spec_ex12 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext11), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "std")

fit_ex12 =ugarchfit(spec=model.spec_ex12,data = lagged_pos_Sent_mva)
fit_ex12.1 =ugarchfit(spec=model.spec_ex12,data = lagged_neg_Sent_mva)
fit_ex12
fit_ex12.1
###################################################################################
#ged
model.spec_ex13 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext11), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "ged")

fit_ex13 =ugarchfit(spec=model.spec_ex13,data = lagged_pos_Sent_mva)
fit_ex13.1 =ugarchfit(spec=model.spec_ex13,data = lagged_neg_Sent_mva)
fit_ex13
fit_ex13.1
###################################################################################
#snorm
model.spec_ex14 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext11), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "snorm")

fit_ex14 =ugarchfit(spec=model.spec_ex14,data = lagged_pos_Sent_mva)
fit_ex14.1 =ugarchfit(spec=model.spec_ex14,data = lagged_neg_Sent_mva)
fit_ex14
fit_ex14.1
###################################################################################
#sstd
model.spec_ex15 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext11), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sstd")

fit_ex15 =ugarchfit(spec=model.spec_ex15,data = lagged_pos_Sent_mva)
fit_ex15.1 =ugarchfit(spec=model.spec_ex15,data = lagged_neg_Sent_mva)
fit_ex15
fit_ex15.1
#################################################################################
#sged
model.spec_ex16 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext11), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sged")

fit_ex16 =ugarchfit(spec=model.spec_ex16,data = lagged_pos_Sent_mva)
fit_ex16.1 =ugarchfit(spec=model.spec_ex16,data = lagged_neg_Sent_mva)
fit_ex16
fit_ex16.1

# Specify a standard GARCH model with constant mean
garchspec21 <- ugarchspec(mean.model = list(armaOrder = c(0,1)),
                          variance.model = list(model = "eGARCH"), 
                          distribution.model = "std")

# Estimate the model
garchfit21 <- ugarchfit(data = close_rets, spec = garchspec21)
garchfit21
# Use the method sigma to retrieve the estimated volatilities 
garchvol21 <- sigma(garchfit21)

# Plot the volatility
plot(garchvol21)
garchvol_ext21 = garchvol21[-1775]

#fit garch for pushing hypothesis
###################################################################################
#norm
model.spec_ex21 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext21), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "norm")

fit_ex21 =ugarchfit(spec=model.spec_ex21,data = lagged_pos_Sent_mva)
fit_ex21.1 =ugarchfit(spec=model.spec_ex21,data = lagged_neg_Sent_mva)
fit_ex21
fit_ex21.1
##################################################################################
#std
model.spec_ex22 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext21), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "std")

fit_ex22 =ugarchfit(spec=model.spec_ex22,data = lagged_pos_Sent_mva)
fit_ex22.1 =ugarchfit(spec=model.spec_ex22,data = lagged_neg_Sent_mva)
fit_ex22
fit_ex22.1
###################################################################################
#ged
model.spec_ex23 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext21), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "ged")

fit_ex23 =ugarchfit(spec=model.spec_ex23,data = lagged_pos_Sent_mva)
fit_ex23.1 =ugarchfit(spec=model.spec_ex23,data = lagged_neg_Sent_mva)
fit_ex23
fit_ex23.1
###################################################################################
#snorm
model.spec_ex24 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext21), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "snorm")

fit_ex24 =ugarchfit(spec=model.spec_ex24,data = lagged_pos_Sent_mva)
fit_ex24.1 =ugarchfit(spec=model.spec_ex24,data = lagged_neg_Sent_mva)
fit_ex24
fit_ex24.1
###################################################################################
#sstd
model.spec_ex25 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext21), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sstd")

fit_ex25 =ugarchfit(spec=model.spec_ex25,data = lagged_pos_Sent_mva)
fit_ex25.1 =ugarchfit(spec=model.spec_ex25,data = lagged_neg_Sent_mva)
fit_ex25
fit_ex25.1
#################################################################################
#sged
model.spec_ex26 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext21), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sged")

fit_ex26 =ugarchfit(spec=model.spec_ex26,data = lagged_pos_Sent_mva)
fit_ex26.1 =ugarchfit(spec=model.spec_ex26,data = lagged_neg_Sent_mva)
fit_ex26
fit_ex26.1

##################################################################################
# Specify a standard GARCH model ged
garchspec31 <- ugarchspec(mean.model = list(armaOrder = c(0,1)),
                          variance.model = list(model = "eGARCH"), 
                          distribution.model = "ged")

# Estimate the model
garchfit31 <- ugarchfit(data = close_rets, spec = garchspec31)
garchfit31
# Use the method sigma to retrieve the estimated volatilities 
garchvol31 <- sigma(garchfit31)

# Plot the volatility
plot(garchvol31)
garchvol_ext31 = garchvol31[-1775]

#fit garch for pushing hypothesis
###################################################################################
#norm
model.spec_ex31 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext31), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "norm")

fit_ex31 =ugarchfit(spec=model.spec_ex31,data = lagged_pos_Sent_mva)
fit_ex31.1 =ugarchfit(spec=model.spec_ex31,data = lagged_neg_Sent_mva)
fit_ex31
fit_ex31.1
##################################################################################
#std
model.spec_ex32 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext31), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "std")

fit_ex32 =ugarchfit(spec=model.spec_ex32,data = lagged_pos_Sent_mva)
fit_ex32.1 =ugarchfit(spec=model.spec_ex32,data = lagged_neg_Sent_mva)
fit_ex32
fit_ex32.1
###################################################################################
#ged
model.spec_ex33 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext31), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "ged")

fit_ex33 =ugarchfit(spec=model.spec_ex33,data = lagged_pos_Sent_mva)
fit_ex33.1 =ugarchfit(spec=model.spec_ex33,data = lagged_neg_Sent_mva)
fit_ex33
fit_ex33.1
###################################################################################
#snorm
model.spec_ex34 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext31), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "snorm")

fit_ex34 =ugarchfit(spec=model.spec_ex34,data = lagged_pos_Sent_mva)
fit_ex34.1 =ugarchfit(spec=model.spec_ex34,data = lagged_neg_Sent_mva)
fit_ex34
fit_ex34.1
###################################################################################
#sstd
model.spec_ex35 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext31), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sstd")

fit_ex35 =ugarchfit(spec=model.spec_ex35,data = lagged_pos_Sent_mva)
fit_ex35.1 =ugarchfit(spec=model.spec_ex35,data = lagged_neg_Sent_mva)
fit_ex35
fit_ex35.1
#################################################################################
#sged
model.spec_ex36 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext31), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sged")

fit_ex36 =ugarchfit(spec=model.spec_ex36,data = lagged_pos_Sent_mva)
fit_ex36.1 =ugarchfit(spec=model.spec_ex36,data = lagged_neg_Sent_mva)
fit_ex36
fit_ex36.1


##################################################################################
# Specify a standard GARCH model snorm
garchspec41 <- ugarchspec(mean.model = list(armaOrder = c(0,1)),
                          variance.model = list(model = "eGARCH"), 
                          distribution.model = "snorm")

# Estimate the model
garchfit41 <- ugarchfit(data = close_rets, spec = garchspec41)
garchfit41
# Use the method sigma to retrieve the estimated volatilities 
garchvol41 <- sigma(garchfit41)

# Plot the volatility
plot(garchvol41)
garchvol_ext41 = garchvol41[-1775]

#fit garch for pushing hypothesis
###################################################################################
#norm
model.spec_ex41 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext41), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "norm")

fit_ex41 =ugarchfit(spec=model.spec_ex41,data = lagged_pos_Sent_mva)
fit_ex41.1 =ugarchfit(spec=model.spec_ex41,data = lagged_neg_Sent_mva)
fit_ex41
fit_ex41.1
##################################################################################
#std
model.spec_ex42 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext41), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "std")

fit_ex42 =ugarchfit(spec=model.spec_ex42,data = lagged_pos_Sent_mva)
fit_ex42.1 =ugarchfit(spec=model.spec_ex42,data = lagged_neg_Sent_mva)
fit_ex42
fit_ex42.1
###################################################################################
#ged
model.spec_ex43 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext41), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "ged")

fit_ex43 =ugarchfit(spec=model.spec_ex43,data = lagged_pos_Sent_mva)
fit_ex43.1 =ugarchfit(spec=model.spec_ex43,data = lagged_neg_Sent_mva)
fit_ex43
fit_ex43.1
###################################################################################
#snorm
model.spec_ex44 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext41), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "snorm")

fit_ex44 =ugarchfit(spec=model.spec_ex44,data = lagged_pos_Sent_mva)
fit_ex44.1 =ugarchfit(spec=model.spec_ex44,data = lagged_neg_Sent_mva)
fit_ex44
fit_ex44.1
###################################################################################
#sstd
model.spec_ex45 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext41), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sstd")

fit_ex45 =ugarchfit(spec=model.spec_ex45,data = lagged_pos_Sent_mva)
fit_ex45.1 =ugarchfit(spec=model.spec_ex45,data = lagged_neg_Sent_mva)
fit_ex45
fit_ex45.1
#################################################################################
#sged
model.spec_ex46 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext41), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sged")

fit_ex46 =ugarchfit(spec=model.spec_ex46,data = lagged_pos_Sent_mva)
fit_ex46.1 =ugarchfit(spec=model.spec_ex46,data = lagged_neg_Sent_mva)
fit_ex46
fit_ex46.1

##################################################################################
# Specify a standard GARCH model sstd
garchspec51 <- ugarchspec(mean.model = list(armaOrder = c(0,1)),
                          variance.model = list(model = "eGARCH"), 
                          distribution.model = "sstd")

# Estimate the model
garchfit51 <- ugarchfit(data = close_rets, spec = garchspec51)
garchfit51
# Use the method sigma to retrieve the estimated volatilities 
garchvol51 <- sigma(garchfit51)

# Plot the volatility
plot(garchvol51)
garchvol_ext51 = garchvol51[-1775]

#fit garch for pushing hypothesis
###################################################################################
#norm
model.spec_ex51 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext51), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "norm")

fit_ex51 =ugarchfit(spec=model.spec_ex51,data = lagged_pos_Sent_mva)
fit_ex51.1 =ugarchfit(spec=model.spec_ex51,data = lagged_neg_Sent_mva)
fit_ex51
fit_ex51.1
##################################################################################
#std
model.spec_ex52 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext51), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "std")

fit_ex52 =ugarchfit(spec=model.spec_ex52,data = lagged_pos_Sent_mva)
fit_ex52.1 =ugarchfit(spec=model.spec_ex52,data = lagged_neg_Sent_mva)
fit_ex52
fit_ex52.1
###################################################################################
#ged
model.spec_ex53 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext51), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "ged")

fit_ex53 =ugarchfit(spec=model.spec_ex53,data = lagged_pos_Sent_mva)
fit_ex53.1 =ugarchfit(spec=model.spec_ex53,data = lagged_neg_Sent_mva)
fit_ex53
fit_ex53.1
###################################################################################
#snorm
model.spec_ex54 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext51), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "snorm")

fit_ex54 =ugarchfit(spec=model.spec_ex54,data = lagged_pos_Sent_mva)
fit_ex54.1 =ugarchfit(spec=model.spec_ex54,data = lagged_neg_Sent_mva)
fit_ex54
fit_ex54.1
###################################################################################
#sstd
model.spec_ex55 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext51), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sstd")

fit_ex55 =ugarchfit(spec=model.spec_ex55,data = lagged_pos_Sent_mva)
fit_ex55.1 =ugarchfit(spec=model.spec_ex55,data = lagged_neg_Sent_mva)
fit_ex55
fit_ex55.1
#################################################################################
#sged
model.spec_ex56 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext51), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sged")

fit_ex56 =ugarchfit(spec=model.spec_ex56,data = lagged_pos_Sent_mva)
fit_ex56.1 =ugarchfit(spec=model.spec_ex56,data = lagged_neg_Sent_mva)
fit_ex56
fit_ex56.1

##################################################################################
# Specify a standard GARCH model sged
garchspec61 <- ugarchspec(mean.model = list(armaOrder = c(0,1)),
                          variance.model = list(model = "eGARCH"), 
                          distribution.model = "sged")

# Estimate the model
garchfit61 <- ugarchfit(data = close_rets, spec = garchspec61)
garchfit61
# Use the method sigma to retrieve the estimated volatilities 
garchvol61 <- sigma(garchfit61)

# Plot the volatility
plot(garchvol61)
garchvol_ext61 = garchvol61[-1775]

#fit garch for pushing hypothesis
###################################################################################
#norm
model.spec_ex61 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext61), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "norm")

fit_ex61 =ugarchfit(spec=model.spec_ex61,data = lagged_pos_Sent_mva)
fit_ex61.1 =ugarchfit(spec=model.spec_ex61,data = lagged_neg_Sent_mva)
fit_ex61
fit_ex61.1
##################################################################################
#std
model.spec_ex62 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext61), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "std")

fit_ex62 =ugarchfit(spec=model.spec_ex62,data = lagged_pos_Sent_mva)
fit_ex62.1 =ugarchfit(spec=model.spec_ex62,data = lagged_neg_Sent_mva)
fit_ex62
fit_ex62.1
###################################################################################
#ged
model.spec_ex63 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext61), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "ged")

fit_ex63 =ugarchfit(spec=model.spec_ex63,data = lagged_pos_Sent_mva)
fit_ex63.1 =ugarchfit(spec=model.spec_ex63,data = lagged_neg_Sent_mva)
fit_ex63
fit_ex63.1
###################################################################################
#snorm
model.spec_ex64 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext61), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "snorm")

fit_ex64 =ugarchfit(spec=model.spec_ex64,data = lagged_pos_Sent_mva)
fit_ex64.1 =ugarchfit(spec=model.spec_ex64,data = lagged_neg_Sent_mva)
fit_ex64
fit_ex64.1
###################################################################################
#sstd
model.spec_ex65 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext61), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sstd")

fit_ex65 =ugarchfit(spec=model.spec_ex65,data = lagged_pos_Sent_mva)
fit_ex65.1 =ugarchfit(spec=model.spec_ex65,data = lagged_neg_Sent_mva)
fit_ex65
fit_ex65.1
#################################################################################
#sged
model.spec_ex66 = ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1,1), 
                        external.regressors = matrix(cbind(garchvol_ext61), ncol=1)), #adding sentiment as external regressor based on our assumption, that it influences our volatility
  mean.model = list(armaOrder = c(0,1)), distribution.model = "sged")

fit_ex66 =ugarchfit(spec=model.spec_ex66,data = lagged_pos_Sent_mva)
fit_ex66.1 =ugarchfit(spec=model.spec_ex66,data = lagged_neg_Sent_mva)
fit_ex66
fit_ex66.1
