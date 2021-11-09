###################################################################################

############################# Master's Thesis #####################################
#Does sentiment from relevant news improve volatility forecast from GARCH models? -
#An application of natural language #processing

###################################################################################

import pandas as pd
import numpy as np

df=pd.read_csv("df.csv", sep=",")
df2=pd.read_csv("Full_Data.csv", sep=",")
df["Headline"] = df["Headline"].map('{}.'.format)#add a point to the end of each headline, thus the headlines match
df2 = df2.rename(columns={"sentence":"Headline"}) #rename the column sentence to merge the dataframes

df_merged = df.merge(df2, how='left') #merge the dataframes based matching headlines

df_merged = df_merged.drop(["Journalists", "Link", "Article", "logit"], axis =1) #drop columns of no interest
df_merged = df_merged.drop_duplicates()#drop duplicates --> delete timestamp and do it again!!
df_merged["Date"] = pd.to_datetime(df["Date"]).dt.date #format the date
df_merged = df_merged[df_merged["prediction"]!="neutral"] #delete neutrals
print(df_merged)
df_merged.to_csv("./Merge1.csv", sep=',', index=False) #create csv

df_Sent = df_merged.drop(["Headline", "Unnamed: 0"], axis = 1)
#df_Sent = df_Sent.groupby('Date')['sentiment_score'].mean()
df_Sent.to_csv("./Sent.csv", sep=',', index=False) #only keep sentiment and date

#get a dataframe with only positive or negative sentiment --> check for difference 

df_pol_Sent = df_merged

#split sentiment in positive and negative Sent
df_pol_Sent["pos_Sent"] = np.where(df_pol_Sent["prediction"] == "positive", df_pol_Sent["sentiment_score"], 0)
df_pol_Sent["neg_Sent"] = np.where(df_pol_Sent["prediction"] == "negative", df_pol_Sent["sentiment_score"], 0)
df_pol_Sent = df_pol_Sent.drop(["Headline", "prediction", "sentiment_score"], axis = 1)
df_pol_Sent = df_pol_Sent.groupby(["Date"])["pos_Sent", "neg_Sent"].mean().reset_index() #Warning? 
df_pol_Sent.to_csv("./Pol_Sent.csv", sep=',', index=False) #create csv

#S&P500 data data from https://www.investing.com/indices/us-spx-500-historical-data
df_S_P500=pd.read_csv("S&P 500 Historical Data.csv", sep=",")
df_S_P500 = df_S_P500.rename(columns={"Price":"close"})
df_S_P500 = df_S_P500.filter(["Date", "close"], axis =1)
df_S_P500["Date"] = pd.to_datetime(df_S_P500["Date"]).dt.date
df_S_P500['close'] = df_S_P500['close'].str.replace(',', '')

#sentiment score as one variable
df_Sent["Date"] = pd.to_datetime(df_Sent["Date"]).dt.date
df_Sent = df_Sent.groupby('Date')['sentiment_score'].mean().reset_index()
df_merged = pd.merge(df_Sent, df_S_P500, how='outer', on = 'Date')
df_merged = df_merged.dropna()
df_merged.to_csv("./Sent_SP500_1.csv", sep=',', index=False)

#Merge pol_Sent (pos and neg sentiment) with S&p 500 Data
df_merged1 = pd.merge(df_pol_Sent, df_S_P500, how = 'outer', on = 'Date')
df_merged1 = df_merged1.dropna()
#df_merged1["Date"] = pd.to_datetime(df_merged1["Date"]).dt.date
#df_merged1 = df_merged1[df_merged1['Date']<= pd.to_datetime('2011-03-03')] #just as long as i dont have run the sentiment analysis on the full dataset
df_merged1.to_csv('./merged_pol_Sent.csv', sep = ',', index = False)

#get the standard deviation of the pos and neg sentiment and only keep the ones, that exceed 2*std
std_pos = np.std(df_merged1["pos_Sent"])
std_neg = np.std(df_merged1["neg_Sent"])
mean_pos = np.mean(df_merged1["pos_Sent"])
mean_neg = np.mean(df_merged1["neg_Sent"])

df_std_pos = df_merged1[df_merged1["pos_Sent"]> (mean_pos+std_pos)]
df_std_pos = df_std_pos.drop(["neg_Sent", "close"], axis = 1)

df_std_neg = df_merged1[df_merged1["neg_Sent"] < (mean_neg - std_neg)]
df_std_neg = df_std_neg.drop(["pos_Sent", "close"], axis = 1)
print(df_std_neg)

#merge the sentiment scores that exceed 2 stds from the mean to the s&p500 data
df_merged2 = pd.merge(df_std_pos, df_S_P500, how = "outer", on = "Date")
df_merged2 = pd.merge(df_std_neg, df_merged2,  how = "outer", on = "Date")
df_merged2["pos_Sent"] = df_merged2["pos_Sent"].fillna(0) #because of problems of inverting the hessian for the garch modells
df_merged2["neg_Sent"] = df_merged2["neg_Sent"].fillna(0)
df_merged2 = df_merged2.drop(["close"], axis = 1)
df_merged2 = df_merged2.sort_values("Date")
df_merged2 = df_merged2.dropna()
print(df_merged2)
df_merged2.to_csv('./std_Sent.csv', sep = ',', index = False)

#moving average to get rid off some noise (rolling window = 5)
df_mva = df_merged1
df_mva["pos_Sent_mva"] = df_mva["pos_Sent"].rolling(5).mean()
df_mva["neg_Sent_mva"] = df_mva["neg_Sent"].rolling(5).mean()
print(df_mva)
df_mva.to_csv('./Sent_mva.csv', sep = ',', index = False)

df_mva_std = pd.merge(df_merged2, df_mva, how = "outer", on = "Date")
df_mva_std = df_mva_std.dropna(subset=["close"])
print(df_mva_std)
df_mva_std.to_csv('./mva_std_Sent.csv', sep = ',', index = False)