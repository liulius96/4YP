import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# load dataset
btc_df = pd.read_csv('C:/Users/Liuli/OneDrive/Documents/4YP/crypto_sentiment_daily_df.csv')

# Turn the date into datetime object, set it as index and sort the order. Also rename
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df.set_index('Date', inplace=True)
btc_df = btc_df.sort_index(ascending=True)
btc_df = btc_df[['BTC']]
btc_df['log_ret'] = np.log(btc_df.BTC) - np.log(btc_df.BTC.shift(1))
btc_df['daily_returns']=btc_df.BTC.pct_change(1)


# create a dataframe which will help us sort the data
df = pd.read_pickle('D:/4YP and related Project (SD Card)/Data/most_likely_topic_dataframe_5_topics.pkl')

# sort the information
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Date'] = [datetime.datetime.date(d) for d in df['DateTime']]  # extracting date from timestamp
df['Time'] = [datetime.datetime.time(d) for d in df['DateTime']]  # extracting time from timestamp

# rename the column so it can be easily accessed
df.rename(columns={'Most Likely Topic': 'MLT'}, inplace=True)

# groupby date
s = df.groupby('Date').MLT.value_counts()

# sort the index
s = s.reset_index(name='Count')

# pivot the values so we have only 1 entry for the dates
s = s.pivot(index='Date', columns='MLT', values='Count')

# set the index
s.index = pd.to_datetime(s.index)

# create a btc dataframe containing the raw topic posts
merged_df = btc_df.merge(s,how='outer',left_index=True,right_index=True)

# fillna to prevent any errors
merged_df  = merged_df.fillna(0)

#
# create a new dataframe from 2014 till the end
btc2 = merged_df.loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2017,month=12,day=31)].fillna(0)

# find the log difference of the columns
for col in range(0,5):
    s.iloc[:,col] = np.log(s.iloc[:,col]) - np.log(s.iloc[:,col].shift(1))

# merge the two dataframes together
merged_df_logdiff = btc_df.merge(s,how='outer',left_index=True,right_index=True)

# fillna to prevent any errors
# merged_df_logdiff = np.log(merged_df).diff(1).dropna()

#
# AMAZING ALTERNATIVE FROM STATSMODELS
# data = np.log(mdata).diff().dropna()

#
# create a new dataframe from 2014 till 2018  - this df contains the log dif topics
btc_log_diff = merged_df_logdiff.loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2017,month=12,day=31)]
btc_log_diff = btc_log_diff.fillna(0)

# check for stationarity??? the Johansen test is similar to the ADF but for multivariate time series
from statsmodels.tsa.vector_ar.vecm import coint_johansen
print('The Johansen Test for multivariate series - Bitcoin with log differenced topics')
print(coint_johansen(btc_log_diff, 1,1).eig)

print('The Johansen Test for multivariate series - bitcoin with raw topics')
print(coint_johansen(btc2, 1,1).eig)


# train validation split- split the data into 80:20 for training and validation
# train
train2 = btc2[:int(0.8*len(btc2))]
train_logdiff = btc_log_diff[:int(0.8*len(btc_log_diff))]

# validate
valid2 = btc2[int(0.8*len(btc2)):]
valid_logdiff = btc_log_diff[int(0.8*len(btc_log_diff)):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR
model2 = VAR(endog=train2)
model_logdiff = VAR(endog=train_logdiff)

# fit the models
model2_fit = model2.fit()
model_logdiff_fit = model_logdiff.fit()

# predictions
prediction2 = model2_fit.forecast(model2_fit.y, steps=len(valid2))
prediction_logdiff = model_logdiff_fit.forecast(model_logdiff_fit.y, steps=len(valid_logdiff))

from sklearn.metrics import mean_squared_error
from math import sqrt
# #converting predictions to dataframe
# pred = pd.DataFrame(index=range(0,len(prediction2)),columns=[btc2.columns])
# for j in range(0,len(btc2.columns)):
#     for i in range(0, len(prediction2)):
#        pred.iloc[i][j] = prediction2[i][j]
#
# #check rmse
# for i in btc2.columns:
#     print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid2[i])))

pred2 = pd.DataFrame(prediction2,columns=[btc2.columns], index=valid2.index)
pred_log_diff = pd.DataFrame(prediction_logdiff,columns=[btc_log_diff.columns], index=valid_logdiff.index)
pred2['reverse_log_ret'] = pred2.log_ret


# plot
f, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(valid2.daily_returns, label ='Actual Daily Returns')
ax1.plot(pred2.daily_returns, color='green', label='Predicted Daily Returns with VAR')
ax1.plot(pred_log_diff.daily_returns, color='orange', label='Predicted Daily Returns with normalised VAR')
ax1.legend(loc='upper left')


ax2.plot(valid_logdiff.log_ret, label='Actual log difference')
ax2.plot(pred2.log_ret, color='green', label='Predicted log difference with VAR')
ax2.plot(pred_log_diff.log_ret, color='orange', label='Predicted log difference with noramlised VAR')
ax2.legend(loc='upper left')

plt.show(f)