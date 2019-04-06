import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.ensemble.forest import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

import math
from dateutil.relativedelta import relativedelta
from datetime import datetime, date
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')

# load data
btc_df = pd.read_csv('C:/Users/Liuli/OneDrive/Documents/4YP/crypto_sentiment_daily_df.csv')
btc_df = btc_df[['Date', 'BTC', 'BVOL24H', 'BTC_Volume', 'SPX']]
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df.set_index('Date', inplace=True)
btc_ld = btc_df.sort_index(ascending=True)
btc_ld = np.log(btc_df) - np.log(btc_df.shift(1))
btc_ld = btc_ld.dropna()

#  split the dataframe into train and test
train = btc_ld.loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2017,month=12,day=31)]
test = btc_ld.loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=31)]

# split into input and output?
trainX = np.asarray(train.drop(columns='BTC'))
trainY = np.asarray(train.BTC)

testX = np.asarray(test.drop(columns='BTC'))
testY = np.asarray(test.BTC)

# Define the RF model
RF_Model = RandomForestRegressor(n_estimators=100,
                                 max_features=1, oob_score=True)
# Fit the model
rf_fitted = RF_Model.fit(trainX,trainY)

# predict the trained data
trainY_predict = rf_fitted.predict(trainX)
trainY_predict = trainY_predict.reshape(-1,1) #reshape (transpose)
# Plot the predicted training data
train_plot_df = pd.DataFrame(trainY_predict, columns='Predicted BTC')
train_plot_df = train_plot_df.set_index(train.index)
train_plot_df['BTC'] = train.BTC
# test
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
r2_train = r2_score(train.BTC,trainY_predict)
mse_train = mean_squared_error(train.BTC,trainY_predict)


# revert the price prediction
btc_price = btc_df.BTC.loc[datetime.date(year=2013,month=12,day=31):datetime.date(year=2017,month=12,day=30)]
btc_price_pred =np.asarray(btc_price).reshape(-1,1)*np.exp(trainY_predict)
train_btc_plot_df = pd.DataFrame(btc_price_pred,columns=['Predicted BTC'])
train_btc_plot_df= train_btc_plot_df.set_index(train.index)
train_btc_plot_df['BTC'] = btc_df.BTC
train_btc_plot_df.plot()
# statistics
r2_train_reverted = r2_score(train_btc_plot_df.BTC,btc_price_pred)
mse_train_reverted = mean_squared_error(train_btc_plot_df.BTC,btc_price_pred)
rmse_train_reverted = sqrt(mse_train_reverted)
print('\nR squared (Trained data reverted): '+ str(r2_train_reverted))
print('\nRMSE (Trained data reverted): ' +str(rmse_train_reverted))

#
# TIME FOR THE TEST
#

# predict the log diff price with the test inputs X
testY_predict = rf_fitted.predict(testX)
testY_predict = testY_predict.reshape(-1,1) #reshape (transpose)
# plotting stuffs
testY_plot_df = pd.DataFrame(testY_predict, columns=['BTC_log_diff_pred'])
testY_plot_df = testY_plot_df.set_index(test.index)
testY_plot_df['BTC_log_diff'] = test.BTC
# statistics
r2_test = r2_score(test.BTC,testY_predict)
print('R squared (test data, log diff): '+str(r2_test))

# revert the predictions
btc_price_test = btc_df.BTC.loc[datetime.date(year=2017,month=12,day=31):datetime.date(year=2018,month=1,day=30)]
btc_price_test_pred = np.asarray(btc_price_test).reshape(-1,1)*np.exp(testY_predict)
test_btc_plot_df = pd.DataFrame(btc_price_test_pred,columns=['Predicted BTC'])
test_btc_plot_df = test_btc_plot_df.set_index(test.index)
test_btc_plot_df['BTC'] = btc_df.BTC
test_btc_plot_df.plot()
# statistics
r2_test_reverted = r2_score(test_btc_plot_df.BTC, test_btc_plot_df['Predicted BTC'])
print('\nR squared (test data reverted): ' +str(r2_test_reverted))
rmse_test_reverted = sqrt(mean_squared_error(test_btc_plot_df.BTC, test_btc_plot_df['Predicted BTC']))
print('\nRMSE (test data reverted): '+str(rmse_test_reverted))
