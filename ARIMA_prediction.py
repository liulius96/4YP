#import packages
import pandas as pd
import numpy as np
import datetime
from pmdarima.arima import auto_arima

#to plot within notebook
import matplotlib.pyplot as plt
# %matplotlib inline

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# goodness of fit
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# define some functions to find the test statistics
def rmse(actual_val, predicted_val):
    return sqrt(mean_squared_error(actual_val,predicted_val))

def r_sq(actual_val, predicted_val):
    return r2_score(actual_val,predicted_val)

# This is a script to run pure linear regression (nothing extra) into the

# load data
btc_df = pd.read_csv('C:/Users/Liuli/OneDrive/Documents/4YP/crypto_sentiment_daily_df.csv')
btc_df = btc_df[['Date', 'BTC', 'BVOL24H', 'BTC_Volume', 'SPX']]
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df.set_index('Date', inplace=True)
btc_ld = btc_df.sort_index(ascending=True)
btc_ld = np.log(btc_df) - np.log(btc_df.shift(1))
btc_ld = btc_ld.dropna()

# initialise a list to store the model aic (hopefully this improves with time)
model_aic = []

# initialise a forecast list to compare with the test validation data later
predict = []

# define the test period and test values
test = btc_ld['BTC'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=31)]

# write the for loop which conducts a rolling window test
for day in range(1,len(test+1)):
    # set up the training data
    train = btc_ld['BTC'].loc[:datetime.date(year=2018, month=1,day=day)]

    # fit the model with the training data
    model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True)
    # print out the model AIC and append it to the model_aic list
    print(model.aic())
    model_aic.append((day, model.aic()))

    # fit the model
    model.fit(train)

    # forecast a one-step ahead prediction
    forecast = model.predict(n_periods=1)
    predict.append(forecast)

r2_ld = r_sq(test.values,predict)
rmse_ld = rmse(test.values,predict)

# Create a fontdict for the plots
font = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

plt.figure(1)
plt.plot(test.index,test.values, label='Bitcoin Price')
plt.plot(test.index,predict, label='Predicted Bitcoin Price')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price (Log Differenced)')
plt.legend()
plt.title('ARIMA rolling window Bitcoin Price Prediction', fontdict=font)


# Create an error plot (absolute error)
error = []
for x in range(0,len(test)):
    error = abs(test.BTC[x]-predict[x])

# Create an Error Plot
plt.figure(2)
plt.plot(test.index,error, label='Absolute Error')
plt.xlabel('Date')
plt.ylabel('Absolute Error')
plt.title('Absolute Error for ARIMA rolling window prediction', fontdict=font)
plt.show()


#plot

# plt.figure(figsize = (18,9))
# plt.plot(btc['Date'],btc['BTC'])
# plt.xticks(range(0,btc.shape[0],500),btc['Date'].loc[::500],rotation=45)
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Bitcoin Price',fontsize=18)
# plt.show()
#
# # or alternatively use the plot function within pandas
# fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)
#
# btc.plot(x='Date', y='BTC', ax=axes[0])
# btc.plot(x='Date', y='BVOL24H',ax=axes[1])
# btc.plot(x='Date', y='Daily_Abs_Return', ax=axes[2])
# btc.plot(x='Date', y='log_dif', ax=axes[3])
# btc.plot(x='Date',y='BTC', title='Bitcoin Price Chart')

# # split the data into two columns of train and test (5 years for training and 10 days for testing)
# train = btc['BTC'].loc[:datetime.date(year=2018, month=1,day=1)]
# test = btc['BTC'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]
#
# # set up daily absolute returns to train and test
# train_dr = btc['Daily_Abs_Return'].loc[:datetime.date(year=2018, month=1,day=1)]
# test_dr = btc['Daily_Abs_Return'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]
#
# # fit the model with the training data
# model = auto_arima(train, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
#
# # Check the AIC (Akaike information criterion) of the model (lower is better)
# print(model.aic())

# # same for log returns
# train_log_dif = btc['log_dif'].loc[:datetime.date(year=2018, month=1,day=1)]
# test_log_dif = btc['log_dif'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]
# # set up model
# model_log_dif = auto_arima(train_log_dif, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
# model_log_dif.fit(train_log_dif)
# # forecast
# forecast_log_dif = model_log_dif.predict(n_periods=len(test_log_dif))
# forecast_log_dif = pd.DataFrame(forecast_log_dif,index = test_log_dif.index,columns=['Prediction'])
# # Goodness of fit
# r2_log_dif = r_sq(btc['log_dif'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_log_dif)
# rmse_log_dif = rmse(btc['log_dif'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_log_dif)
# # simple plot comparison
# result_log_dif = pd.merge(btc.loc[datetime.date(year=2017,month=12,day=1):datetime.date(year=2018,month=1,day=10)], forecast_log_dif, how='outer', left_index=True, right_index=True)
# dx = result_log_dif.plot(x='Date', y=['log_dif', 'Prediction'], title='Bitcoin Price Logarithmic Difference Forecast')
# dx.set_ylabel("Logarithmic Difference in Price")
# # set up the text box
# textstr_log_dif = '\n'.join((
#     r'$RMSE=%.2f$' % (rmse_log_dif, ),
#     r'$R squared=%.2f$' % (r2_log_dif, )))
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# dx.text(0.05, 0.95, textstr_log_dif, transform=dx.transAxes, fontsize=12,
#         verticalalignment='top', bbox=props)