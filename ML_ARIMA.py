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

# load the dataframes
# df = pd.read_pickle('C:/Users/tliu/Documents/4YP/Outputs/Pickles/Initial_DF_all.pkl')
# btc = pd.read_csv('C:/Users/tliu/Documents/4YP/Outputs/Pickles/btc2013.pkl')
btc = pd.read_csv('C:/Users/tliu/Documents/4YP/crypto_sentiment_daily_df.csv')
btc['Daily_Abs_Return'] = btc['BTC'].pct_change(1)
btc['log_dif'] = np.log(btc.BTC).diff().dropna()

# set the date as the index
btc['Date'] =pd.to_datetime(btc['Date'])
btc.index = btc.Date
btc =btc.truncate(before=datetime.date(year=2013,month=1,day=1))

pd.to_pickle(btc,'C:/Users/tliu/Documents/4YP/2013_btc_data.pkl')

#plot
import matplotlib.pyplot as plt
plt.figure(figsize = (18,9))
plt.plot(btc['Date'],btc['BTC'])
plt.xticks(range(0,btc.shape[0],500),btc['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Bitcoin Price',fontsize=18)
plt.show()

# or alternatively use the plot function within pandas
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)

btc.plot(x='Date', y='BTC', ax=axes[0])
btc.plot(x='Date', y='BVOL24H',ax=axes[1])
btc.plot(x='Date', y='Daily_Abs_Return', ax=axes[2])
btc.plot(x='Date', y='log_dif', ax=axes[3])
# btc.plot(x='Date',y='BTC', title='Bitcoin Price Chart')

# split the data into two columns of train and test (5 years for training and 10 days for testing)
train = btc['BTC'].loc[:datetime.date(year=2018, month=1,day=1)]
test = btc['BTC'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]

# set up daily absolute returns to train and test
train_dr = btc['Daily_Abs_Return'].loc[:datetime.date(year=2018, month=1,day=1)]
test_dr = btc['Daily_Abs_Return'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]

# fit the model with the training data
model = auto_arima(train, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)

# Check the AIC (Akaike information criterion) of the model (lower is better)
print(model.aic())

# actually fit the data
model.fit(train)

# forecast some predictions
forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
# Goodness of fit
r2_price = r_sq(btc['BTC'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast)
rmse_price = rmse(btc['BTC'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast)
# Simple plot comparison
result= pd.merge(btc.loc[datetime.date(year=2017,month=12,day=1):datetime.date(year=2018,month=1,day=10)], forecast, how='outer', left_index=True, right_index=True)
ax = result.plot(x='Date', y=['BTC', 'Prediction'], title='Bitcoin Price Forecast')
ax.set_ylabel("Price (USD)")
# set up text box
textstr = '\n'.join((
    r'$RMSE=%.2f$' % (rmse_price, ),
    r'$R squared=%.2f$' % (r2_price, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05,0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


# repeat for the daily returns
model_dr =  auto_arima(train_dr, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model_dr.fit(train_dr)
# forecast some predictions
forecast_dr = model_dr.predict(n_periods=len(test_dr))
forecast_dr = pd.DataFrame(forecast_dr,index = test_dr.index,columns=['Prediction'])
# Goodness of fit
r2_dr = r_sq(btc['Daily_Abs_Return'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_dr)
rmse_dr = rmse(btc['Daily_Abs_Return'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_dr)
# simple plot comparison
result_dr = pd.merge(btc.loc[datetime.date(year=2017,month=12,day=1):datetime.date(year=2018,month=1,day=10)], forecast_dr, how='outer', left_index=True, right_index=True)
bx = result_dr.plot(x='Date', y=['Daily_Abs_Return', 'Prediction'], title='Bitcoin Daily Returns Forecast')
bx.set_ylabel("Daily returns")
# set up the text box
textstr_dr = '\n'.join((
    r'$RMSE=%.2f$' % (rmse_dr, ),
    r'$R squared=%.2f$' % (r2_dr, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
bx.text(0.05, 0.95, textstr_dr, transform=bx.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


# now the same for volatility
train_vol = btc['BVOL24H'].loc[:datetime.date(year=2018, month=1,day=1)]
test_vol = btc['BVOL24H'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]
# set up model
model_vol = auto_arima(train_vol, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model_vol.fit(train_vol)
# forecast
forecast_vol = model_vol.predict(n_periods=len(test_vol))
forecast_vol = pd.DataFrame(forecast_vol,index = test_vol.index,columns=['Prediction'])
# Goodness of fit
r2_vol = r_sq(btc['BVOL24H'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_vol)
rmse_vol = rmse(btc['BVOL24H'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_vol)
# simple plot comparison
result_vol = pd.merge(btc.loc[datetime.date(year=2017,month=12,day=1):datetime.date(year=2018,month=1,day=10)], forecast_vol, how='outer', left_index=True, right_index=True)
cx = result_vol.plot(x='Date', y=['BVOL24H', 'Prediction'], title='Bitcoin 24 Hour Volatility Forecast')
cx.set_ylabel("24 Hour Volatility")
# set up the text box
textstr_vol = '\n'.join((
    r'$RMSE=%.2f$' % (rmse_vol, ),
    r'$R squared=%.2f$' % (r2_vol, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
cx.text(0.05, 0.95, textstr_vol, transform=cx.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)


# same for log returns
train_log_dif = btc['log_dif'].loc[:datetime.date(year=2018, month=1,day=1)]
test_log_dif = btc['log_dif'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=10)]
# set up model
model_log_dif = auto_arima(train_log_dif, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model_log_dif.fit(train_log_dif)
# forecast
forecast_log_dif = model_log_dif.predict(n_periods=len(test_log_dif))
forecast_log_dif = pd.DataFrame(forecast_log_dif,index = test_log_dif.index,columns=['Prediction'])
# Goodness of fit
r2_log_dif = r_sq(btc['log_dif'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_log_dif)
rmse_log_dif = rmse(btc['log_dif'].loc[datetime.date(year=2018,month=1,day=1):datetime.date(year=2018,month=1,day=10)], forecast_log_dif)
# simple plot comparison
result_log_dif = pd.merge(btc.loc[datetime.date(year=2017,month=12,day=1):datetime.date(year=2018,month=1,day=10)], forecast_log_dif, how='outer', left_index=True, right_index=True)
dx = result_log_dif.plot(x='Date', y=['log_dif', 'Prediction'], title='Bitcoin Price Logarithmic Difference Forecast')
dx.set_ylabel("Logarithmic Difference in Price")
# set up the text box
textstr_log_dif = '\n'.join((
    r'$RMSE=%.2f$' % (rmse_log_dif, ),
    r'$R squared=%.2f$' % (r2_log_dif, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
dx.text(0.05, 0.95, textstr_log_dif, transform=dx.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)