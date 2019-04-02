import numpy as np
import pandas as pd
import datetime
import feather

df = pd.read_pickle('C:/Users/tliu/Documents/4YP/Outputs/Pickles/merged_df_slda.pkl')
df['daily_ret'] = df['BTC'].pct_change(1)
df = df.truncate(before=datetime.date(year=2013,month=1,day=1))

# find 5% of the mean of the log returns
log_ret = np.asarray(df.log_ret)
log_ret = np.absolute(log_ret)

mean = log_ret.mean()
five_pct_mean_log_ret = 0.05*mean

# find 5% of the mean of the daily returns
daily_ret = np.asarray(df.daily_ret)
daily_ret = np.absolute(daily_ret)

mean = daily_ret.mean()
five_pct_mean_daily_ret = 0.05*mean

# cut the dataframe for log returns
bins_log_ret = [-np.inf, -five_pct_mean_log_ret, five_pct_mean_log_ret, np.inf]
names = [-1, 0, 1]
df['log_ret_categories'] = pd.cut(df.log_ret, bins_log_ret, labels=names)

# cut the dataframe for daily returns
bins_daily_ret = [-np.inf, -five_pct_mean_daily_ret, five_pct_mean_daily_ret, np.inf]
df['daily_ret_categories'] = pd.cut(df.daily_ret, bins_daily_ret, labels=names)

feather.write_dataframe(df, 'C:/Users/tliu/Documents/4YP/sLDA/data/slda_categorical_data_2013.feather')