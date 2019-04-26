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
rcParams['figure.figsize'] = 15,8

# #for normalizing data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))

# goodness of fit
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# define some functions to find the test statistics
def rmse(actual_val, predicted_val):
    return sqrt(mean_squared_error(actual_val,predicted_val))

def r_sq(actual_val, predicted_val):
    return r2_score(actual_val,predicted_val)


# load data
btc_df = pd.read_csv('C:/Users/tliu/Documents/4YP/crypto_sentiment_daily_df.csv')
btc_df = btc_df[['Date', 'BTC', 'BVOL24H', 'BTC_Volume', 'SPX']]
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
btc_df.set_index('Date', inplace=True)
btc_df = btc_df.sort_index(ascending=True)
btc_df['Daily_Returns'] = btc_df.BTC.pct_change(1)

# Load the dataframe
btc_dr = pd.read_pickle('C:/Users/tliu/Documents/4YP/Outputs/dataframes_14-18/btc_dr_with_mlt_02threshold_alpha-1_topics-5_complete.pkl')
btc_dr = btc_dr.fillna(0)

# initialise a list to store the model aic (hopefully this improves with time)
model_aic = []

# initialise a list to store the model order
model_order = []

# initialise a forecast list to compare with the test validation data later
predict = []

# initialise a list to store the confidence interval
conf_interval =[]

# define the test period and test values
test = btc_df['Daily_Returns'].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=31)]
test_exog = btc_dr[4].loc[datetime.date(year=2018, month=1,day=1):datetime.date(year=2018, month=1,day=31)]

# write the for loop which conducts a rolling window test
for day in range(1,len(test)+1):
    # set up the training data
    train = btc_df['Daily_Returns'].loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2018, month=1,day=day)]
    train_exog = btc_dr[4].loc[datetime.date(year=2014,month=1,day=1):datetime.date(year=2018, month=1,day=day)]

    # merge the two dataframes because of variable length
    train = pd.DataFrame(train)
    train_exog = train.merge(train_exog, how='outer', left_index=True, right_index=True)
    # reshape into a shape amenable to auto arima
    train_ex = np.asarray(train_exog[4].fillna(0)).reshape(-1,1)
    # fit the model with the training data
    model = auto_arima(train,exogenous=train_ex, start_p=1, start_q=1, max_p=6, max_q=5, m=7, max_order=20, start_P=0, seasonal=True, d=1, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True)
    # print out the model AIC and append it to the model_aic list
    print(model.aic())
    model_aic.append( model.aic())

    # print out the model order and append it to the model order list
    print(model.order)
    model_order.append(model.order)
    # fit the model
    model.fit(train)

    # forecast a one-step ahead prediction
    forecast,conf = model.predict(n_periods=1, return_conf_int=True)
    predict.append(forecast[0])
    conf_interval.append(conf)
    print(day)

r2_ld = r_sq(test.values,predict)
mape = np.mean(np.abs(predict - test.values) / np.abs(test.values))  # MAPE
mae = mean_absolute_error(test.values,predict)
print('r2 :'+str(r2_ld)+'\nmape :'+str(mape)+'\nmae :'+str(mae))

# Create a fontdict for the plots
font = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

plt.figure(1)
plt.fill_between(test.index,y1=[x[0] for x in conf_interval], y2=[y[1] for y in conf_interval])
plt.plot(test.index,test.values, label='Bitcoin Price')
plt.plot(test.index, predict, label='Predicted Bitcoin Daily Returns')
plt.xlabel('Date')
plt.ylabel('Bitcoin Daily Returns')
# plt.fill_between(test.index,y1=[x[0] for x in conf_interval], y2=[y[1] for y in conf_interval])
plt.legend()
plt.title('ARIMA rolling window Bitcoin Daily Returns Prediction', fontdict=font)


# Create an error plot (absolute error)
error = []
for x in range(0,len(test)):
    error.append(abs(test[x]-predict[x]))

# Create an Error Plot
plt.figure(2)
plt.plot(test.index,error, label='Absolute Error')
plt.xlabel('Date')
plt.ylabel('Absolute Error')
plt.title('Absolute Error for ARIMA rolling window prediction', fontdict=font)
plt.show()

# Create Jan's error plot (lolololol)
#
# First decide on a threshold for determining whether today's trade matters or not
# Below we have chosen eta to be 1% - trades only matter if the underlying move by more than 1%
eta = 0.005

# Turn the data into categorical data - -1 for negative, 0 for neutral , +1 for positive
# create the bins (the slices) and the names for the categories
bins = [-np.inf, -eta, eta, np.inf]
names = [-1, 0, +1]
# Now change test values into categorical
test_cat = pd.cut(test, bins, labels=names)

# Now change the prediction values into categorical
predict_cat = pd.cut(pd.Series(predict), bins, labels=names)

# import some metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

def classification_stats(y_true, y_pred):
    acc = accuracy_score(y_true,y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true,y_pred,average='weighted')
    print('Accuracy :'+str(acc))
    print('Balanced Accuracy :'+str(bal_acc))
    print('F1 Score :'+str(f1))
    return acc, bal_acc, f1

print('Confusion Matrix')
print(confusion_matrix(test_cat,predict_cat))
print('Precision, Recall, F1, Support')
print(precision_recall_fscore_support(test_cat,predict_cat,average='weighted'))

acc, ba, f1 = classification_stats(test_cat,predict_cat)

# new error - correct (1) when a true positive or true negative is given
# incorrect (0) otherwise
new_err = []
for x in range(0,len(predict_cat)):
    # run through all four examples
    if predict_cat[x] == test_cat[x]:
        new_err.append(0)
    else:
        new_err.append(1)

# plt.figure(3)
fig, [ax1, ax2] = plt.subplots(2,1)
# ax1 = plt.plot(test.index, test_cat, label = 'True Daily Returns')
# ax2 = plt.plot(test.index, predict_cat, label ='Predicted Daily Returns')
# ax2 = plt.plot(test.index, new_err, label='Error')

index = np.arange(len(test))
bar_width = 0.3

opacity = 0.6
rec1 = ax1.bar(index, test_cat,bar_width,alpha=opacity, color='b', label='Daily Returns')
rec2 = ax1.bar(index+bar_width, predict_cat, bar_width, alpha=opacity, color='g', label='Predicted Returns')
ax1.set_xlabel('Day')
ax1.set_ylabel('Categorical Daily Returns')
ax1.set_title('Comparison Between Predicted Categorical Returns and Categorical Returns')
ax1.set_xticks(index + bar_width / 2)
x_tick_labels = list(range(len(test)))
x_tick_labels = [x+1 for x in x_tick_labels]
ax1.set_xticklabels(x_tick_labels)
ax1.legend()

rec3 = ax2.bar(index,new_err,2*bar_width,alpha=0.6, color='r', label='Error for each day')
ax2.set_xlabel('Day')
ax2.set_ylabel('Error Plot (Prediction != Test)')
ax2.set_title('Error')
ax2.set_xticks(index + bar_width / 2)
ax2.set_xticklabels(x_tick_labels)
ax2.legend()
fig.tight_layout()
plt.show()

# Check out the model statistics

# plot the model AIC
plt.figure(4)
plt.plot(test.index, model_aic,label='model AIC')
plt.legend()
plt.title('Model AIC over time', fontdict=font)
plt.xlabel('Day')
plt.ylabel('Model AIC score')
plt.show()

# plot the model statistics over time

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(test_cat,predict_cat,names,True,' (eta:'+str(eta)+')')