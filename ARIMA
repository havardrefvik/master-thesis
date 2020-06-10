import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import math
import random
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from numpy import log
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
"""df = pd.read_csv("results_time_series_voltage11kW.csv", names=["value0", "value1", "value2", "value3", "value4", \
                                                               "value_9", "value6", "value7", "value8", "value9"])"""
#df = pd.read_csv("voltage_1000_ID_9.csv", sep=";", parse_dates=["Timestamp"], index_col=["Timestamp"])

#df = pd.read_csv("results_one_year.csv", names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

df = pd.read_csv("new_results_voltage.csv", names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                   "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"])

#df["1].plot()
#plt.show()

series = df["1"][:1000]

#REMOVING RANDOM DATA POINTS
#Setting the basic parameters
original_length = len(series)
print('Original length: ' + str(original_length))
percentage_to_remove = 50
counter = 0
probability_of_point_removal = 0.2
probability_of_propagation = 0
number_of_points_to_remove = original_length * (percentage_to_remove/100)
number_of_points_to_remove = math.floor(number_of_points_to_remove)
print('Number of points to remove: ' + str(number_of_points_to_remove))

def decision(probability):
    return random.random() < probability

def remove_points(dataframe):
    global counter
    for i in range(len(dataframe)):
        #while counter < number_of_points_to_remove:
        if decision(probability_of_point_removal):
            # Set lost values to 0
            # dataframe.at[i] = 0
            # Set lost values to the previous value
            dataframe.at[i] = dataframe.iloc[i - 1]
            #print(df.loc[i])
            #print(counter)
            counter += 1
            i += 1
        elif i == len(dataframe):
            break
        else:
            i += 1
#remove_points(df["1])

def remove_points_with_propagation(dataframe):
    global counter
    for i in range(len(dataframe)):
        while counter < number_of_points_to_remove:
            if decision(probability_of_point_removal):
                # Set lost values to the previous value
                dataframe.at[i] = dataframe.iloc[i - 1]
                counter += 1
                i += 1
                prop_counter = i + 1
                if decision(probability_of_propagation):
                    while counter < number_of_points_to_remove and decision(probability_of_propagation):
                        #df.at[prop_counter] = 0
                        # Set lost values to the previous value
                        dataframe.at[prop_counter] = dataframe.iloc[prop_counter - 1]
                        counter += 1
                        i = prop_counter + 1
                        prop_counter += 1
                        continue
            elif i == len(dataframe):
                break
            else:
                i += 1
#remove_points_with_propagation(df["9"][:1000])

print('Counter: ' + str(counter))
print('Number of rows with 0 as value: ' + str((df["9"]== 0).astype(int).sum(axis=0)))
series.plot()
plt.show()

#Creating model
model = ARIMA(series, order=(1,0,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


model_fit.plot_predict(dynamic=False)
plt.xlim(["2018-10-08","2018-10-12"])
plt.ylabel("Voltage")
plt.show()

train = series[:850]
test = series[850:874]
# Build Model
model = ARIMA(train, order=(1, 0, 3))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(24, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

#Inne på noe her, må bruke disse til å forecaste 1 frem i tid
print(fc_series.head(1))
print(test.values[0])


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train[800:850], label='training', color="blue")
plt.plot(test[:24], label='actual', color="orange")
plt.plot(fc_series[:24], label='forecast', color="green")
#plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.ylabel("Voltage")
plt.legend(loc='upper left', fontsize=8)
plt.show()

diff = (fc_series.head(2) - test.values[1])/test.values[1]
diff_real = (test.values[8] - test.values[7])/test.values[7]
print("DIFF IS " + str(diff))
print("REAL DIFF IS " + str(diff_real))
print("MEAN OF THE TRAINING SERIES IS " + str(train.mean()))
print("FORECAST SERIES IS " + str(fc_series))

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))*100  # MAPE
    #mape = np.mean(np.abs((fc_series.head(1))-test.values[0])/np.abs(test.values[0]))
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    #rmse = np.mean((fc_series.head(1))-test.values[0])/np.abs(test.values[0])
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc-test)[1]                      # ACF1

    print({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax})
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

print(fc)
print(test.values)

val_num = 0
correct_counter = 0
not_correct_counter = 0
for i in range(0,24):
    #print(val_num)
    diff = (fc[val_num] - fc[val_num-1])/fc[val_num-1]
    #print("DIFF " + str(val_num - 700) + " FOR FORECAST " + str(diff))

    diff_real = (test.values[val_num] - test.values[val_num-1])/test.values[val_num-1]
    #print("DIFF REAL " + str(val_num - 700) + " FOR FORECAST " + str(diff_real))
    val_num = val_num+1

    if (diff > 0 and diff_real > 0) or (diff < 0 and diff_real < 0):
        correct_counter += 1
    else:
        not_correct_counter += 1

print("Correct " + str(correct_counter))
print("Not correct " + str(not_correct_counter))
