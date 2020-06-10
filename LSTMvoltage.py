import time

start_time = time.time()
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math
from math import sqrt
import random
from matplotlib import pyplot
from numpy import array
import glob
import numpy as np
import csv

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    actual_list = []
    predicted_list = []
    rmse_list = []
    mape_list = []
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)

        #print("Actual " + str(actual))
        #print("Predicted" + str(predicted))


        actual_list.append(actual[0])
        predicted_list.append(predicted[0])
        rmse_list.append(rmse)
        mape_list.append(mape)


        #print('Test MAPE: %.3f' % mape)
        print("Predicted value: " + str(predicted[0]))
        print("Actual value: " + str(actual[0]))

    mape = mean_absolute_percentage_error(actual_list, predicted_list)
    print('Test MAPE: %.3f' % mape)
    print('t+%d RMSE: %f' % ((i + 1), rmse))

    """f = open('results.csv', 'a')

    with f:
        writer = csv.writer(f)
        writer.writerow(actual[0])
        writer.writerow(predicted[0])
        writer.writerow([mape])
        writer.writerow([rmse])"""

    """f = open('results.csv', 'w')

    with f:
        writer = csv.writer(f)
        writer.writerows(list_to_csv)"""

    """with f:
        write = csv.writer(f)"""

    """f = open('results.csv', 'a')

    with f:
        writer = csv.writer(f)
        writer.writerow(actual[0])

    with f:
        write = csv.writer(f)"""

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    #pyplot.plot(series.values)
    ax = df["9"][800:874].plot(label='training', figsize=(14, 4))
    #pyplot.plot(series[800:850].values, color="blue")
    pyplot.plot(series[850:870].values, label="actual", color="orange")

    # plot the forecasts
    for i in range(1):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        pyplot.plot(xaxis, yaxis, color='green', label="forecast")

    # show the plot
    pyplot.legend()
    pyplot.show()

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# load dataset
"""df = pd.read_csv("results_time_series_voltage11kW.csv", names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
"""

#df = pd.read_csv("voltage_5000.csv", sep=";")

df = pd.read_csv("new_results_voltage.csv", names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                                                  "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"])

starting_point = 850
increment = 0
for i in range(1):

    starting_point = starting_point + increment

    series = df["9"][:starting_point+25]
    print(series.mean())


    #REMOVING RANDOM DATA POINTS
    original_length = len(series)
    print('Original length: ' + str(original_length))
    percentage_to_remove = 100
    counter = 0
    probability_of_point_removal = 0.2
    probability_of_propagation = 0.2
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
                #Set lost values to 0
                #dataframe.at[i] = 0
                #Set lost values to the previous value
                dataframe.at[i] = dataframe.iloc[i-1]
                #print(df.loc[i])
                #print(counter)
                counter += 1
                i += 1
            elif i == len(dataframe):
                break
            else:
                i += 1
    #series.plot(color="red")
    #pyplot.show()
    #remove_points(df)

    def remove_points_with_propagation(dataframe):
        global counter
        for i in range(len(dataframe)):
            #while counter < number_of_points_to_remove:
            if decision(probability_of_point_removal):
                df.at[i] = 0
                counter += 1
                i += 1
                prop_counter = i + 1
                if decision(probability_of_propagation):
                    #while counter < number_of_points_to_remove and decision(probability_of_propagation):
                    df.at[prop_counter] = 0
                    counter += 1
                    i = prop_counter + 1
                    prop_counter += 1
                    continue
            elif i == len(dataframe):
                break
            else:
                i += 1
    #remove_points_with_propagation(series)

    print('Counter: ' + str(counter))
    print('Number of rows with 0 as value: ' + str((df == 0).astype(int).sum(axis=0)))
    series.plot()
    pyplot.show()

    #print(series)

    # configure
    n_lag = 24
    n_seq = 24
    n_test = 24
    n_epochs = 400
    n_batch = 4
    n_neurons = 1
    # prepare data
    scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
    print(test)
    print('Train: %s, Test: %s' % (train.shape, test.shape))
    # fit model
    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    # make forecasts
    forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
    # inverse transform forecasts and test
    forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(series, actual, scaler, n_test+2)
    # evaluate forecasts
    evaluate_forecasts(actual, forecasts, n_lag, n_seq)
    # plot forecasts
    plot_forecasts(series, forecasts, n_test+2)



    print(forecasts)

    """ax = series[780:800].plot(label='observed', figsize=(14, 4))
    for i in range(len(forecasts)):
        forecasts[i].plot(ax=ax, label='Forecast')
    pyplot.show()"""

print("My program took", time.time() - start_time, " seconds to run")
