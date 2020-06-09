import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import math
import random
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from numpy import log
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import sklearn
from math import sqrt
import csv


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

df = pd.read_csv("new_results_voltage.csv", names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", \
                                                   "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"])

increment = 25
starting_point = 850
node_number = 1

#Iterate over the length of the dataframe to repeat it over the course of a year
for i in range(len(df)):
    starting_point = starting_point + increment
    if (starting_point - 60) > len(df):
        break

    #Define series to use model on
    series = df[str(node_number)][:starting_point]
    #print(series)

    #REMOVING RANDOM DATA POINTS
    original_length = len(series)
    print('Original length: ' + str(original_length))
    percentage_to_remove = 25
    counter = 0
    probability_of_point_removal = 0.10
    probability_of_propagation = 0.5
    number_of_points_to_remove = original_length * (percentage_to_remove/100)
    number_of_points_to_remove = math.floor(number_of_points_to_remove)
    print('Number of points to remove: ' + str(number_of_points_to_remove))

    def decision(probability):
        return random.random() < probability
    
    #This is for only removing random datapoints without propagation. Not used in the thesis results
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

    #Remove random points with a propagation probability
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
                            # Set lost values to the previous value
                            dataframe.at[prop_counter] = dataframe.iloc[prop_counter-1]
                            counter += 1
                            i = prop_counter + 1
                            prop_counter += 1
                            continue
                elif i == len(dataframe):
                    break
                else:
                    i += 1
    #Uncomment the line below if we want to lose data points
    #remove_points_with_propagation(series)


    #Train a model with the correct parameters
    mod = sm.tsa.statespace.SARIMAX(series,
                                    order=(1, 0, 1),
                                    seasonal_order=(1, 1, 2, 24),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    #print(results.summary().tables[1])

    pred = results.get_prediction(start=starting_point, dynamic=False)
    pred_ci = pred.conf_int()
    ax = series.plot(label='training', color="blue")
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
    ax.set_xlabel('')
    ax.set_ylabel('Voltage')
    plt.legend()
    plt.show()

    df_truth = df[str(node_number)][starting_point:starting_point+24]

    #Original
    pred_uc = results.get_forecast(steps=24)
    dfvalue_forecasted = pred.predicted_mean
    print(pred_uc.predicted_mean)
    print(df_truth)
    pred_ci = pred_uc.conf_int()
    ax = series[starting_point-50:starting_point].plot(label='training', figsize=(14, 4))
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color="green")
    df_truth.plot(ax=ax, x="Timestamp", label="Actual", color="orange")
    ax.set_xlabel('')
    ax.set_ylabel('Voltage')
    plt.xlim(["2018-10-08","2018-10-12"])
    plt.legend()
    plt.show()

    y_forecasted = pred_uc.predicted_mean
    print("Y FORECASTED " + str(y_forecasted))
    y_truth = df_truth
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))



    # Accuracy metrics
    val_num = starting_point
    correct_counter = 0
    not_correct_counter = 0
    mape_list = []
    abs_error_list = []
    root_square_error_list = []

    for i in range(24):
        if val_num != starting_point:
            #Calculate the accuracy to check
            diff = abs((y_forecasted[val_num] - y_forecasted[val_num-1])/y_forecasted[val_num-1])
            print("DIFF " + str(val_num - 700) + " FOR FORECAST " + str(diff))

            diff_real = (df_truth[val_num] - df_truth[val_num-1])/df_truth[val_num-1]
            print("DIFF REAL " + str(val_num - 700) + " FOR FORECAST " + str(diff_real))

            if (diff > 0 and diff_real > 0) or (diff < 0 and diff_real < 0):
                correct_counter += 1
            else:
                not_correct_counter += 1

        root_square_error = ((y_forecasted[val_num] - df_truth[val_num])**2)
        absolute_error = abs(y_forecasted[val_num] - df_truth[val_num])
        absolute_percentage_error = abs((y_forecasted[val_num] - y_truth[val_num])/y_forecasted[val_num])

        #add to lists to be able to write to file
        root_square_error_list.append(root_square_error)
        abs_error_list.append(absolute_error)
        mape_list.append(absolute_percentage_error)

        val_num = val_num+1

        print("ABS ERROR " + str(absolute_error))
        print("ABS PERCENTAGE ERROR " + str(absolute_percentage_error))

    #convert to percentage
    mape_list = [i * 100 for i in mape_list]
    # Writing to file to get the evaluation metrics
    root_square_error_file = open('root_square_error.csv', 'a')
    abs_error_file = open('abs_error.csv', "a")
    mape_error_file = open("mape_error_25_percent_loss_11.csv", "a")

    writer = csv.writer(root_square_error_file)
    writer.writerow(root_square_error_list)

    writer = csv.writer(abs_error_file)
    writer.writerow(abs_error_list)

    writer = csv.writer(mape_error_file)
    writer.writerow(mape_list)


    #diff_real = (test.values[8] - test.values[7])/test.values[7]
    #print("DIFF " + str(diff))
    print("Correct " + str(correct_counter))
    print("Not correct " + str(not_correct_counter))
    mape_check = np.mean(mape_list)
    print("MAPE CHECK " + str(mape_check))
    rmse_check = np.mean(root_square_error_list)**.5
    print("RMSE CHECK " + str(rmse_check))
