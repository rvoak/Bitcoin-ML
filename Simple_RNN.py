import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def read_data(dataset='DATA/btcdata.csv',num=1257)
    training_set = pd.read_csv(dataset)

    # Obtain Open-High values of first 1258 transactions
    test_set=training_set[num+1:]
    test_set=test_set.iloc[:,2:3].values

    sc = MinMaxScaler()
    training_set = training_set.iloc[:,2:3].values
    training_set = sc.fit_transform(training_set)

    X_train = training_set[0:num]
    y_train = training_set[1:num+1]
    X_train = np.reshape(X_train, (num, 1, 1))

    inputs = test_set
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], 1, 1))

    return X_train,y_train,inputs,test_set


def build_model():
    regressor = Sequential()
    regressor.add(LSTM(return_sequences=True,units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    regressor.add(LSTM(return_sequences=True,units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor

def plot_graph(predicted_stock_price,test_set):
    plt.plot(test_set, color = 'red', label = 'Real BTC Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Price')
    plt.title('BTC Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('BTC Price')
    plt.legend()
    plt.show()


def main():
    X_train,y_train,inputs,test_set=read_data()
    regressor=build_model()
    regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    plot_graph(predicted_stock_price,test_set)
    rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
    print('RMSE: ',rmse)

if __name__ == '__main__':
    main()
