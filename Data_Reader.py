import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

training_set = pd.read_csv('btcdata.csv')
test_set=training_set[1258:]
test_set=test_set.iloc[:,2:3].values
training_set = training_set.iloc[:,2:3].values


sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

X_train = training_set[0:1257]
y_train = training_set[1:1258]
X_train = np.reshape(X_train, (1257, 1, 1))





regressor = Sequential()
regressor.add(LSTM(return_sequences=True,units = 4, activation = 'sigmoid', input_shape = (None, 1)))
regressor.add(LSTM(return_sequences=True,units = 4, activation = 'sigmoid', input_shape = (None, 1)))
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

inputs = test_set
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], 1, 1))

predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print('RMSE: ',rmse)

# Visualising the results
plt.plot(test_set, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
