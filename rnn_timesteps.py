from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def get_data(dataset='DATA/btcdataset.csv',n=1258,timesteps=60):
    dataset_train = pd.read_csv(dataset)
    training_set = dataset_train.iloc[:,2:3].values
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    test_set = dataset_train.iloc[n:,2:3].values
    real_stock_price = np.concatenate((training_set[0:n], test_set), axis = 0)
    scaled_real_stock_price = sc.fit_transform(real_stock_price)
    inputs = []
    for i in range(1258, 1500):
        inputs.append(scaled_real_stock_price[i-timesteps:i, 0])
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))

    return training_set_scaled,inputs

def create_timestepped_data(training_set_scaled,timesteps=60)
    X_train = []
    y_train = []
    for i in range(timesteps, 1258):
        X_train.append(training_set_scaled[i-timesteps:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train

def build_model(deep=4,wide=3)
    regressor = Sequential()
    regressor.add(LSTM(units = wide, return_sequences = True, input_shape = (None, 1)))
    for i in range(deep-2):
        regressor.add(LSTM(units = wide, return_sequences = True))
    regressor.add(LSTM(units = wide))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    return regressor



def plot_graph(real_stock_price,predicted_stock_price):
    plt.plot(real_stock_price[1258:], color = 'red', label = 'Real BTC Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BTC Price')
    plt.title('BTC Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('BTC Price')
    plt.legend()
    plt.show()

def main():
    training_set_scaled,inputs=get_data()
    X_train,y_train=create_timestepped_data(training_set_scaled,60)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    regressor=build_model()
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    predicted_stock_price = regressor.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    plot_graph(real_stock_price,predicted_stock_price)

if __name__ == '__main__':
    main()
