# keras using tensorflow
from msilib.schema import Error
import investpy
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('fivethirtyeight')

TRAINING_DAYS = 60


def get_stock_data(ticker, current_date):
    try:
        return investpy.get_stock_historical_data(
            stock=f'{ticker}',
            country='United States',
            from_date='01/01/2020',
            to_date=current_date
        )
    except RuntimeError:
        raise RuntimeError('Please check that you enter a valid ticker')


def build_basic_graph(stock):
    plt.figure(figsize=(16, 8))
    plt.title(f'Close price history {stock}')
    plt.plot(stock['Close'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def build_neural_ntwrk(shape1, shape2, x_train, y_train, model) -> None:
    model.add(LSTM(50, return_sequences=True, input_shape=(shape1, shape2)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=20)


def data_preparations(length_of_training_data):
    data_for_train = scaled_data[0:length_of_training_data]
    x_train = []
    y_train = []
    for point in range(TRAINING_DAYS, len(data_for_train)):
        x_train.append(data_for_train[point-TRAINING_DAYS:point])
        y_train.append(data_for_train[point])
    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train,


def create_dataset(length_of_training_data, scaled_data, dataset) -> tuple:
    test_data = scaled_data[length_of_training_data - TRAINING_DAYS:]
    x_test = []
    y_test = dataset[length_of_training_data:]
    for point in range(TRAINING_DAYS, len(test_data)):
        x_test.append(test_data[point-TRAINING_DAYS:point])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (
                                x_test.shape[0],
                                x_test.shape[1],
                                x_test.shape[2]
                                )
                        )
    return x_test, y_test


def create_predictions(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # rmse = np.sqrt(np.mean(predictions - y_test)**2)
    return predictions


def build_predictions_graph(length_of_training_data, predictions):
    train = data[:length_of_training_data]
    valid = data[length_of_training_data:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker}')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Pred'], loc='lower right')
    plt.show()


if __name__ == '__main__':
    today = datetime.now().strftime('%d/%m/%Y')
    print('Write down the ticker of stock to analyze:'),
    ticker = input()
    historical_data = get_stock_data(ticker, today)
    data = historical_data.filter(['Close'])
    dataset = data.values
    length_of_training_data = math.ceil(len(dataset) * .8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = data_preparations(length_of_training_data)
    shape1, shape2 = x_train.shape[1], x_train.shape[2]
    x_train = np.reshape(x_train, (x_train.shape[0], shape1, shape2))
    model = Sequential()
    build_neural_ntwrk(shape1, shape2, x_train, y_train, model)
    x_test, y_test = create_dataset(
                                    length_of_training_data,
                                    scaled_data,
                                    dataset
                                    )
    build_predictions_graph(
                            length_of_training_data,
                            create_predictions(
                                                model,
                                                x_test,
                                                y_test,
                                                scaler
                                                )
                            )
