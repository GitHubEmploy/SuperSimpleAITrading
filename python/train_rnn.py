# This program trains the rnn that will trade stocks

import numpy as np
import pandas as pd
import coloredlogs, logging
global logger
from yahoo_fin import stock_info as si
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

def prepare(trainset_path, NUMBARS):
    # Read in the dataset and save as panda dataframe
    logger.info("Reading in dataset...")
    dataset_train = pd.read_csv(trainset_path, sep=r'\s*,\s*', engine='python')
    # convert panda dataframe to numpy array
    training_set = dataset_train.to_numpy()
    logger.info("Dataset Head: ")
    print(dataset_train.head())


    # Normalize data
    logger.info("Normalizing data...")
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)


    # Convert dataset into x_train with shape of (Number of tuples, NUMBARS, data per bar)
    # y_train being shape of (Number of tuples,)
    # Data per bar being price, volume, etc
    # x_train is data that is being trained on
    # y_train is considered the true value that the network should get
    x_train = []
    y_train = []
    for i in range(NUMBARS, 1000):
        x_train.append(training_set_scaled[i-NUMBARS:i])
        try:
            y_train.append(training_set_scaled[i, 0])
        except IndexError:
            print('skipped', i)

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train


"""Build Neural Network"""

def train_network(x_train, y_train, num_epochs):
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense, Flatten

    model = Sequential()

    model.add(LSTM(units=50,return_sequences=True, input_shape=(x_train.shape[1], 5)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    logger.info('Model: ')
    model.summary()
    model.compile(optimizer='adam', loss= 'mean_squared_error')

    model.fit(x_train,y_train,epochs=num_epochs, batch_size=1)
    
    return model

"""Test the Results"""

def test_results(trainset_path, testset_path, model, NUMBARS):
    dataset_train = pd.read_csv(trainset_path, sep=r'\s*,\s*', engine='python')
    # convert panda dataframe to numpy array
    training_set = dataset_train.to_numpy()
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    
    # Read in test dataset
    logger.info('Reading test dataset...')
    dataset_test = pd.read_csv(testset_path, sep=r'\s*,\s*', engine='python')
    real_stock_price = dataset_test.to_numpy()

    # Reshape data
    inputs = sc.transform(dataset_test[:].values)
    X_test = []
    for i in range(NUMBARS, 1000):
      X_test.append(inputs[i-NUMBARS:i])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],  5))
    logger.info('X_test shape: ')
    logger.info(X_test.shape)

    # Predict next prices
    logger.info('Predicting prices...')
    predicted_stock_price = model.predict(X_test)

    # Add 4 columns of 0 onto predictions so it can be fed back through sc
    shaped_predictions = np.empty(shape = (990, 5))
    for row in range(0, 990):
        try:
            shaped_predictions[row, 0] = predicted_stock_price[row, 0]
        except IndexError:
            shaped_predictions[row, 0] = predicted_stock_price[0]
        for col in range (1, 5):
            shaped_predictions[row, col] = 0
        
    # un-Normalize data
    predicted_stock_price = sc.inverse_transform(shaped_predictions)

    # Graph results
    logger.info('Results were saved to mygraph.png ')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(real_stock_price[:,0], color = 'black', label = 'Stock Price')
    plt.plot(predicted_stock_price[:,0], color = 'green', label = 'Price Prediction')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig("mygraph.png")
    logger.info('Done')
