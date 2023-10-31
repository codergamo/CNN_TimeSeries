import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Conv1D, LeakyReLU, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pickle import dump
from pickle import load
from matplotlib import pyplot
import torch 
import time
import os

class EDA:

    def __init__(self, df ,n_steps_in, n_steps_out, feature, split_ratio ):
        self.data, self.data_old, self.X_train, self.X_test, self.y_train, self.y_test, self.yc_train, self.yc_test, self.index_train, self.index_test = self.CleanData(df, n_steps_in, n_steps_out, feature, split_ratio)

    def CleanData(self, data, n_steps_in = 10, n_steps_out = 1, feature='Price', split_ratio = 0.8):
        data = data.dropna()
        data['Price'] = data['Price'].str.replace(',', '', regex=True).astype(float)
        data['Open'] = data['Open'].str.replace(',', '', regex=True).astype(float)
        data['High'] = data['High'].str.replace(',', '', regex=True).astype(float)
        data['Low'] = data['Low'].str.replace(',', '', regex=True).astype(float)
        data['Vol'] = data['Vol'].str.replace('K', 'e3').str.replace('M', 'e6').map(pd.eval).astype(int)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date')
        data.set_index('Date', inplace=True)
        data = data.drop('Change %', axis=1)
        data_old = data
        X_value = data[[feature]]
        y_value = data[['Price']]

        X_scale_dataset, y_scale_dataset = self.normalize_data(X_value, y_value)
        n_features = X_value.shape[1]
        X, y, yc = self.get_X_y(X_scale_dataset, y_scale_dataset, n_steps_in, n_steps_out)
        X_train, X_test, = self.split_train_test(X, split_ratio)
        y_train, y_test, = self.split_train_test(y, split_ratio)
        yc_train, yc_test, = self.split_train_test(yc, split_ratio)
        index_train, index_test, = self.predict_index(data, X_train, n_steps_in, n_steps_out)

        return data, data_old, X_train, X_test, y_train, y_test, yc_train, yc_test, index_train, index_test


    def normalize_data(self, X_value, y_value):
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_scaler.fit(X_value)
        y_scaler.fit(y_value)
        dump(X_scaler, open('./static/X_scaler.pkl', 'wb'))
        dump(y_scaler, open('./static/y_scaler.pkl', 'wb'))
        X_scale_dataset = X_scaler.fit_transform(X_value)
        y_scale_dataset = y_scaler.fit_transform(y_value)

        return X_scale_dataset, y_scale_dataset

    def get_X_y(self, X_data, y_data, n_steps_in, n_steps_out):
        X = list()
        y = list()
        yc = list()

        length = len(X_data)
        for i in range(0, length, 1):
            X_value = X_data[i: i + n_steps_in][:, :]
            y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
            yc_value = y_data[i: i + n_steps_in][:, :]
            if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
                X.append(X_value)
                y.append(y_value)
                yc.append(yc_value)

        return np.array(X), np.array(y), np.array(yc)


    def split_train_test(self, data, split_ratio = 0.8):
        train_size = round(len(data) * split_ratio)
        data_train = data[0:train_size]
        data_test = data[train_size:]

        return data_train, data_test

    def predict_index(self, dataset, X_train, n_steps_in, n_steps_out):
        train_predict_index = dataset.iloc[n_steps_in : X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
        test_predict_index = dataset.iloc[X_train.shape[0] + n_steps_in:, :].index

        return train_predict_index, test_predict_index

    def LSTM_Model(self, input_dim=10, output_dim=1, feature_size=1, epochs=50, batch_size=32, activation='relu', learning_rate=0.0001) -> tf.keras.models.Model:
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(input_dim, feature_size), activation=activation))
        model.add(LSTM(units=64, activation=activation))
        model.add(Dense(32, activation=activation))
        model.add(Dense(units=output_dim))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test),
                verbose=2, shuffle=False)
        torch.save(model,'./model/LSTM_Model.pth')

        return model

    def TestingModel(self, model):
        predictions = model.predict(self.X_test, verbose=0)
        y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
        rescaled_real_y = y_scaler.inverse_transform(self.y_test)
        rescaled_predicted_y = y_scaler.inverse_transform(predictions)

        return rescaled_predicted_y, rescaled_real_y, self.index_test
    
    def CNN_Model(self,input_dim=10, output_dim=1, feature_size=1, epochs=50, batch_size=32, activation='relu', learning_rate=0.0001) -> tf.keras.models.Model:
        model = tf.keras.Sequential()
        model.add(Conv1D(8, input_shape=(input_dim, feature_size), kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Conv1D(128, kernel_size=1, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(220, use_bias=True))
        model.add(LeakyReLU())
        model.add(Dense(220, use_bias=True, activation=activation))
        model.add(Dense(units=output_dim))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test),
                verbose=2, shuffle=False)
        torch.save(model,"./model/CNN_Model.pth")
        return model