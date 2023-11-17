import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from tensorflow.keras.layers import Dense, LSTM, Conv1D, LeakyReLU, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from kerastuner.tuners import RandomSearch
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pickle import dump
from pickle import load
from matplotlib import pyplot
import torch 
import time
import os
import pickle
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor

class EDA:

    def __init__(self, df ,n_steps_in, n_steps_out, feature, split_ratio, scaler):
          self.data, self.data_old, self.X_train, self.X_test, self.y_train, self.y_test, self.index_train, self.index_test = self.CleanData(df, n_steps_in, n_steps_out, feature, split_ratio, scaler)

    def CleanData(self, data, n_steps_in = 10, n_steps_out = 1, feature = 'Price', split_ratio = 0.8, scaler = 'Min-Max'):
        
        data = data.dropna()
        column_names = tuple(data.drop(data.columns[0], axis=1).columns.values)
        for column in column_names:
            data[column] = data[column].fillna('0').astype(str).str.replace(',', '').str.replace('K', 'e3').str.replace('M', 'e6').str.replace('%', 'e-2').map(lambda x: pd.eval(x) if x != 'nan' else np.nan).astype(float)

        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
        data = data.sort_values(by=data.columns[0])
        data.set_index(data.columns[0], inplace=True)
        data_old = data
        X_value = data[[feature]]
        y_value = data[[feature]]

        X_scale_dataset, y_scale_dataset = self.normalize_data(X_value, y_value, scaler)

        
        X, y = self.get_X_y(X_scale_dataset, y_scale_dataset, n_steps_in, n_steps_out)
        X_train, X_test, = self.split_train_test(X, split_ratio)
        y_train, y_test, = self.split_train_test(y, split_ratio)

        index_train, index_test, = self.predict_index(data, X_train, n_steps_in, n_steps_out)

        return data, data_old, X_train, X_test, y_train, y_test, index_train, index_test


    def normalize_data(self, X_value, y_value, scaler = 'Min-Max'):
        if scaler == 'Min-Max':
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()


        elif scaler == 'Zero-Mean':
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()

        else:
            # X_scaler = StandardScaler()
            # y_scaler = StandardScaler()
            # Khởi tạo trình tạo FunctionTransformer
            
            return X_value.to_numpy(), y_value.to_numpy()

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

        length = len(X_data)
        for i in range(0, length, 1):
            X_value = X_data[i: i + n_steps_in][:, :]
            y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
            if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
                X.append(X_value)
                y.append(y_value)

        return np.array(X), np.array(y)

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

        return rescaled_predicted_y, rescaled_real_y, self.index_test, predictions, self.y_test
    
    def train_model(self, model):
        history = model.fit(self.X_train, self.y_train, validation =(self.X_test, self.y_test))
        return history
        #return model.fit(self.X_test, self.y_test)

    def CNN_Model(self,input_dim=10, output_dim=1, activation='relu', learning_rate=0.0001):
        model = Sequential()
        
        # Thêm lớp Convolutional 1D đầu tiên
        model.add(Conv1D(8, input_shape=(1, input_dim), kernel_size=3, strides=1, padding='same', activation=activation))

        # Thêm các lớp Convolutional 1D và MaxPooling1D tiếp theo
        model.add(Conv1D(16, kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
        model.add(Conv1D(128, kernel_size=1, strides=1, padding='same', activation=activation))
        model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
            
        # Hoàn thiện mô hình
        model.add(Flatten())
        model.add(Dense(220, use_bias=True))
        model.add(LeakyReLU())
        model.add(Dense(220, use_bias=True, activation=activation))
        model.add(Dense(units=output_dim))

        # Thiết lập cấu hình cho mô hình để sẵn sàng cho quá trình huấn luyện.
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model