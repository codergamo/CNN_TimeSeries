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

class EDA:

    def __init__(self, df ,n_steps_in, n_steps_out, feature, train_ratio, valid_ratio, scaler):
          self.data, self.data_old, self.X_train, self.X_test, self.X_valid, self.y_train, self.y_test, self.y_valid, self.index_train, self.index_test, self.index_valid = self.CleanData(df, n_steps_in, n_steps_out, feature, train_ratio,valid_ratio, scaler)
        # self.yc_train, self.yc_test,

    def CleanData(self, data, n_steps_in = 10, n_steps_out = 1, feature = 'Price', train_ratio = 0.3, valid_ratio = 0.2, scaler = 'Min-Max'):
        
        data = data.dropna()
        column_names = tuple(data.drop(data.columns[0], axis=1).columns.values)
        for column in column_names:
            data[column] = data[column].fillna('0').astype(str).str.replace(',', '').str.replace('K', 'e3').str.replace('M', 'e6').str.replace('%', 'e-2').map(lambda x: pd.eval(x) if x != 'nan' else np.nan).astype(float)
        # data['Price'] = data['Price'].str.replace(',', '', regex=True).astype(float)
        # data['Open'] = data['Open'].str.replace(',', '', regex=True).astype(float)
        # data['High'] = data['High'].str.replace(',', '', regex=True).astype(float)
        # data['Low'] = data['Low'].str.replace(',', '', regex=True).astype(float)
     
        data[data.columns[0]] = pd.to_datetime(data[data.columns[0]])
        data = data.sort_values(by=data.columns[0])
        data.set_index(data.columns[0], inplace=True)
        data_old = data

        X_train = data[[feature]]
        y_train = data[[feature]]

        X_scale_dataset, y_scale_dataset = self.normalize_data(X_train, y_train, scaler)

        # n_features = X_value.shape[1]
        X, y = self.get_X_y(X_scale_dataset, y_scale_dataset, n_steps_in, n_steps_out)
        
        X_train, X_valid, X_test = self.split_train_test(X, train_ratio, valid_ratio)
        y_train, y_valid, y_test = self.split_train_test(y, train_ratio, valid_ratio)
        # yc_train, yc_test, = self.split_train_test(yc, split_ratio)
        index_train, index_valid, index_test = self.predict_index(data, X_train, X_valid, n_steps_in, n_steps_out)

        return data, data_old, X_train, X_test, X_valid, y_train, y_test, y_valid, index_train, index_test, index_valid


    def normalize_data(self, X_value, y_value, scaler = 'Min-Max'):
        if scaler == 'Min-Max':
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()


        elif scaler == 'Zero-Mean':
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()

        else:
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            # Khởi tạo trình tạo FunctionTransformer
            # return X_value
            
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
        # yc = list()

        length = len(X_data)
        for i in range(0, length, 1):
            X_value = X_data[i: i + n_steps_in][:, :]
            y_value = y_data[i + n_steps_in: i + (n_steps_in + n_steps_out)][:, 0]
            # yc_value = y_data[i: i + n_steps_in][:, :]
            if len(X_value) == n_steps_in and len(y_value) == n_steps_out:
                X.append(X_value)
                y.append(y_value)
                # yc.append(yc_value)

        return np.array(X), np.array(y)
    # , np.array(yc)



    def split_train_test(self, data, train_ratio = 0.3, valid_ratio = 0.2):
        train_size = round(len(data) * train_ratio)
        valid_size = round(len(data) * valid_ratio)

        data_train = data[:train_size - 1, :]
        data_valid = data[train_size:train_size + valid_size, :]
        data_test = data[train_size + valid_size:, :]

        return data_train, data_valid, data_test

    def predict_index(self, dataset, X_train, X_valid, n_steps_in, n_steps_out):
        train_predict_index = dataset.iloc[n_steps_in: X_train.shape[0] + n_steps_in + n_steps_out - 1, :].index
    
        valid_start_index = X_train.shape[0] + n_steps_in + n_steps_out # Define the starting index for the validation set
        valid_end_index = valid_start_index + X_valid.shape[0]  # Define the ending index for the validation set
        valid_predict_index = dataset.iloc[valid_start_index:valid_end_index, :].index
        
        test_start_index = valid_end_index  # Define the starting index for the test set
        test_end_index = test_start_index + n_steps_out  # Define the ending index for the test set
        test_predict_index = dataset.iloc[test_start_index:test_end_index, :].index
        
        return train_predict_index, valid_predict_index, test_predict_index
    

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

        predictions = model.predict(self.X_valid)
        y_scaler = load(open('./static/y_scaler.pkl', 'rb'))
        rescaled_real_y = y_scaler.inverse_transform(self.y_valid)
        rescaled_predicted_y = y_scaler.inverse_transform(predictions)

        return rescaled_predicted_y, rescaled_real_y, self.index_valid, predictions, self.y_valid
    


    def CNN_Model(self,input_dim=10, output_dim=1, feature_size=1, epochs=50, batch_size=32, activation='relu', learning_rate=0.0001, feature_step=1 ) -> tf.keras.models.Model:
        model = tf.keras.Sequential()
        
        # Thêm lớp Convolutional 1D đầu tiên
        model.add(Conv1D(8, input_shape=(input_dim, feature_size), kernel_size=3, strides=1, padding='same', activation=activation))

        # Thêm các lớp Convolutional 1D và MaxPooling1D tiếp theo
        for i in range (2,feature_step):
            if(i>=5):
                model.add(Conv1D(128, kernel_size=1, strides=1, padding='same', activation=activation))
                model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
            else:
                model.add(Conv1D(8*(2**(i-1)), kernel_size=3, strides=1, padding='same', activation=activation))
                model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
            
            # model.add(Conv1D(32, kernel_size=3, strides=1, padding='same', activation=activation))
            # model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
            # model.add(Conv1D(64, kernel_size=3, strides=1, padding='same', activation=activation))
            # model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
            # model.add(Conv1D(128, kernel_size=1, strides=1, padding='same', activation=activation))
            # model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))

        # Hoàn thiện mô hình
        model.add(Flatten())
        model.add(Dense(220, use_bias=True))
        model.add(LeakyReLU())
        model.add(Dense(220, use_bias=True, activation=activation))
        model.add(Dense(units=output_dim))

        # Thiết lập cấu hình cho mô hình để sẵn sàng cho quá trình huấn luyện.
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])

        # Bắt đầu huấn luyện mô hình, với đầu ra là kêt quả của mô hình : loss_values, accuracy_values, val_loss_values, val_accuracy_values
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_valid, self.y_valid),
                verbose=2, shuffle=False)


        # torch.save(model,"./model/CNN_Model.pth")
        # torch.load(model)
        return model