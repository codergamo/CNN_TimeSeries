import time
import os
import xlwings as xlw
import win32com.client
import openpyxl as ox
import pandas as pd
import numpy as np
from EDA import EDA
import streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM
from keras.callbacks import Callback
from streamlit.logger import get_logger
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score



st.set_page_config(page_title="Forecast Time Series",page_icon=":bar_chart:",layout="centered")

st.write("# DỰ ĐOÁN BẰNG MẠNG NƠ-RON TÍCH CHẬP (CONVOLUTIONAL NEURAL NETWORK)")
st.divider()
st.sidebar.write("# Thiết lập mô hình")

# ------------------------- Biến toàn cục ------------------------- #
df = None
df_target = None
df_scaled = None
model = None
d = None
t = None
train_size = None
degree = None
unit = None
epoch= None
batch_size = None
w = None
scaler = None
test_size = None
train = None
test = None
train_time = None
test_time = None
predict = None
actual = None
info_table = None
result_table = None
metric_table = None
metrics = None
learning_rate = None
generator = None
discriminator = None

# ------------------------- Function ------------------------- #
# load data.csv
@st.cache_data
def LoadData(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# Hàm đánh giá
@st.cache_data
def Score(predict, actual):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predict)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }
    return metrics

# Xóa dữ liệu lưu trong streamlit
def ClearCache():
    st.session_state.clear()

# Sidebar
# Chọn mô hình
model = st.sidebar.selectbox(
    "Chọn mô hình:",
    ["CNN", "LSTM"],
    on_change=ClearCache).lstrip('*').rstrip('*')

# Chọn ngày để dự đoán
col1, col2 = st.sidebar.columns(2)
with col1:
    input_dim = st.number_input('**Số ngày dùng để dự đoán:**',
                            value=2, step=1, min_value=1, on_change=ClearCache)

with col2:
    output_dim = st.number_input('**Số ngày muốn dự đoán:**', value=1,
                            step=1, min_value=1, on_change=ClearCache)

# Chọn tỉ lệ chia tập train/test
train_size = st.sidebar.slider('**Tỉ lệ training**', 10, 90, 80, step=10)
split_ratio = train_size/100

# Chọn SL Epoch & SL Batch Size
col3, col4 = st.sidebar.columns(2)
with col3:
    epochs = st.number_input(
        '**Epoch**', value=50, step=1, min_value=1, on_change=ClearCache)
with col4:
    batch_size = st.number_input(
        '**Batch Size**', value=32, step=1, min_value=1, on_change=ClearCache)

# Chọn tốc độ học
default_value = 0.0001
learning_rate = st.sidebar.number_input("**Learning Rate**", value=default_value, step=0.00005, min_value=0.0001, format="%.5f")

# Chọn cột dự đoán & activation function

selected_predict_column_name = st.sidebar.selectbox(
    '**Chọn cột để dự đoán:**', ('Price', 'Open', 'High', 'Low', 'Vol'), on_change=ClearCache)

activation = st.sidebar.selectbox(
    '**Chọn Activation funcion**', ('ReLU', 'Sigmoid', 'Tanh'), on_change=ClearCache)


# Chọn tập dữ liệu
st.header("Dữ liệu")
uploaded_file = st.file_uploader(
    "Chọn tệp dữ liệu", type=["csv"], on_change=ClearCache)

if uploaded_file is not None:
    file_name = uploaded_file.name
    df = LoadData(uploaded_file)

    # Tạo đối tượng EDA
    eda = EDA(df = df, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name, split_ratio = split_ratio)

    # Thông tin tập dữ liệu
    st.subheader('Tập dữ liệu ' + file_name)
    st.write(df)

    # Vẽ biểu đồ đường cho tập dữ liệu
    st.subheader('Trực quan hóa tập dữ liệu ' + file_name)

    column_names = eda.data_old.columns.tolist()
    selected_column_name = st.selectbox("**Chọn cột:**", column_names)
    trace = go.Scatter(x=eda.data_old.index ,y=eda.data_old[selected_column_name], mode='lines', name='Giá cổ phiếu')

    layout = go.Layout(
        title='Biểu đồ giá cổ phiếu',
        xaxis=dict(title='Ngày'),
        yaxis=dict(title='Giá cổ phiếu'),
        hovermode='closest'
    )

    fig = go.Figure(data=[trace], layout=layout)
    
    fig.update_layout()  # Kích thước tùy chỉnh 800x400

    st.plotly_chart(fig)

    df_target = df[selected_column_name]

    # Training
    st.divider()
    st.header("Huấn Luyện Mô Hình")
    if st.button('Train Model', type="primary"):
        with st.spinner('Đang tiến hành training...'):
            start_time = time.time()
            if model == 'CNN':
                w = eda.CNN_Model(input_dim , output_dim , feature_size = 1, epochs=epochs , batch_size=batch_size, activation=activation, learning_rate=learning_rate)
            if model == 'LSTM':
                w = eda.LSTM_Model(input_dim , output_dim , feature_size = 1, epochs=epochs , batch_size=batch_size, activation=activation, learning_rate=learning_rate)

            train_time = "{:.2f}".format(time.time() - start_time)

            st.session_state.train_time = train_time
            st.session_state.w = w

            #predict, actual, index = eda.TestingModel(w)
            st.write("Training Complete!")
     #       Plot(predict, actual, index, model)
            test = eda.TestingModel(w)
            st.write(test.shape)

        
        
            # result_test_table = pd.DataFrame(
            #     {"Dự đoán": predict.tolist(), "Thực tế": actual.tolist()})

            # st.session_state.result_table = result_table

            # st.table(result_test_table[:10])    