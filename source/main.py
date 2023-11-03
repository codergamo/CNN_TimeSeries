import time
import os
import io
import xlwings as xlw
import math 
import torch
import xlsxwriter
import win32com.client
import openpyxl as ox
import pandas as pd
import numpy as np
from EDA import EDA
from Multiple_Lines import MultipleLines
import streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM
from keras.callbacks import Callback
from streamlit.logger import get_logger
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from kerastuner.tuners import RandomSearch
import xlsxwriter


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
m = None
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



def CV_RMSE(predict, actual):
    # Số lượng fold (chẳng hạn, 5-fold cross-validation)
    num_folds = 5

    # Khởi tạo K-fold cross-validation
    kf = KFold(n_splits=num_folds)

    # Tạo danh sách để lưu kết quả RMSE từ từng fold
    rmse_scores = []

    for train_index, test_index in kf.split(actual):
        predicted_test, actual_test = predict[test_index], actual[test_index]
        
        mse = mean_squared_error(actual_test, predicted_test)
        rmse = math.sqrt(mse)
        
        rmse_scores.append(rmse)

    # Tính tổng RMSE từ các fold và tính RMSE trung bình
    average_rmse = np.mean(rmse_scores)
    return average_rmse

def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    st.download_button(label='📥 Download Current Result',
                                data=processed_data,
                                file_name= 'report.xlsx')
    # return processed_data

# Hàm đánh giá
@st.cache_data
def Score(predict, actual):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predict)
    cv_rmse = CV_RMSE(predict,actual)
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "CV_RMSE": cv_rmse
    }
    return metrics
#Tính CV_RMSE
@st.cache_data
def CV_RMSE(predict,actual):
    num_folds = 5
    rmse_scores = []
    kf = KFold(n_splits=num_folds)
    for train_index, test_index in kf.split(actual):
        actual_test, predicted_test = actual[test_index], predict[test_index]
    
        mse = mean_squared_error(actual_test, predicted_test)
        rmse = np.sqrt(mse)
        
        rmse_scores.append(rmse)
    average_rmse = np.mean(rmse_scores)
    return average_rmse

# Xóa dữ liệu lưu trong streamlit
def ClearCache():
    st.session_state.clear()

def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    
    return processed_data
if 'clicked_train' not in st.session_state:
    st.session_state.clicked_train = False

def click_button_train():
    st.session_state.clicked_train = True

if 'clicked_save' not in st.session_state:
    st.session_state.clicked_save = False

def click_button_save():
    st.session_state.clicked_save = True

#--------------------------------------
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
                            value=7, step=1, min_value=1, on_change=ClearCache)

with col2:
    output_dim = st.number_input('**Số ngày muốn dự đoán:**', value=1,
                            step=1, min_value=1, on_change=ClearCache)

# Chọn tỉ lệ chia tập train/test
train_size = st.sidebar.slider('**Tỉ lệ training**', 10, 90, 80, step=10)
split_ratio = train_size/100

# Chọn SL Epoch & SL Batch Size
col3, col4, col5 = st.sidebar.columns(3)
with col3:
    epochs = st.number_input(
        '**Epoch**', value=3, step=1, min_value=1, on_change=ClearCache)
with col4:
    batch_size = st.number_input(
        '**Batch Size**', value=100, step=1, min_value=1, on_change=ClearCache)
with col5:
    feature_loop = st.number_input(
        '**Feature_Loop**', value=1, step=1, min_value=1, on_change=ClearCache)



# Chọn tốc độ học
default_value = 0.0001
learning_rate = st.sidebar.number_input("**Learning Rate**", value=default_value, step=0.00005, min_value=0.0000, format="%.5f")



activation = st.sidebar.selectbox(
    '**Chọn Activation funcion**', ('ReLU', 'sigmoid', 'tanh'), on_change=ClearCache)


scaler = st.sidebar.selectbox(
    '**Chọn phương pháp chuẩn hóa dữ liệu**', ('Min-Max', 'Zero-Mean', 'Dữ liệu gốc'), on_change=ClearCache)

# Chọn tập dữ liệu
st.header("Dữ liệu")
uploaded_file = st.file_uploader(
    "Chọn tệp dữ liệu", type=["csv"], on_change=ClearCache)


def training_model(feature_step, start_time, train_time, m, rmse_traning, feature_hyper):
    st.write(f"Số lớp ẩn: {feature_step}")
    train_time = "{:.4f}".format((time.time() * 1000) - (start_time * 1000))
    st.write(f"Thời gian huấn luyện {train_time}ms")
    st.session_state.train_time = train_time
    st.session_state.m = m
        
    predict, actual, index, predict_scale, actua_scale = eda.TestingModel(m)

    mse = mean_squared_error(actua_scale, predict_scale)
    rmse = np.sqrt(mse)

    if rmse < rmse_traning:
        rmse_traning = rmse
        feature_hyper = feature_step
        torch.save(model,"./model/CNN_Model.pth")

    st.write("MSE:", mse)
    st.write("RMSE:", rmse)

    return predict, actual, index, predict_scale, actua_scale, rmse_traning, feature_hyper

if uploaded_file is not None:
    file_name = uploaded_file.name
    df = LoadData(uploaded_file)


    # Chọn cột dự đoán & activation function

    selected_predict_column_name = st.sidebar.selectbox(
        '**Chọn cột để dự đoán:**', tuple(df.drop(df.columns[0],axis = 1).columns.values), on_change=ClearCache)
    # Tạo đối tượng EDA
    eda = EDA(df = df, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name, split_ratio = split_ratio, scaler = scaler)

    # Thông tin tập dữ liệu
    st.subheader('Tập dữ liệu ' + file_name)
    st.write(df)

    # Vẽ biểu đồ đường cho tập dữ liệu
    st.subheader('Trực quan hóa tập dữ liệu ' + file_name)

    column_names = eda.data_old.columns.tolist()
    selected_column_name = st.selectbox("**Chọn cột:**", column_names)
    fig = MultipleLines.OneLine(eda, selected_column_name)
    st.plotly_chart(fig)

    df_target = df[selected_column_name]
    
    # Training
    if st.sidebar.button('Train Model', type="primary"):
        st.divider()
        st.header("Huấn Luyện Mô Hình")
        #with st.spinner('Đang tiến hành training...'):
        start_time = time.time()
        rmse_traning = 1
        feature_hyper = 0
        if model == 'CNN':
            for feature_step in range (feature_loop , 11):
                m = eda.CNN_Model(input_dim , output_dim , feature_size = 1, epochs=epochs , batch_size=batch_size, activation=activation, learning_rate=learning_rate, feature_step = feature_step)
                predict, actual, index, predict_scale, actua_scale, rmse_traning, feature_hyper = training_model(feature_step, start_time, train_time, m, rmse_traning, feature_hyper)
            st.write("RMSE nhỏ nhất:" , rmse_traning, "Số vòng lặp lớp ẩn: ", feature_hyper)
                

        elif model == 'LSTM':
            m = eda.LSTM_Model(input_dim , output_dim , feature_size = 1, epochs=epochs , batch_size=batch_size, activation=activation, learning_rate=learning_rate)

        st.write("Training Complete!")

    st.header("Dữ liệu")
    uploaded_file1 = st.file_uploader(
    "Chọn tệp dữ liệu test", type=["csv"],on_change=ClearCache)


    if uploaded_file1 is not None:
        file_name_test = uploaded_file1.name
        df_test = LoadData(uploaded_file1)

        selected_predict_column_name_test = st.sidebar.selectbox(
        '**Chọn cột để dự đoán Test:**', tuple(df.drop(df_test.columns[0],axis = 1).columns.values), on_change=ClearCache)

        # Tạo đối tượng EDA
        eda = EDA(df = df_test, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name_test, split_ratio = split_ratio, scaler = scaler)
        # Thông tin tập dữ liệu
        st.subheader('Tập dữ liệu test ' + file_name_test)
        st.write(df_test)

        # Vẽ biểu đồ đường cho tập dữ liệu
        st.subheader('Trực quan hóa tập dữ liệu ' + file_name_test)

        column_names_test = eda.data_old.columns.tolist()
        selected_column_name_test = st.selectbox("**Chọn cột vẽ biểu đồ:**", column_names_test)
        fig_test = MultipleLines.OneLine(eda, selected_column_name_test)
        st.plotly_chart(fig_test)

        st.sidebar.button('Test Model', type="primary", on_click= click_button_train)   
        if st.session_state.clicked_train:
            
            m = eda.CNN_Model(input_dim , output_dim , feature_size = 1, epochs=epochs , batch_size=batch_size, activation=activation, learning_rate=learning_rate, feature_step = feature_hyper)

            with st.spinner('Đang tiến hành training...'):
                start_time = time.time()
                # Hiển thị thời gian train
                train_time = "{:.4f}".format((time.time() * 1000) - (start_time * 1000))
                st.write(f"Thời gian huấn luyện {train_time}ms")
                st.session_state.train_time = train_time

                predict, actual, index, predict_scale, actua_scale = eda.TestingModel(m)

                # Kiểm tra kết quả dự đoán và thực tế 
                result_test_table = pd.DataFrame(
                    {"Ngày" : index,"Giá trị dự đoán": predict.tolist(), "Giá trị thực": actual.tolist()})
                
                st.session_state.resul_test_table = result_test_table
                st.table(result_test_table[:10])    

                # Tính các lỗi 
                metrics=Score(predict_scale,actua_scale)
                
                st.table(metrics)

                # Biểu đồ so sánh
                mline = MultipleLines.MultipLines(predict,actual, index)
                
                st.plotly_chart(mline)
    


        # #Download kết quả về file excel
        # download_button = st.download_button(label='📥 Download Current Result',
        #                     data=to_excel(result_test_table),
        #                     file_name= 'report.xlsx')
        # #Lưu kết quả về thư mục hiện hành
        # st.button('Lưu dữ liệu Excel', type="secondary", on_click=click_button_save, key='save_button')
        # if st.clicked_save:
        #     # csv = result_test_table
        #     # csv.to_excel('./output/data.xlsx', engine='xlsxwriter')  
        #     st.success("Xuất dữ liệu thành công!!")
    

            
            
            
            
            
