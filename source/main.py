import time
import io
import math 
import torch
import win32com.client
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import numpy as np
from EDA import EDA
from Multiple_Lines import MultipleLines
import streamlit as st
import tensorflow as tf
from streamlit.logger import get_logger
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
tf.get_logger().setLevel('ERROR')

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

# Function to save all dataframes to one single excel

def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df[0].to_excel(writer, index=False, sheet_name='Result Test')
    df[1].to_excel(writer, index=False, sheet_name='Metrics')
    workbook = writer.book
    worksheet1 = writer.book
    worksheet = writer.sheets['Result Test']
    worksheet1 = writer.sheets['Metrics']

    format1 = workbook.add_format({'num_format': '0.00'}) 

    worksheet.set_column('A:A', None, format1) 
    worksheet1.set_column('A:A', None, format1) 
    writer.close()
    processed_data = output.getvalue()

    return processed_data

# Hàm đánh giá
@st.cache_data
def Score(predict, actual):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predict)
    cv_rmse = CV_RMSE(predict,actual)
    return mae, mse, rmse ,mape ,cv_rmse
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

def dfs_tabs(df_list, sheet_list):

    output = io.BytesIO()

    writer = pd.ExcelWriter(output,engine='xlsxwriter')   
    for dataframe, sheet in zip(df_list, sheet_list):
        dataframe.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)   
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
mod = st.sidebar.selectbox(
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
col3, col4 = st.sidebar.columns(2)
with col3:
    epochs = st.number_input(
        '**Epoch**', value=3, step=1, min_value=1, on_change=ClearCache)
with col4:
    feature_loop = st.number_input(
        '**Feature_Loop**', value=1, step=1, min_value=1, on_change=ClearCache)



batch_size = st.sidebar.selectbox(
    '**Batch Size**', (16, 32, 64, 128, 256, 512), on_change=ClearCache)


# Chọn tốc độ học
default_value = 0.0001
learning_rate = st.sidebar.number_input("**Learning Rate**", value=default_value, step=0.00005, min_value=0.0000, format="%.5f")



activation = st.sidebar.selectbox(
    '**Chọn Activation funcion**', ('ReLU', 'sigmoid', 'tanh'), on_change=ClearCache)


scaler = st.sidebar.selectbox(
    '**Chọn phương pháp chuẩn hóa dữ liệu**', ('Min-Max', 'Zero-Mean', 'Dữ liệu gốc'), on_change=ClearCache)

# Chọn tập dữ liệu
st.header("Chọn tập dữ liệu tiến hành huấn luyện")
uploaded_file = st.file_uploader(
    "Chọn tệp dữ liệu", type=["csv"], on_change=ClearCache)


# Tìm ra thông số cho mô hình train có số lỗi rmse nhỏ nhất và lưu vào thư mục
def hyper_paramter(m, rmse_min, rmse_loop, feature_step, feature_hyper, feature_train):  

    # Test model --> Trả về giá trị dự đoán-thực tế rescale , ngày, giá trị dự đoán - thực tế scale
    predict, actual, index, predict_scale, actua_scale = eda.TestingModel(m)

    mae, mse, rmse, mape, cv_rmse = Score(predict_scale,actua_scale)
    # Thực hiện tính lỗi với mỗi giá trị đã được scale

    rmse_loop.append(rmse)
    feature_train.append(feature_step)

    # Nếu giá trị rmse cũ lớn hơn rmse hiện tại thì lưu giá trị hiện tại
    if rmse < rmse_min:
        rmse_min = rmse
        feature_hyper = feature_step
        # Lưu model vào state hiện tại
        st.session_state.m = m

        #Lưu các paramter vào file CNN_Model.pth
        torch.save({
        'model': m,
        'epochs': epochs,
        'batch_size': batch_size,
        'feature_loop': feature_hyper
        }, "./model/CNN_Model.pth")

    # Trả về giá trị dự đoán-thực tế rescale , ngày, giá trị dự đoán - thực tế scale, rmse nhỏ nhất và số lớp ẩn tương ứng
    return rmse_min, feature_hyper, rmse_loop, feature_train

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
        with st.spinner('Đang tiến hành training...'):
            start_time = time.time()
            # rmse_min = 1
            # rmse_loop = []
            # feature_hyper = 0
            # feature_train = []

            param_dist = {
            'epochs': range(1, 101),
            'batch_size': [16, 32, 64, 128, 256]
            }
            if mod == 'CNN':
                # for feature_step in range (feature_loop , 11):
                    #Truyền các thông số vào model
                    
                model = eda.CNN_Model(input_dim=input_dim, output_dim=output_dim, activation=activation, learning_rate=learning_rate)
                m = KerasRegressor(model=model)
                random_search = RandomizedSearchCV(m, param_distributions=param_dist, cv=3, n_iter=10, n_jobs=-1, scoring='neg_mean_squared_error', error_score='raise')

                X_train = np.reshape(eda.X_train, (eda.X_train.shape[0], 1, eda.X_train.shape[1]))
                

                random_search.fit(eda.X_train, eda.y_train)

                    # Trả về giá trị rmse nhỏ nhất, số lớp ẩn tương ứng
                    # rmse_min, feature_hyper, rmse_loop, feature_train = hyper_paramter(m, rmse_min, rmse_loop, feature_step, feature_hyper, feature_train)
          
            elif mod == 'LSTM':
                m = eda.LSTM_Model(input_dim , output_dim , feature_size = 1, epochs=epochs, batch_size=batch_size, activation=activation, learning_rate=learning_rate)
            

            # st.write("Thông số của vòng lặp có RMSE nhỏ nhất:")
            # result_train_table = pd.DataFrame(
            #     {"epochs": [epochs], "batch_zize": [batch_size],"feature": [feature_hyper],"rmse": [rmse_min]})
            # st.table(result_train_table[:])  

            st.write("Best Parameters:", random_search.best_params_)

            #In thời gian optimize
            optimize_time = "{:.4f}".format((time.time() * 1000) - (start_time * 1000))
            st.write(f"Thời gian Optimize {optimize_time}ms")
            st.session_state.optimize_time = optimize_time
            st.write("Optimize Complete!")

            with st.spinner("Đang huấn luyện mô hình với bộ siêu tham số..."):
                # Lấy bộ tham số tốt nhất từ quá trình tối ưu hóa
                best_params = random_search.best_params_
                if mod == 'CNN':
                    m1 = eda.CNN_Model(input_dim=input_dim, output_dim=output_dim, activation=activation, learning_rate=learning_rate)
                elif mod == 'LSTM':
                    m1 = eda.LSTM_Model(input_dim , output_dim , feature_size = 1, epochs=epochs, batch_size=batch_size, activation=activation, learning_rate=learning_rate)
            
                model_training = eda.train_model(m1,epochs=best_params['epochs'], batch_size=best_params['batch_size'])

                st.session_state.model_training = model_training

                #Lưu các paramter vào file CNN_Model.pth
                torch.save({
                'model': model_training,
                'epochs': best_params['epochs'],
                'batch_size': best_params['batch_size']
                }, "./model/CNN_Model.pth")

                train_time = "{:.4f}".format((time.time() * 1000) - (start_time * 1000))
                st.write(f"Thời gian Training {train_time}ms")
                st.session_state.train_time = train_time
                st.write("Training Complete!")
#Load tập dữ liệu test
st.header("Chọn tập dữ liệu tiến hành dự đoán")
uploaded_file1 = st.file_uploader(
"Chọn tệp dữ liệu test", type=["csv"],on_change=ClearCache)

# Nếu đã upload file
if uploaded_file1 is not None:
    file_name_test = uploaded_file1.name
    df_test = LoadData(uploaded_file1)
    
    #Chọn cột để dự đoán
    selected_predict_column_name_test = st.sidebar.selectbox(
    '**Chọn cột để dự đoán Test:**', tuple(df_test.drop(df_test.columns[0],axis = 1).columns.values), on_change=ClearCache)

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

    #Thực hiện nút test model
    st.sidebar.button('Test Model', type="primary", on_click= click_button_train)   
    if st.session_state.clicked_train:
        #try:
            # Load các paramter được lưu trong CNN_Model.pth
            checkpoint = torch.load("./model/CNN_Model.pth")

            epoch_train = checkpoint["epochs"]
            batch_size_train = checkpoint["batch_size"]
            model_train = checkpoint["model"]

            # Thể hiện các giá trị đã train lên bảng và dùng để test
            st.write("****Các siêu tham số được dùng để dự đoán:****")
            train_table = pd.DataFrame(
                {"epochs": [epoch_train], "batch_zize": [batch_size_train]})
            st.table(train_table[:10])  

            # Thực hiện test
            predict, actual, index, predict_scale, actua_scale = eda.TestingModel(model_train)
            st.write("****So sánh kết quả dự đoán và thực tế:****")
            # Kiểm tra kết quả dự đoán và thực tế 
            result_test_table = pd.DataFrame(
                {"Ngày" : index.tolist(),"Giá trị dự đoán": predict.tolist(), "Giá trị thực": actual.tolist()})
            #Tính lỗi trên từng datapoint để xuất ra exel 
            mse_test = (predict_scale-actua_scale)**2
            result_test_table['MSE'] = mse_test
            
            st.session_state.result_test_table = result_test_table
            st.table(result_test_table[:10])    

            # Tính lỗi của tập dữ liệu và in ra màn hình 
            mae, mse, rmse, mape, cv_rmse = Score(predict_scale,actua_scale)

            metrics = pd.DataFrame({
                "MAE": [mae],
                "MSE": [mse],
                "RMSE": [rmse],
                "MAPE": [mape],
                "CV_RMSE": [cv_rmse]})
            st.write("****Thông số lỗi sau khi dự đoán:****")
            st.table(metrics)

            # Biểu đồ so sánh
            mline = MultipleLines.MultipLines(predict,actual, index)
            
            st.plotly_chart(mline)

            csv_output = [result_test_table,metrics, train_table]

            # list of sheet names
            sheets = ['Result test','metrics', 'train parameters']  

            #df_xlsx = dfs_tabs(csv_output, sheets, 'multi-test.xlsx')  

            #Download kết quả về file excel
            st.download_button(label='📥 Download Current Result',
                                data=dfs_tabs(csv_output, sheets) ,
                                file_name= 'Result-test.xlsx')
            
        #except:
            #st.error("****Hiện tại chưa có Model!****")
            #Lưu kết quả về thư mục hiện hàn
            # st.button('Lưu dữ liệu Excel', type="secondary", on_click=click_button_save, key='save_button')
        # if st.clicked_save:
        #     # csv = result_test_table
        #     # csv.to_excel('./output/data.xlsx', engine='xlsxwriter')  
        #     st.success("Xuất dữ liệu thành công!!")
    

            
            
            
            
            
