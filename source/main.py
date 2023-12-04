import time
import io
import math 
import torch
import pandas as pd
import numpy as np
from EDA import EDA
from Multiple_Lines import MultipleLines
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.layers import Dense, LSTM, Conv1D, LeakyReLU, Flatten, MaxPooling1D, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
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

#Tính CV_RMSE
@st.cache_data
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

# Hàm đánh giá
@st.cache_data
def Score(predict, actual):
    mae = mean_absolute_error(actual, predict)
    mse = mean_squared_error(actual, predict)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predict) / predict))
    cv_rmse = CV_RMSE(predict,actual)
    return mae, mse, rmse ,mape ,cv_rmse



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
if 'display_info' not in st.session_state:
        st.session_state.display_info = {}
def click_button_save():
    st.session_state.clicked_save = True
    

#--------------------------------------
# Sidebar
# Chọn mô hình
mod = st.sidebar.selectbox(
    "Chọn mô hình:",
    ["CNN", "LSTM","RNN"],
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
train_size = st.sidebar.slider('**Tỉ lệ training**', 10, 70, 70, step=10)
valid_size = st.sidebar.slider('**Tỉ lệ Validation**', 10, 90 - train_size, 20, step=10)
train_ratio = train_size/100
valid_ratio = valid_size/100

activation = st.sidebar.selectbox(
    '**Chọn Activation funcion**', ('ReLU', 'LeakyReLU', 'tanh'), on_change=ClearCache)


scaler = st.sidebar.selectbox(
    '**Chọn phương pháp chuẩn hóa dữ liệu**', ('Min-Max', 'Zero-Mean', 'Dữ liệu gốc'), on_change=ClearCache)

# Chọn tập dữ liệu
st.header("Chọn tập dữ liệu tiến hành huấn luyện")
uploaded_file = st.file_uploader(
    "Chọn tệp dữ liệu", type=["csv"], on_change=ClearCache)


def LSTM_Model(input_dim=10, output_dim=1, units =32, learning_rate=0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(LSTM(units=units, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def RNN_Model(input_dim=10, output_dim=1, units =32, learning_rate=0.0001, activation = 'relu') -> tf.keras.models.Model:
    model = Sequential()
    model.add(SimpleRNN(units=units, return_sequences=True, input_shape=(input_dim, 1), activation=activation))
    model.add(SimpleRNN(units=units, activation=activation))
    model.add(Dense(32, activation=activation))
    model.add(Dense(units=output_dim))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

def CNN_Model(input_dim=10, output_dim=1, units = 32, learning_rate = 0.0001, activation = 'relu'):
    model = Sequential()
    # Thêm lớp Convolutional 1D đầu tiên
    model.add(Conv1D(units, input_shape=(input_dim, 1), kernel_size=3, strides=1, padding='same', activation=activation)) 
    model.add(Conv1D(units, kernel_size=3, strides=1, padding='same', activation=activation))
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
def CNN_Retrain(input_dim=10, output_dim=1):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(input_dim,1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(loss='mse', optimizer='adam')
    return model

if uploaded_file is not None:
    file_name = uploaded_file.name
    df = LoadData(uploaded_file)

    # Chọn cột dự đoán & activation function
    selected_predict_column_name = st.sidebar.selectbox(
        '**Chọn cột để tiến hành training:**', tuple(df.drop(df.columns[0],axis = 1).columns.values), on_change=ClearCache)
    # Tạo đối tượng EDA
    eda = EDA(df = df, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name, train_ratio = train_ratio, valid_ratio = valid_ratio, scaler = scaler)

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
    
    # Optimize Model
    if st.sidebar.button('Optimize Model', type="primary"):
        st.divider()
        st.header("Optimize Mô Hình")
        with st.spinner('Đang tiến hành Optimize...'):
            start_time = time.time()
            
            param_dist = {
            'units': [16, 32, 64, 128, 256],
            'epochs': range(1, 101),
            'batch_size': [16, 32, 64, 128, 256],
            'learning_rate': [0.0001]
            }
            if mod == 'CNN':
                m = KerasRegressor(model = CNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)

            elif mod == 'LSTM':
                m =  KerasRegressor(model=LSTM_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
            elif mod == 'RNN':
                m =  KerasRegressor(model=RNN_Model, input_dim=input_dim, output_dim=output_dim, units =32, learning_rate = 0.0001, activation= activation)
                
            
            random_search = RandomizedSearchCV(m, param_distributions=param_dist, cv=3, n_iter=10, n_jobs=-1, scoring='neg_mean_squared_error')
            random_search.fit(eda.X_valid, eda.y_valid)
            #Lưu tham số sau khi optimize
            torch.save({
            'model': random_search,
            'best_params':random_search.best_params_
            }, "./model/Optimize_Model.pth")
            st.write("Best Parameters:", random_search.best_params_)

            #In thời gian optimize
            optimize_time = "{:.4f}".format((time.time() * 1000) - (start_time * 1000))
            st.write(f"Thời gian Optimize {optimize_time}ms")
            st.session_state.optimize_time = optimize_time
            st.session_state.display_info['best_params'] = random_search.best_params_
            st.write("Optimize Complete!")
    #Traing Model        
    if st.sidebar.button('Train Model'):
        st.divider()
        st.header("Huấn luyện Mô Hình")
        st.subheader('Mô hình đã optimize')
        #Load siêu tham số sau khi optimize
        model_op = torch.load("./model/Optimize_Model.pth")
        st.session_state.display_info = model_op['best_params']
        st.write(st.session_state.display_info)
        start_time_train = time.time()
        with st.spinner("Đang huấn luyện mô hình với bộ siêu tham số..."):
            # Lấy bộ tham số tốt nhất từ quá trình optimize
            best_params =  model_op['best_params']
            if mod == 'CNN':
                m1 = CNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
            elif mod == 'LSTM':
                m1 = LSTM_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
            elif mod == 'RNN':
                m1 = RNN_Model(input_dim=input_dim, output_dim=output_dim, units = best_params['units'], learning_rate = best_params['learning_rate'], activation= activation)
            #Tiến hành training
            model_training = eda.train_model(m1,epochs=best_params['epochs'], batch_size=best_params['batch_size'])

            st.session_state.model_training = model_training

            #Lưu các paramter vào file Model.pth
            torch.save({
            'model': model_training,
            'units': best_params['units'],
            'epochs': best_params['epochs'],
            'batch_size': best_params['batch_size'],
            'learning_rate': best_params['learning_rate']
            }, "./model/Model.pth")

            train_time = "{:.4f}".format((time.time() * 1000) - (start_time_train* 1000))
            st.write(f"Thời gian Training {train_time}ms")
            st.session_state.train_time = train_time
            st.write("Training Complete!")
    #Retain Model
    if st.sidebar.button('Retrain Model'):
        st.divider()
        st.header("Huấn luyện Mô Hình")
        start_time_train = time.time()
        with st.spinner("Đang retrain mô hình với tập dữ liệu..."):
            if mod == 'CNN':
                 m=CNN_Retrain(input_dim=input_dim,output_dim=output_dim)
            #Siêu tham số 
            epochs, batch_size = 20, 4
            #Tiến hành train
            model_training = eda.train_model(m,epochs=epochs, batch_size=batch_size)
            st.session_state.model_training = model_training

            #Lưu các paramter vào file Model.pth
            torch.save({
            'model': model_training,
            'units': 16,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': None
            }, "./model/Model.pth")
            #In thời gian 
            train_time = "{:.4f}".format((time.time() * 1000) - (start_time_train* 1000))
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
    '**Chọn cột để dự đoán:**', tuple(df_test.drop(df_test.columns[0],axis = 1).columns.values), on_change=ClearCache)

    # Tạo đối tượng EDA
    eda = EDA(df = df_test, n_steps_in = input_dim, n_steps_out = output_dim, feature=selected_predict_column_name_test, train_ratio = 0, valid_ratio = 0, scaler = scaler)
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
        # try:
            # Load các paramter được lưu trong CNN_Model.pth
            checkpoint = torch.load("./model/Model.pth")

            unit_train = checkpoint["units"]
            epoch_train = checkpoint["epochs"]
            batch_size_train = checkpoint["batch_size"]
            LR_train = checkpoint["learning_rate"]
            model_train = checkpoint["model"]

            # Thể hiện các giá trị đã train lên bảng và dùng để test
            st.write("****Các siêu tham số được dùng để dự đoán:****")
            train_table = pd.DataFrame(
                {"units": [unit_train],"epochs": [epoch_train], "batch_zize": [batch_size_train], "learning_rate": [LR_train]})
            st.table(train_table[:10])  

            # Thực hiện test
            predict, actual, index, predict_scale, actua_scale = eda.TestingModel(model_train)
            st.write("****So sánh kết quả dự đoán và thực tế:****")
            # Kiểm tra kết quả dự đoán và thực tế 
            mse_test = (predict_scale-actua_scale)**2
            result_test_table = pd.DataFrame(
                {"Ngày" : index.tolist(),"Giá trị dự đoán": predict.tolist(), "Giá trị thực": actual.tolist(), "MSE": mse_test.tolist()})
            #Tính lỗi trên từng datapoint để xuất ra exel 
            
            # result_test_table['MSE'] = mse_test
            
            # result_test_table['MSE'] = result_test_table['MSE'].apply(lambda x: format(x, '.10f'))

            st.session_state.result_test_table = result_test_table
            st.write(result_test_table)    

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
            compare_date = st.selectbox("****Chọn ngày để so sánh kết quả dự đoán****",list(range(1,output_dim+1)))
            mline = MultipleLines.MultipLines(predict[:,compare_date-1], actual[:,compare_date-1], index)
            st.plotly_chart(mline)

            csv_output = [result_test_table,metrics, train_table]

            # list of sheet names
            sheets = ['Result test','metrics', 'train parameters']  

            #Download kết quả về file excel
            st.download_button(label='📥 Download Current Result',
                                data=dfs_tabs(csv_output, sheets) ,
                                file_name= 'Result-test.xlsx')           
        # except:
        #     st.error("****Hiện tại chưa có Model!****")