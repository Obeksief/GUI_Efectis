import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import base64
import time
import matplotlib.pyplot as plt
import plotly.express as px
import base64
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import tensorflow as tf
from io import BytesIO
import base64

TF_ENABLE_ONEDNN_OPTS=0

############################################
##         Data Cleaning Functions        ##
############################################

def split_input_output(input_data, inputs, outputs):
    X = input_data[inputs]
    y = input_data[outputs]
    return X, y

def scale_data(X, y):

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()

    X_scaled = scaler_1.fit_transform(X)
    y_scaled = scaler_2.fit_transform(y)

    st.session_state['scaler_X'] = scaler_1
    st.session_state['scaler_y'] = scaler_2
    
    return X_scaled, y_scaled


############################################
##         Model Creating Functions       ##
############################################

def create_random_forest():
    random_forest = RandomForestRegressor(n_estimators=100,
                                           max_depth=3,
                                           random_state=123)
    return random_forest

def create_neural_network():
    neural_network = MLPRegressor(hidden_layer_sizes=(32, 16),
                                      activation="relu",
                                      solver="adam",
                                      learning_rate="adaptive",
                                      learning_rate_init=0.1,
                                      alpha=0.0001,
                                      early_stopping=True,
                                      n_iter_no_change=50,
                                      max_iter=1000,
                                      random_state=123)
    return neural_network

def create_xgboost():
    xgboost = xgb.XGBRegressor(n_estimators=100,
                                  max_depth=3,
                                  learning_rate=0.1,
                                  random_state=123)
    return xgboost
    
def create_catboost():
    catboost = cb.CatBoostRegressor(iterations=100,
                                      depth=3,
                                      learning_rate=0.1,
                                      loss_function='RMSE',
                                      random_state=123)
    return catboost

############################################
##         Model Training Functions       ##
############################################
    
def train_neural_network(neural_network, X, y):
    X_scaled, y_scaled = scale_data(X, y)
    neural_network.fit(X_scaled, y_scaled)
    return neural_network
    
def train_xgboost(xgboost, X, y):
    xgboost.fit(X, y)
    return xgboost
    
def train_catboost(catboost, X, y):
    catboost.fit(X, y)
    return catboost
    
def train_random_forest(random_forest, X, y):
    random_forest.fit(X, y)
    return random_forest

def create_model():
    mini_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='gelu'),
        tf.keras.layers.Dense(32, activation='gelu'),
        tf.keras.layers.Dense(1)
    ])
    mini_model.compile(loss=tf.keras.losses.mse,
                       optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.012),
                       metrics=['mse'])
    
    return mini_model

def train_model(model, X_scaled, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=3)
    
        
    model.fit(X_train, y_train, epochs=100, validation_split=0.15, shuffle=True, verbose=0)
    
    
    y_pred = model.predict(X_test)
    st.write(f"y_pred shape: {y_pred.shape}")
        
    acc = mean_absolute_percentage_error(y_test, y_pred) * 100
    return acc

########################################
##         Download Functions         ##
########################################


def download_scaler(scaler):
    output_scaler = pickle.dumps(scaler)
    b64 = base64.b64encode(output_scaler).decode()
    href = f'<a href="data:file/output_scaler;base64,{b64}" download="scaler.pkl">Download scaler .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def generate_excel_download_link(df, nom):
    towrite = BytesIO()
    df.to_excel(towrite, index=False, header=True)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{nom}.xlsx">Download {nom} File</a>'
    return st.markdown(href, unsafe_allow_html=True)

def download_model(model, model_name):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{model_name}.pkl">Download Trained {model_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)
