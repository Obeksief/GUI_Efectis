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
from sklearn.multioutput import MultiOutputRegressor

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
    random_forest = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                           max_depth=3,
                                           random_state=123))
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
    xgboost = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100,
                                  max_depth=3,
                                  learning_rate=0.1,
                                  random_state=123))
    return xgboost
    
def create_catboost():
    catboost = MultiOutputRegressor(cb.CatBoostRegressor(iterations=100,
                                      depth=3,
                                      learning_rate=0.1,
                                      loss_function='RMSE',
                                      random_state=123))
    return catboost

############################################
##         Model Training Functions       ##
############################################
    
def train_neural_network(neural_network, X_scaled, y_scaled):
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

def download_model_and_scalers(model,scaler_X, scaler_y, file_name):
    file = (model, scaler_X, scaler_y)
    output = pickle.dumps(file)
    b64 = base64.b64encode(output).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="modele_{file_name}.pkl">Download Trained {file_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)