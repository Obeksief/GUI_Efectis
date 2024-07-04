import streamlit as st
import supervised as sp
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os
import pickle
import base64
import tempfile

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import catboost as cb
from sklearn.ensemble import RandomForestRegressor

def download_models(automl):
    
    archive_path = shutil.make_archive('models', 'zip', automl._results_path)
    

    with open(archive_path, "rb") as f:
        bytes_content = f.read()
        b64 = base64.b64encode(bytes_content).decode()

    href = f'<a href="data:application/zip;base64,{b64}" download="models.zip">Download Trained Models.zip File</a>'
    st.markdown(href, unsafe_allow_html=True)

def cleaned_data(input_data, inputs, outputs):
    X = input_data[inputs]
    y = input_data[outputs]
    return X, y



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

def scale_data(X, y):

    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()

    X_scaled = scaler_1.fit_transform(X)
    y_scaled = scaler_2.fit_transform(y)

    st.session_state['scaler_1'] = scaler_1
    st.session_state['scaler_2'] = scaler_2
    
    return X_scaled, y_scaled


        

st.title('Modeles')

tab1, tab2, tab3 = st.tabs(["1", "2", "3"])

display_features = False
entrainement = False

with tab1:
    st.dataframe(st.session_state['data'])


                


