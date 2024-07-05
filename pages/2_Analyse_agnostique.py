import streamlit as st
import supervised as sp
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os
import pickle
import base64
import tempfile
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


def cleaned_data(input_data, inputs, outputs):
    X = input_data[inputs]
    y = input_data[outputs]
    return X, y

def download_model(model, model_name):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="{model_name}.pkl">Download Trained {model_name} Model</a>'
    st.markdown(href, unsafe_allow_html=True)


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

# A Supprimer ###########################
st.session_state['data'] = pd.read_excel('Donnees.xlsx')
#######################################

with tab1:
    st.dataframe(st.session_state['data'])
    st.subheader('Inputs et Outputs')
    if 'data' in st.session_state:
        features = st.session_state['data'].columns
        liste = [str(_) for _ in features]
        inputs = st.multiselect("What are the inputs:", liste)
        outputs = st.multiselect('What are the outputs:', liste)

        # A Supprimer ############
        inputs = ['x1', 'x2', 'x3']
        outputs = ['y']
        st.write('Salut Kilian, valide simplement la saisie pour passer à l\'étape suivante et ne t\'embetes pas' )
        ##########################

        if st.button("valider la saisie"):

            X, y = cleaned_data(st.session_state.data, inputs, outputs)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=3)

            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test


            st.write('Data preparation done')

with tab2:
    st.subheader('Choix du modele')
    liste_models = st.multiselect('Choisir les modeles', ['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'])
    if st.button('Valider les choix'):
        ## A Supprimer ##################
        liste_models = ['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost']
        ################################
        st.session_state['liste_models'] = liste_models
        for model in liste_models:
            st.session_state[model] = globals()[f'create_{model.lower().replace(" ", "_")}']()
        st.write('Models created')
        st.session_state['times']= []
        app = st.session_state['times'].append
        for model in liste_models:
            start_time = time.time()
            st.session_state[model] = globals()[f'train_{model.lower().replace(" ", "_")}'](st.session_state[model], st.session_state['X_train'], st.session_state['y_train'])
            end_time = time.time()
            training_time = round(end_time - start_time, 2)
            app(training_time)
            st.success(model + ' trained')
        st.write('All Models trained')
        st.subheader('Performance')
    if st.button('Calculer les erreurs'):
        for model in st.session_state['liste_models']:
            y_pred = st.session_state[model].predict(st.session_state['X_test'])
            acc = round(mean_absolute_percentage_error(st.session_state['y_test'], y_pred) * 100,2)
            st.write(f"Modele : {model}, erreur : {acc} %")



       

        # Create a dataframe to store the accuracy and training time of each model
        data1 = {'Model': [], 'Accuracy': []}
        data2 = {'Model': [], 'Training Time': []}

        # Iterate over each model in the list
        for model in st.session_state['liste_models']:
            # Get the accuracy and training time for the current model
            y_pred = st.session_state[model].predict(st.session_state['X_test'])
            acc = round(mean_absolute_percentage_error(st.session_state['y_test'], y_pred) * 100, 2)
            training_time = st.session_state['times'][st.session_state['liste_models'].index(model)]

            # Add the model, accuracy, and training time to the dataframe
            data1['Model'].append(model)
            data2['Model'].append(model)
            data1['Accuracy'].append(acc)
            data2['Training Time'].append(training_time)

        # Convert the data dictionary to a dataframe
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        st.subheader('Model Performance')
        
        fig1 = px.scatter(df1, x='Model', y='Accuracy',size='Accuracy', title='Model Performance')

        st.plotly_chart(fig1)

        fig2 = px.bar(df2, x='Model', y='Training Time', title='Training Time')

        st.plotly_chart(fig2)

with tab3:
    st.subheader('Télécharger un modèle')

    if st.button('Télécharger les modèles'):
       

        for i in range(len(st.session_state['liste_models'])):
            model_name = st.session_state['liste_models'][i]
            download_model(st.session_state[model_name], model_name)



