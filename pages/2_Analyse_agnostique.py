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

from utils import *



st.title('Modeles')

tab1, tab2, tab3 = st.tabs(["1", "2", "3"])

display_features = False
entrainement = False

# A Supprimer ###########################
st.session_state['data'] = pd.read_excel('Donnees.xlsx')
#######################################


#########################################
####         Tab 1                #######
#########################################

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


#########################################
####         Tab 2                #######
#########################################

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

#########################################
####         Tab 3                #######
#########################################

with tab3:
    st.subheader('Télécharger un modèle')

    if st.button('Télécharger les modèles'):
       

        for i in range(len(st.session_state['liste_models'])):
            model_name = st.session_state['liste_models'][i]
            download_model(st.session_state[model_name], model_name)



