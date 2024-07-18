import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import os
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

from utils import *



st.title('Modeles')

tab1, tab2, tab3 = st.tabs(["Aperçu des données de travail", "Choix des modèles à entraîner", "Téléchargement"])

display_features = False
entrainement = False




#########################################
####         Tab 1                #######
#########################################

with tab1:

    #################################
    ##    Aperçu des données       ##
    #################################
    col_1, col_2, col_3 = st.columns([1,1,1])
    if 'data' in st.session_state and st.session_state['data'] is not None:
        with col_1:
            st.subheader('Données d\'entrées ')
            st.dataframe(st.session_state['data'][st.session_state['inputs']])
        with col_2:
            st.subheader('Données de sorties ')
            st.dataframe(st.session_state['data'][st.session_state['outputs']])

        with col_3:
            st.subheader('Description')
            st.write('Ici plusieurs modèles de machine learning vont tenter de prédire les sorties à partir des entrées. ')
            st.write('  Les modèles seront ensuite comparés pour déterminer le plus performant. ')
    else : 
        st.write('Pas de données chargées')

    


    if st.button("valider la saisie"):

        pass


#########################################
####         Tab 2                #######
#########################################

with tab2:
    st.subheader('Choix du modele')
    liste_models = st.multiselect(label='Choisir les modeles', options=['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'], default=['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'])
    if st.button('Valider les choix'):
        st.session_state['liste_models'] = liste_models
        for model in liste_models:
            st.session_state[model] = globals()[f'create_{model.lower().replace(" ", "_")}']()
        st.write('Models created')
        st.session_state['times']= []
        app = st.session_state['times'].append
        with st.spinner('Training models...'):
            for model in liste_models:
                start_time = time.time()
                st.session_state[model] = globals()[f'train_{model.lower().replace(" ", "_")}'](st.session_state[model], st.session_state['X_train'], st.session_state['y_train'])
                end_time = time.time()
                training_time = round(end_time - start_time, 2)
                app(training_time)
                st.success(model + ' trained')
        st.success('All Models trained')
        st.subheader('Performance')
    if st.button('Calculer les erreurs'):
        for model in st.session_state['liste_models']:
            y_pred = st.session_state[model].predict(st.session_state['X_test'])
            acc = round(mean_absolute_percentage_error(st.session_state['y_test'], y_pred) * 100,2)
            st.write(f"Modele : {model}, erreur : {acc} %")



       

      
        data1 = {'Model': [], 'Accuracy': []}
        data2 = {'Model': [], 'Training Time': []}

       
        for model in st.session_state['liste_models']:
            
            y_pred = st.session_state[model].predict(st.session_state['X_test'])
            acc = round(mean_absolute_percentage_error(st.session_state['y_test'], y_pred) * 100, 2)
            training_time = st.session_state['times'][st.session_state['liste_models'].index(model)]

           
            data1['Model'].append(model)
            data2['Model'].append(model)
            data1['Accuracy'].append(acc)
            data2['Training Time'].append(training_time)

        
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



