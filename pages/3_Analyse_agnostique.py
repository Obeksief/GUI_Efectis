import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import base64
import time
import matplotlib.pyplot as plt
import plotly.express as px
import base64


from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

from utils import *
from ut import *



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
            st.write('BLUBLUBLABLABLA')
    else : 
        st.error('Pas de données chargées')

    



#########################################
####         Tab 2                #######
#########################################

with tab2:
    data1 = {'Modèles': [], 'Erreur relative (en %)': []}
    data2 = {'Modèles': [], 'temps d\'entraînement': []}
    data3 = {'Modèles': [], 'Erreur RMSE': []}

    ### Choix des modèles à entraîner
    st.subheader('Choix des algorithmes')

    liste_models = st.multiselect(label='Choisir les algotithmes à entraîner', 
                                  options=['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'], 
                                  default=['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'])
    
    if st.button('Valider les choix'):

        st.session_state['liste_models'] = liste_models

        ### Création des modèles
        for model in liste_models:
            st.session_state[model] = globals()[f'create_{model.lower().replace(" ", "_")}']()
        

        ### Entraînement des modèles
        
        with st.spinner('Entraînement des modèles...'):
            # Calcul du temps d'entrainement
            st.session_state['times']= []
            app = st.session_state['times'].append

            # Boucle d'entrainement
            for model in liste_models:

                # clock start
                start_time = time.time()

                # Entrainement des modèles avec données mises à l'échelle 
                if model == 'Neural Network' or model == 'XGBoost' or model == 'CatBoost':
                    st.session_state[model] = globals()[f'train_{model.lower().replace(" ", "_")}'](st.session_state[model], 
                                                                                                    st.session_state['X_train_scaled'], 
                                                                                                    st.session_state['y_train_scaled'])
                    
                # Entrainement des autres modèles
                else:
                    st.session_state[model] = globals()[f'train_{model.lower().replace(" ", "_")}'](st.session_state[model], 
                                                                                                    st.session_state['X_train'], 
                                                                                                    st.session_state['y_train'])
                # clock end
                end_time = time.time()
                training_time = round(end_time - start_time, 2)
                app(training_time)

                # Modèle entraîné avec succès
                st.info(model + ' entraîné avec succès')
            st.success('Tous les modèles ont été entraînés avec succès')

            for model in st.session_state['liste_models']:
                # Ajout des temps d'entrainement dans le dictionnaire
                training_time = st.session_state['times'][st.session_state['liste_models'].index(model)]
                data2['Modèles'].append(model)
                data2['temps d\'entraînement'].append(training_time)

                ## A Retirer dès que possible
                ## Retrait du Temsp d'entrainement XGBoost si ce dernier est beaucoup plus long que CatBoost
                ## Car il met trop de temps une fois déployé
                pointeur_XGB = -1
                pointeur_Cat = -1
                for i in range(len(data2['Modèles'])):
                    if data2['Modèles'][i] == 'XGBoost':
                        pointeur_XGB = i
                    if data2['Modèles'][i] == 'CatBoost':
                        pointeur_Cat = i
                    i = i + 1
                if pointeur_XGB!= 0 and data2['temps d\'entraînement'][pointeur_XGB] > 20*data2['temps d\'entraînement'][pointeur_Cat]:
                    st.warning('CatBoost est plus rapide que XGBoost')
                    data2['Modèles'].pop(pointeur_XGB)
                    data2['temps d\'entraînement'].pop(pointeur_XGB)

                #######################

            st.subheader('Performances')
            
            
            # Clés de st.session_state créées dans le fichier de fonctions
            # st.session_state['scaler_X']
            # st.session_state['scaler_y']
            
            ### Test des modèles
            for model in st.session_state['liste_models']:
                i = 0
                # Test des modèles avec données mises à l'échelle
                if model == 'Neural Network' or model == 'XGBoost' or model == 'CatBoost':
                    y_pred = st.session_state[model].predict(st.session_state['X_test_scaled'])
                    y_pred_inverse_scaled = st.session_state['scaler_y'].inverse_transform(y_pred.reshape(-1, 1))
                    y_test_inverse_scaled = st.session_state['scaler_y'].inverse_transform(st.session_state['y_test_scaled'].reshape(-1, 1))

                    
                    mape_err = round(mean_absolute_percentage_error(st.session_state['y_test_scaled'], y_pred) * 100, 2)
                    rmse_err = round(root_mean_squared_error(st.session_state['y_test_scaled'], y_pred), 4) 
                    mape_err = round(mean_absolute_percentage_error(y_test_inverse_scaled, y_pred_inverse_scaled) * 100, 2)
                    rmse_err = round(np.sqrt(mean_squared_error(y_test_inverse_scaled, y_pred_inverse_scaled)), 4)

                # Test des autres modèles
                else:
                    y_pred = st.session_state[model].predict(st.session_state['X_test'])
                    mape_err = round(mean_absolute_percentage_error(st.session_state['y_test'], y_pred) * 100, 2)
                    rmse_err = round(np.sqrt(mean_squared_error(st.session_state['y_test'], y_pred)), 4)

                # Ajout des erreurs dans les dictionnaires
                data1['Modèles'].append(model)
                data1['Erreur relative (en %)'].append(mape_err)
                data3['Modèles'].append(model)
                data3['Erreur RMSE'].append(rmse_err)
                
                


       

      
        st.warning('Mémo à moi même : Mettre les formules MAPE et RMSE')

       
        df4 = pd.DataFrame(data3)
        
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)
        df3 = pd.merge(df1, 
                       df4, 
                       on='Modèles')
        df3 = pd.merge(df3, 
                       df2, 
                       on='Modèles')
        
        
        


        st.dataframe(df3,
                     hide_index=True)
        
        st.subheader('Erreurs en valeurs relatives des modèles sur l\'ensemble de test')
        ### Affichage graphique des erreurs relatives ( MAPE)
        fig1 = px.scatter(df1, 
                          x='Modèles', 
                          y='Erreur relative (en %)', 
                          title='Erreur de chaque modèle (en %)', 
                          color='Erreur relative (en %)', 
                          color_continuous_scale='RdYlGn_r')
        fig1.update_traces(marker=dict(size=30),selector=dict(mode='markers'))
        st.plotly_chart(fig1)

        st.subheader('Erreurs en valeurs absolues des modèles sur l\'ensemble de test')

        ### Affichage graphique des erreurs absolues (RMSE)
        fig3 = px.scatter(df4, 
                          x='Modèles', 
                          y='Erreur RMSE', 
                          title='Erreur de chaque modèle', 
                          color='Erreur RMSE', 
                          color_continuous_scale='RdYlGn_r')
        fig3.update_traces(marker=dict(size=30),selector=dict(mode='markers'))
        st.plotly_chart(fig3)           



        st.subheader('Temps d\'entrainement des modèles sur l\'ensemble d\'entrainement')
        ### Affichage graphique des temps d'entrainement
        fig2 = px.bar(df2, 
                      x='Modèles', 
                      y='temps d\'entraînement', 
                      title='Temps d\'entraînement de chaque modèle (en secondes)',
                      color='temps d\'entraînement',
                      color_continuous_scale='RdYlGn_r')

        st.plotly_chart(fig2)

#########################################
####            Tab 3                ####
#########################################

with tab3:
    st.subheader('Télécharger un modèle')

    if st.button('Télécharger les modèles'):
      
        for i in range(len(st.session_state['liste_models'])):
            model_name = st.session_state['liste_models'][i]
            if model_name == 'Neural Network' or model_name == 'XGBoost' or model_name == 'CatBoost':
                download_model_and_scalers(st.session_state[model_name],
                                           st.session_state['scaler_X'], 
                                           st.session_state['scaler_y'], 
                                           model_name)
            else :
                download_model(st.session_state[model_name], model_name)



