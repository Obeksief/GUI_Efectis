import streamlit as st
import pandas as pd
import pickle 
import keras
import numpy as np
import os
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import base64
import traceback
#from utils import *
from ut import *
from sklearn.model_selection import cross_val_score
import optuna
import xgboost as xgb
import plotly.graph_objects as go
import math
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor
import plotly.express as px
import matplotlib.pyplot as plt

#############################################
###        Variables de session           ###
#############################################

st.session_state['spinner_text_modele_optimise'] = 'Optimisation des hyperparamètres...'


st.title('Page d\'entrainement spécifique')

#############################################
###            Onglets                    ###
#############################################

tab1, tab2, tab3 = st.tabs(["Aperçu des données de travail", "Entrainement d\'un modèle dédié", "Téléchargement du modèle"])

### Méta-données
# Modèle par défaut ( Celui qui a le mieux performé dans Analyse Agnostique )

#st.session_state['Model_opt'] = str_model

st.session_state['nb_trials'] = 1000
nb_trials = 1000

##############################################
###             Tab 1                      ###
##############################################

with tab1:
    
    col_1, col_2 = st.columns([2,1])
    col_1_1, col_1_2 = st.columns([1,1])

    
    if 'data' in st.session_state and st.session_state['data'] is not None:
        with col_1 :
            st.subheader('Données de travail')
            with col_1_1:
                st.write('Données d\'entrées ')
                if st.session_state['one_hot_labels'] is not None:
                    st.dataframe(st.session_state['data'][st.session_state['inputs'] + st.session_state['one_hot_labels']])
                else:
                    st.dataframe(st.session_state['data'][st.session_state['inputs']])
            with col_1_2:
                st.write('Données de sorties ')
                st.dataframe(st.session_state['data'][st.session_state['outputs']])

        with col_2:
            pass
    else :
        st.info('Veuillez charger un jeu de données')
                
                
##############################################
###             Tab 2                      ###
##############################################

with tab2:
 
    colo_1, colo_2 = st.columns([1,1])

    with colo_1:
        ### Choix du modèle
        st.subheader('Choix du modèle')
        model = st.selectbox('Choix du modèle', ['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'], placeholder='Choix du modèle')

        ### Choix uilisateur
        if 'afficher_radar_param_optim' not in st.session_state:
            st.session_state['afficher_radar_param_optim'] = False

        st.session_state['temps_max'] = st.number_input('Temps maximum d\'entrainement (en secondes)', min_value=10, max_value=10000, value=300)
        st.session_state['mape_tolerance'] = st.number_input('Arrêt anctipé, tolérance d\'eurreur en (%)', min_value=0, max_value=100, value=10)

    # colo_2 :: Information et documentation
    with colo_2:

        st.write('Sur cette page, vous pouvez optimiser les hyperparamètres de différents modèles de machine learning afin d\'améliorer leurs performances sur vos données. Trois éléments principaux sont à paramétrer pour réaliser cette optimisation :')
        st.write('- Choix du modèle : Sélectionnez le modèle de machine learning que vous souhaitez optimiser. Chaque modèle possède un ensemble spécifique d’hyperparamètres influençant son comportement et ses performances.')
        st.write('- Temps de recherche : Spécifiez la durée pendant laquelle l’algorithme d’optimisation peut explorer les combinaisons possibles d\'hyperparamètres. Un temps de recherche plus long permet d’explorer un plus grand nombre de configurations et peut potentiellement trouver des réglages plus performants.')
        st.write('- Seuil d\'arrêt anticipé : Ce seuil définit le niveau d\'erreur qu\'il vous paraît acceptable, si un des essais passe sous ce seuil l\'entraînement prendra fin immédiatement.')
     
    ###################################
    ##            XGBoost            ##
    ###################################

    if model == 'XGBoost':

        col_1, col_2 = st.columns([1,1])
        
        with col_1:
            # Paramétrages de base
            st.session_state['range_nbr_estimateurs'] = [50, 2000]
            st.session_state['range_max_depth'] = [2, 15]
            st.session_state['range_eta'] = [0.001, 0.5]
            st.session_state['range_min_child_weight'] = [1,3]

            # Choix utilisateurs 
            st.session_state['slider_range_nbr_estimateurs'] = st.session_state['range_nbr_estimateurs']
            st.session_state['slider_range_max_depth'] = st.session_state['range_max_depth']
            st.session_state['slider_range_eta'] = st.session_state['range_eta']
            st.session_state['slider_range_min_child_weight'] = st.session_state['range_min_child_weight']

          
            
            st.session_state['categories'] = ['nombre d\'estimateurs', 'profondeur maximale', 'taux d\'apprentissage', 'poids minimal des feuilles']

            if st.button('Valider les choix'):
                launch_optim_xgboost()
                st.session_state['type_model'] = 'XGBoost'

        with col_2:

            if st.session_state['afficher_radar_param_optim']:
                st.write(st.session_state['best_params'])
                st.write(type(st.session_state['best_params']))

                for key, value in st.session_state['best_params']:
                    print('Le paramètre :', key,'a la valeur', value )

                st.write(st.session_state['best_params'])

    ####################################
    ##        Neural Network          ##
    ####################################

    elif model == 'Neural Network':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            st.session_state['range_first_layer'] = [2,4,8,16,32,64, 128]
            st.session_state['range_second_layer'] = [2,4,8,16,32,64, 128]
            st.session_state['range_batch_size'] = [1,2,4,8,16,32]
            st.session_state['learning_rate_initial'] = [0.0001, 0.1]

            st.session_state['slider_range_first_layer'] = [2,256]
            st.session_state['slider_range_second_layer'] = [2,256]
            st.session_state['slider_range_batch_size'] = [1,50]
            st.session_state['nb_epoch'] = 500

         

            st.session_state['categories'] = ['nombre de neurones première couche', 
                                              'nombre de neurones seconde couche', 
                                              'taille de batch',
                                              'taux d\'apprentissage initial'] 

            if st.button('Valider les choix'):
                launch_optim_nn()



                st.session_state['type_model'] = 'Neural_network'

                    
                    
                    
        with col_2:
       
            if st.session_state['afficher_radar_param_optim']:
                st.write(st.session_state['afficher_radar_param_optim'])
                
                st.write(st.session_state['best_params'])
        
    ####################################
    ##        Random Forest           ##
    ####################################

    elif model == 'Random Forest':
        col_1, col_2 = st.columns([1,1])

        with col_1:
           

            st.session_state['range_n_estimators'] = [10,500]
            st.session_state['range_max_depth'] = [1,50]
            st.session_state['range_min_samples_split'] = [2,10]
            st.session_state['range_min_samples_leaf'] = [1,10]


            st.session_state['categories'] = ['number of estimators', 'max depth', 'min samples split', 'min samples leaf']

            

            if st.button('Valider les choix'):
                
                launch_optim_random_forest()
                st.session_state['type_model'] = 'Random Forest'

        with col_2:
            if st.session_state['afficher_radar_param_optim']:
                st.write(st.session_state['best_params'])

    ####################################
    ##        CatBoost                ##
    ####################################
       
    elif model == 'CatBoost':
        col_1, col_2 = st.columns([1,1])
    
        with col_1:
            st.session_state['range_n_iterations'] = [50,2000]
            st.session_state['range_learning_rate'] = [0.001, 0.5]
            st.session_state['range_depth'] = [1, 15]
            st.session_state['range_subsample'] = [0.05, 1]

            st.session_state['slider_range_n_iterations'] = st.session_state['range_n_iterations']
            st.session_state['slider_range_learning_rate'] = st.session_state['range_learning_rate']
            st.session_state['slider_range_depth'] = st.session_state['range_depth']
            st.session_state['slider_range_subsample'] = st.session_state['range_subsample']

         

            st.session_state['categories'] = ['number of iterations', 'learning rate', 'depth', 'subsample']

            if st.button('Valider les choix'):
                launch_optim_catboost()
                
                st.session_state['type_model'] = 'CatBoost'

                

        with col_2 :
            if st.session_state['afficher_radar_param_optim']:
                st.write(st.session_state['best_params'])

    else :
        st.write('Veuillez choisir un modèle')

    ##### Calcul sur l'ensemble test #####
    #####       Mono output          #####

    if st.session_state['afficher_radar_param_optim']:
        st.write('Erreur MAPE :', st.session_state['model_MAPE'],'%')
        if len(st.session_state['outputs']) > 1:
            #for i in range()
            st.write(type(st.session_state['model_MAPE_rawvalues']))
            st.write(st.session_state['model_MAPE_rawvalues'],'%')
        st.write('Erreur RMSE:', st.session_state['model_RMSE'])
        if len(st.session_state['outputs']) > 1:
            st.write(st.session_state['model_RMSE_rawvalues'])

        ##### Graphe de parité Mono output #####
        if len(st.session_state['outputs']) == 1:
            if st.session_state['type_model'] == 'Neural_network' or st.session_state['type_model'] == 'XGBoost' or st.session_state['type_model'] == 'CatBoost':

                y_test_scaled = np.array(st.session_state['y_test_scaled'])
                y_test = st.session_state['scaler_y'].inverse_transform(y_test_scaled.reshape(-1,1)).flatten()
                y_pred_scaled = np.array(st.session_state['model'].predict(st.session_state['X_test_scaled']))
                y_pred = st.session_state['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

            else:

                y_test = st.session_state['y_test'].to_numpy().flatten()
                y_pred = st.session_state['model'].predict(st.session_state['X_test']).flatten()


            

            # Tracer le graphe de parité
            fig = plt.figure(figsize=(8, 8))
            sns.scatterplot(x=y_test, y=y_pred, color='#1f77b4', alpha=0.7, edgecolor='k', s=80, marker="+")

            # Tracer la ligne y = x pour référence
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=1., label="Parité parfaite")

            # Ajouter les labels et le titre
            plt.xlabel('Valeurs réelles')
            plt.ylabel('Valeurs prédites')
            plt.title('Graphe de Parité: Valeurs Prédites vs Valeurs Réelles')

            # Ajouter le quadrillage
            plt.grid(True)

            # Afficher le graphe
            st.pyplot(fig)
        ##### Graphe de parité Multi output #####
        elif len(st.session_state['outputs']) > 1:
            if st.session_state['type_model'] == 'Random Forest':
                y_pred = st.session_state['model'].predict(st.session_state['X_test'])
                y_test = st.session_state['y_test'].to_numpy()
            else:
                pred = st.session_state['model'].predict(st.session_state['X_test_scaled'])
                y_test = st.session_state['scaler_y'].inverse_transform(st.session_state['y_test_scaled'])
                y_pred = st.session_state['scaler_y'].inverse_transform(pred)
            y_columns = st.session_state['outputs']

            for i in range(len(st.session_state['outputs'])):
                fig = plt.figure(figsize=(8, 8))
                sns.scatterplot(x=y_test[:, i], y=y_pred[:, i])
                plt.plot([min(y_test[:, i]), max(y_test[:, i])], [min(y_test[:, i]), max(y_test[:, i])], color='red', linestyle='--')
                plt.xlabel('Valeurs réelles')
                plt.ylabel('Valeurs prédites')
                plt.title(f'Graphe de parité pour la variable {y_columns[i]}')
                plt.grid(True)
                st.pyplot(fig)

        else :
            st.warning('Erreur lors de l\'affichage du graphe de parité')
        

##############################################
###             Tab 3                      ###
##############################################


with tab3:
    if st.button('Afficher le lien de téléchargement'):
        st.write(st.session_state['afficher_radar_param_optim'])
        if st.session_state['afficher_radar_param_optim']:
        
            st.write(st.session_state['type_model'])

            if len(st.session_state['one_hot_labels']) > 0:
                if st.session_state['type_model'] == 'Neural_network' or st.session_state['type_model'] == 'XGBoost' or st.session_state['type_model'] == 'CatBoost':
                
                    download_model_and_scalers_and_encoder_and_labels(st.session_state['model'],
                                           st.session_state['scaler_X'], 
                                           st.session_state['scaler_y'],
                                           st.session_state['encoder'], 
                                             st.session_state['all_inputs'],
                                                st.session_state['outputs'],
                                           st.session_state['type_model'])
                else:
                    
                    download_model_and_encoder_and_labels(st.session_state['model'],st.session_state['encoder'], st.session_state['type_model'])

            elif len(st.session_state['one_hot_labels']) == 0:
                if st.session_state['type_model'] == 'Neural_network' or st.session_state['type_model'] == 'XGBoost' or st.session_state['type_model'] == 'CatBoost':
                
                    download_model_and_scalers_and_labels(st.session_state['model'],
                                           st.session_state['scaler_X'], 
                                           st.session_state['scaler_y'], 
                                        
                                             st.session_state['all_inputs'],
                                                st.session_state['outputs'],
                                           st.session_state['type_model'])
                else:
                    
                    download_model_and_labels(st.session_state['model'],
                                              st.session_state['all_inputs'],
                                                st.session_state['outputs'],
                                                  st.session_state['type_model'])
                
            else :
                st.error('erreur')

        