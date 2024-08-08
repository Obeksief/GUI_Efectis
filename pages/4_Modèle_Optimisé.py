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
from utils import *
from ut import *
from sklearn.model_selection import cross_val_score
import optuna
import xgboost as xgb
import plotly.graph_objects as go
import math
import catboost as cb
from sklearn.multioutput import MultiOutputRegressor

### CatBoost related functions
def objective_catboost(trial):
    ### Recehrche des hyperparamètres
    n_iterations = trial.suggest_int('n_iterations', st.session_state['range_n_iterations'][0], st.session_state['range_n_iterations'][1])
    learning_rate = trial.suggest_float('learning_rate', st.session_state['range_learning_rate'][0], st.session_state['range_learning_rate'][1])
    depth = trial.suggest_int('depth', st.session_state['range_depth'][0], st.session_state['range_depth'][1])
    subsample = trial.suggest_float('subsample', st.session_state['range_subsample'][0], st.session_state['range_subsample'][1])

    model = MultiOutputRegressor(cb.CatBoostRegressor(iterations=n_iterations,
                                    learning_rate=learning_rate,
                                    depth=depth,
                                    subsample=subsample,
                                    random_state=123))
    
    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    ### Score MAPE pour arrêt anticipé
    # Entraînement
    model = MultiOutputRegressor(cb.CatBoostRegressor(iterations=n_iterations,
                                    learning_rate=learning_rate,
                                    depth=depth,
                                    subsample=subsample,
                                    random_state=123))
    model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], verbose=False)

    # Erreur MAPE et RMSE
    mape_error = mean_absolute_percentage_error(st.session_state['y_test_scaled'], model.predict(st.session_state['X_test_scaled']))*100    

    y_pred_unscaled = st.session_state['scaler_y'].inverse_transform(model.predict(st.session_state['X_test_scaled']).reshape(-1,1))
    y_true_unscaled = st.session_state['scaler_y'].inverse_transform(st.session_state['y_test_scaled'])

    rmse_error = math.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))

    ### Sauvegarde du modèle et les scores RMSE et MAPE
    st.session_state['model'] = model
    st.session_state['model_MAPE'] = mape_error
    st.session_state['model_RMSE'] = rmse_error

    ### Arrêt anticipé
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar 
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement)
    

            
            
    return score.mean()

def launch_optim_catboost():
    ### Initialisation Progress bar 
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ###
    ### Intialisation du temps
    st.session_state['start_time'] = time.time()
    ###

    

    with st.spinner('Optimisation des hyperparamètres...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_catboost, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
        ### Progress bar
        st.session_state['my_bar'].empty()
        ###
        st.session_state['best_params'] = study.best_params
        st.session_state['afficher_radar_param_optim'] = True


### NN related functions

def objective_nn(trial):  
    
    ### Recehrche des hyperparamètres
    first_layer = trial.suggest_categorical('first_layer', st.session_state['range_first_layer'])
    second_layer = trial.suggest_categorical('second_layer', st.session_state['range_second_layer'])
    batch_size = trial.suggest_categorical('batch_size', st.session_state['range_batch_size'])
    learning_rate_initial = trial.suggest_float('learning_rate_initial', st.session_state['learning_rate_initial'][0],
                                                st.session_state['learning_rate_initial'][1])

    model = MLPRegressor(hidden_layer_sizes=[first_layer, second_layer],
                         batch_size=batch_size,
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate='adaptive',
                            learning_rate_init = learning_rate_initial,
                            beta_1=0.9,
                            beta_2=0.999,
                            max_iter=500,
                            validation_fraction=0.15,
                            early_stopping=True,
                            tol=0.0001,
                            n_iter_no_change=20)
    
    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    #### Score MAPE pour arrêt anticipé
    # Entraînement
    model = MLPRegressor(hidden_layer_sizes=[first_layer, second_layer],
                         batch_size=batch_size,
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate='adaptive',
                            learning_rate_init = learning_rate_initial,
                            beta_1=0.9,
                            beta_2=0.999,
                            max_iter=500,
                            validation_fraction=0.15,
                            early_stopping=True,
                            tol=0.0001,
                            n_iter_no_change=20)
    model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'])


    # Erreur MAPE et RMSE
    mape_error = mean_absolute_percentage_error(st.session_state['y_test_scaled'], model.predict(st.session_state['X_test_scaled']))*100

    y_pred_unscaled = st.session_state['scaler_y'].inverse_transform(model.predict(st.session_state['X_test_scaled']).reshape(-1,1))
    y_true_unscaled = st.session_state['scaler_y'].inverse_transform(st.session_state['y_test_scaled'])

    rmse_error = math.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))

    ### Sauvegarde du modèle et les scores RMSE et MAPE
    st.session_state['model'] = model
    st.session_state['model_MAPE'] = mape_error
    st.session_state['model_RMSE'] = rmse_error

    ### Arrêt anticipé
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar 
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement)
    #
    return score.mean()

def launch_optim_nn():
    ### Progress bar 
    progress_text = "Calcul en cours. Veuillez patienter."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ###
    ### Intialisation du temps
    st.session_state['start_time'] = time.time()
    ###

    with st.spinner('Optimisation des hyperparamètres...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_nn, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
        ### Vider la barre de progression
        st.session_state['my_bar'].empty()
        ###
        st.session_state['best_params'] = study.best_params 
        st.session_state['afficher_radar_param_optim'] = True
  
### xgboost related functions
def objective_xgboost(trial):
    # Hyperparamètres à optimiser
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_nbr_estimateurs'][0], st.session_state['range_nbr_estimateurs'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    eta = trial.suggest_float('eta', st.session_state['range_eta'][0], st.session_state['range_eta'][1])
    min_child_weight = trial.suggest_int('min_child_weight', st.session_state['range_min_child_weight'][0]  , st.session_state['range_min_child_weight'][1])

    # Création du modèle
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             eta=eta,
                             min_child_weight=min_child_weight,
                             random_state=123))

    # Score MSE en cross-validation 
    score = cross_val_score(model, st.session_state['X_train_scaled'], st.session_state['y_train_scaled'], cv=3, scoring='neg_mean_squared_error')

    # Score MAPE pour arrêt anticipé
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             eta=eta,
                             min_child_weight=min_child_weight,
                             random_state=123))
    model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'])
    mape_error = mean_absolute_percentage_error(st.session_state['y_test_scaled'], model.predict(st.session_state['X_test_scaled']))*100
    y_pred_unscaled = st.session_state['scaler_y'].inverse_transform(model.predict(st.session_state['X_test_scaled']).reshape(-1,1))
    y_true_unscaled = st.session_state['scaler_y'].inverse_transform(st.session_state['y_test_scaled'])
    rmse_error = math.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))

    st.session_state['model'] = model
    st.session_state['model_MAPE'] = mape_error
    st.session_state['model_RMSE'] = rmse_error
    if mape_error < st.session_state['mape_tolerance']:
        trial.study.stop()
        st.success('Arrêt anticipé')

    ### Progress bar
    temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
    if temps_chargement > 1:
        temps_chargement = 1
    st.session_state['my_bar'].progress(temps_chargement)
    ###
    return score.mean()

def launch_optim_xgboost():
    ### Initialisation Progress bar
    progress_text = "Calcul en cours. Veuillez patienter..."
    st.session_state['my_bar'] = st.progress(0, text=progress_text)
    ###
    ### Initialisation du temps
    st.session_state['start_time'] = time.time()
    ###

    with st.spinner('Optimisation des hyperparamètres...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_xgboost, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
        ### vider la barre de progression
        st.session_state['my_bar'].empty()
        ###
        st.session_state['best_params'] = study.best_params
        st.session_state['afficher_radar_param_optim'] = True

#######################################                                                
### Random Forest related functions ###
#######################################
def objective_random_forest(trial):
        ### Hyperparamètres à optimiser
        n_estimators = trial.suggest_int('n_estimators', st.session_state['range_n_estimators'][0], st.session_state['range_n_estimators'][1])
        max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
        min_samples_split = trial.suggest_int('min_samples_split', st.session_state['range_min_samples_split'][0], st.session_state['range_min_samples_split'][1])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', st.session_state['range_min_samples_leaf'][0], st.session_state['range_min_samples_leaf'][1])

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=123))

        ### Score MSE en cross-validation
        score = cross_val_score(model, st.session_state['X_train'], st.session_state['y_train'], cv=3, scoring='neg_mean_squared_error')

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=123))

        model.fit(st.session_state['X_train'], st.session_state['y_train'])
        y_pred = model.predict(st.session_state['X_test'])
        mape_error = mean_absolute_percentage_error(st.session_state['y_test'], y_pred)*100
        rmse_error = math.sqrt(mean_squared_error(st.session_state['y_test'], y_pred))

        st.session_state['model'] = model
        st.session_state['model_MAPE'] = mape_error
        st.session_state['model_RMSE'] = rmse_error
        if mape_error < st.session_state['mape_tolerance']:
            trial.study.stop()
            st.success('Arrêt anticipé')

        ### Progress bar
        temps_chargement = (time.time()-st.session_state['start_time'])/st.session_state['temps_max']
        if temps_chargement > 1:
            temps_chargement = 1
        st.session_state['my_bar'].progress(temps_chargement)
        ###
        return score.mean()

def launch_optim_random_forest():
        ### Initialisation Progress bar
        progress_text = "Calcul en cours. Veuillez patienter."
        st.session_state['my_bar'] = st.progress(0, text=progress_text)
        ###
        ### Initialisation du temps
        st.session_state['start_time'] = time.time()
        ###
        
      

        with st.spinner('Optimisation des hyperparamètres...'):
            study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
            study.optimize(objective_random_forest, n_trials=st.session_state['nb_trials'], timeout=st.session_state['temps_max'])
            ### Vider la barre de progression
            st.session_state['my_bar'].empty()
            ###
            st.session_state['best_params'] = study.best_params
            st.session_state['afficher_radar_param_optim'] = True
        return True


### Utilitary functions
def liste_puissance_de_deux(min, max):
    puissance = 1
    while puissance < min:
        puissance *= 2
    puissances = []
    while puissance <= max:
        puissances.append(puissance)
        puissance *= 2

    return puissances

def dizaines(debut, fin):
    dizaines_list = []

    if debut < 10:
        dizaines_list.append(debut)
        debut = 10

    dizaine = (debut // 10) * 10
    if dizaine < debut:
        dizaine += 10

    while dizaine <= fin:
        dizaines_list.append(dizaine)
        dizaine += 10

    return dizaines_list




st.title('Page d\'entrainement spécifique')

#############################################
###            Onglets                    ###
#############################################

tab1, tab2, tab3 = st.tabs(["Aperçu des données de travail", "Entrainement d\'un modèle dédié", "Téléchargement du modèle"])

### Méta-données
# Modèle par défaut ( Celui qui a le mieux performé dans Analyse Agnostique )

#st.session_state['Model_opt'] = str_model


##############################################
###             Tab 1                      ###
##############################################

with tab1:
    
    col_1, col_2, col_3 = st.columns([1,1,1])

    ### Hyperparamètres 
    st.session_state['nb_trials'] = 200

       
    if 'data' in st.session_state and st.session_state['data'] is not None:
        with col_1:
            st.subheader('Données d\'entrées ')
            st.dataframe(st.session_state['data'][st.session_state['inputs']])
        with col_2:
            st.subheader('Données de sorties ')
            st.dataframe(st.session_state['data'][st.session_state['outputs']])

        with col_3:
            st.write('ttttest')
                
                
##############################################
###             Tab 2                      ###
##############################################

with tab2:
 

    ### Choix du modèle
    st.subheader('Choix du modèle')
    model = st.selectbox('Choix du modèle', ['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'])

    ### Choix uilisateur
    if 'afficher_radar_param_optim' not in st.session_state:
        st.session_state['afficher_radar_param_optim'] = False
    st.session_state['temps_max'] = st.number_input('Temps maximum d\'entrainement (en secondes)', min_value=1, max_value=10000, value=300)
    st.session_state['mape_tolerance'] = st.number_input('Arrêt anctipé, tolérance d\'eurreur en (%)', min_value=0, max_value=100, value=5)

    
     
    ###################################
    ##            XGBoost            ##
    ###################################

    if model == 'XGBoost':

        col_1, col_2 = st.columns([1,1])
        
        with col_1:
            st.session_state['range_nbr_estimateurs'] = [50, 2000]
            st.session_state['range_max_depth'] = [2, 15]
            st.session_state['range_eta'] = [0.001, 0.5]
            st.session_state['range_min_child_weight'] = [1,3]
            st.session_state['nb_trials'] = 200
            
            st.session_state['categories'] = ['nombre d\'estimateurs', 'profondeur maximale', 'taux d\'apprentissage', 'poids minimal des feuilles']

            if st.button('Valider les choix'):
                launch_optim_xgboost()
                best_param = st.session_state['best_params']
                best_nb_estimators = best_param['n_estimators']
                best_max_depth = best_param['max_depth']
                best_eta = best_param['eta']
                best_min_child_weight = best_param['min_child_weight']
                ###### Faire une fonction
                best_model = xgb.XGBRegressor(n_estimators=best_nb_estimators,
                                             max_depth=best_max_depth,
                                             eta=best_eta,
                                             min_child_weight=best_min_child_weight,
                                             random_state=123)
                ######
                best_model.fit(st.session_state['X_train_scaled'], st.session_state['y_train_scaled'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'XGBoost'

        with col_2:

            if st.session_state['afficher_radar_param_optim']:

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
            ### A SUPPRIMER
            st.error('Ce modèle ne fonctionne pas')
            ##############

            st.session_state['range_n_estimators'] = [10,500]
            st.session_state['range_max_depth'] = [1,50]
            st.session_state['range_min_samples_split'] = [1,10]
            st.session_state['range_min_samples_leaf'] = [1,10]


            st.session_state['categories'] = ['number of estimators', 'max depth', 'min samples split', 'min samples leaf']

            

            if st.button('Valider choix'):
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

            st.session_state['nb_trials'] = 200

            st.session_state['categories'] = ['number of iterations', 'learning rate', 'depth', 'subsample']

            if st.button('Valiser choix'):
                launch_optim_catboost()
                best_param = st.session_state['best_params']
                best_n_iterations = best_param['n_iterations']
                best_learning_rate = best_param['learning_rate']
                best_depth = best_param['depth']
                best_subsample = best_param['subsample']
                best_model = cb.CatBoostRegressor(iterations=best_n_iterations,
                                               learning_rate=best_learning_rate,
                                               depth=best_depth,
                                               subsample=best_subsample,
                                               random_state=123)
                best_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'CatBoost'

                

        with col_2 :
            if st.session_state['afficher_radar_param_optim']:
                st.write(st.session_state['best_params'])

    if 'model_MAPE' in st.session_state and "model_RMSE" in st.session_state:
        st.write(st.session_state['model_MAPE'],'%')
        st.write(st.session_state['model_RMSE'])

##############################################
###             Tab 3                      ###
##############################################


with tab3:
    if st.button('Afficher le lien de téléchargement'):
        st.write(st.session_state['afficher_radar_param_optim'])
        if st.session_state['afficher_radar_param_optim']:
        
            st.write(st.session_state['type_model'])
            if st.session_state['type_model'] == 'Neural_network' or st.session_state['type_model'] == 'XGBoost' or st.session_state['type_model'] == 'CatBoost':
                st.write('yo la couille')
                download_model_and_scalers(st.session_state['model'],
                                           st.session_state['scaler_X'], 
                                           st.session_state['scaler_y'], 
                                           st.session_state['type_model'])
            else:
                st.write('flubim')
                download_model(st.session_state['model'], st.session_state['type_model'])

