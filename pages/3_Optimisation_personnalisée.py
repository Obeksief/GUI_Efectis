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

### NN related functions
def get_radar_nn_optim():
    a_scaled = math.log2(st.session_state['best_params']['first_layer'])-1/math.log2(128)-1
    b_scaled = math.log2(st.session_state['best_params']['second_layer'])-1/math.log2(128)-1
    c_scaled = math.log10(st.session_state['best_params']['batch_size'])/math.log10(100)

    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)
    
    st.plotly_chart(fig_radar_optim)

def get_radar_nn_slider():
    min_a = st.session_state['slider_range_first_layer'][0]
    max_a = st.session_state['slider_range_first_layer'][1]
    min_b = st.session_state['slider_range_second_layer'][0]
    max_b = st.session_state['slider_range_second_layer'][1]
    min_c = st.session_state['slider_range_batch_size'][0]
    max_c = st.session_state['slider_range_batch_size'][1]

    min_a_prescaled = math.log2(min_a)
    max_a_prescaled = math.log2(max_a)
    min_b_prescaled = math.log2(min_b)
    max_b_prescaled = math.log2(max_b)


    min_a_scaled = min_a_prescaled/7
    max_a_scaled = max_a_prescaled/7
    min_b_scaled = min_b_prescaled/7
    max_b_scaled = max_b_prescaled/7
    min_c_scaled = math.log10(min_c)/math.log10(100)
    max_c_scaled = math.log10(max_c)/math.log10(100)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))
    
    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )
    
    st.plotly_chart(fig_radar)

def objective_nn(trial):  
    
    range_layer_1 = liste_puissance_de_deux(st.session_state['slider_range_first_layer'][0], st.session_state['slider_range_first_layer'][1])
    range_layer_2 = liste_puissance_de_deux(st.session_state['slider_range_second_layer'][0], st.session_state['slider_range_second_layer'][1])
    range_batch_size = dizaines(st.session_state['slider_range_batch_size'][0], st.session_state['slider_range_batch_size'][1])
    first_layer = trial.suggest_categorical('first_layer', range_layer_1)
    second_layer = trial.suggest_categorical('second_layer', range_layer_2)
    batch_size = trial.suggest_categorical('batch_size', range_batch_size)

    model = MLPRegressor(hidden_layer_sizes=[first_layer, second_layer],
                         batch_size=batch_size,
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            learning_rate='adaptive',
                            learning_rate_init=0.1,
                            beta_1=0.9,
                            beta_2=0.999,
                            max_iter=st.session_state['nb_epoch'],
                            validation_fraction=0.15,
                            early_stopping=True,
                            tol=0.0001,
                            n_iter_no_change=50)
    score = cross_val_score(model, st.session_state['X_scaled'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error')

    return score.mean()
  
### xgboost related functions
def get_radar_xgboost_optim():
    a_scaled = (st.session_state['best_params']['n_estimators'] -20)/(500-20)
    b_scaled = (st.session_state['best_params']['max_depth'] -1)/(15-1)
    c_scaled = (st.session_state['best_params']['eta'] -0.01)/(0.5-0.01)
    d_scaled = (st.session_state['best_params']['min_child_weight'] -1)/(10-1)
    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled,
                        d_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)

    st.plotly_chart(fig_radar_optim)

    st.markdown('#### Les hyperparamètres optimaux sont :')
    st.write('nombre d estimateurs :', st.session_state['best_params']['n_estimators'])
    st.write('profondeur maximale :', st.session_state['best_params']['max_depth'])
    st.write('taux d apprentissage :', round(st.session_state['best_params']['eta'],4))
    st.write('poids minimal des feuilles :', st.session_state['best_params']['min_child_weight'])

def get_radar_xgboost_slider():
    min_a = st.session_state['slider_range_nbr_estimateurs'][0]
    max_a = st.session_state['slider_range_nbr_estimateurs'][1]
    min_b = st.session_state['slider_range_max_depth'][0]
    max_b = st.session_state['slider_range_max_depth'][1]
    min_c = st.session_state['slider_range_eta'][0]
    max_c = st.session_state['slider_range_eta'][1]
    min_d = st.session_state['slider_range_min_child_weight'][0]
    max_d = st.session_state['slider_range_min_child_weight'][1]

    min_a_scaled = (min_a -20)/(500-20)
    max_a_scaled = (max_a -20)/(500-20)
    min_b_scaled = (min_b -1)/(15-1)
    max_b_scaled = (max_b -1)/(15-1)
    min_c_scaled = (min_c -0.01)/(0.5-0.01)
    max_c_scaled = (max_c -0.01)/(0.5-0.01)
    min_d_scaled = (min_d -1)/(10-1)
    max_d_scaled = (max_d -1)/(10-1)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled,
                        min_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled,
                        max_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))

    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )

    st.plotly_chart(fig_radar)

def objective_xgboost(trial):
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_nbr_estimateurs'][0], st.session_state['range_nbr_estimateurs'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    eta = trial.suggest_float('eta', st.session_state['range_eta'][0], st.session_state['range_eta'][1])
    min_child_weight = trial.suggest_int('min_child_weight', st.session_state['range_min_child_weight'][0]  , st.session_state['range_min_child_weight'][1])

    model = xgb.XGBRegressor(n_estimators=n_estimators,
                             max_depth=max_depth,
                             eta=eta,
                             min_child_weight=min_child_weight,
                             random_state=123)

    score = cross_val_score(model, st.session_state['X_scaled'], st.session_state['y_scaled'], cv=3, scoring='neg_mean_squared_error')


    ########### Early stopping test + Affichage de l'erreur relative test

    mape_score = cross_val_score(model, st.session_state['X_scaled'], st.session_state['y_scaled'], cv=3, scoring='neg_mean_absolute_percentage_error')
    

    if -mape_score.mean() < (st.session_state['mape_tolerance']/100):
        st.write('yo',st.session_state['mape_tolerance']/100,'gars sur',-mape_score.mean())
        st.write('erreur relative :', -mape_score.mean()*100)
        st.success('Arrêt anticipé')
        trial.study.stop()
    ############


    ############# Progress bar test
    error_displayed = -round(mape_score.mean()*100,2)
    st.session_state['my_bar'].progress(trial.number/st.session_state['nb_trials'], text=f'erreur : {error_displayed} %')
    #############
    
    return score.mean()

def launch_optim_xgboost():
                    ############# Progress bar test
                    progress_text = "Operation in progress. Please wait."
                    st.session_state['my_bar'] = st.progress(0, text=progress_text)
                    #############
                    st.session_state['range_nbr_estimateurs'] = st.session_state['slider_range_nbr_estimateurs']
                    st.session_state['range_max_depth'] = st.session_state['slider_range_max_depth']
                    st.session_state['range_eta'] = st.session_state['slider_range_eta']
                    st.session_state['range_min_child_weight'] = st.session_state['slider_range_min_child_weight']
                    st.session_state['nb_trials'] = nb_trial

                    with st.spinner('Optimisation des hyperparamètres...'):
                        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
                        study.optimize(objective_xgboost, n_trials=st.session_state['nb_trials'], timeout=st.session_state['timeout_tolerance'])
                        ############# Progress bar test
                        st.session_state['my_bar'].empty()
                        #############
                        st.session_state['best_params'] = study.best_params
                        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
                        st.session_state['afficher_radar_param_optim'] = True


### Random Forest related functions
def objective_random_forest(trial):
    n_estimators = trial.suggest_int('n_estimators', st.session_state['range_n_estimators'][0], st.session_state['range_n_estimators'][1])
    max_depth = trial.suggest_int('max_depth', st.session_state['range_max_depth'][0], st.session_state['range_max_depth'][1])
    min_samples_split = trial.suggest_int('min_samples_split', st.session_state['range_min_samples_split'][0], st.session_state['range_min_samples_split'][1])
    min_samples_leaf = trial.suggest_int('min_samples_leaf', st.session_state['range_min_samples_leaf'][0], st.session_state['range_min_samples_leaf'][1])

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=123)

    score = cross_val_score(model, st.session_state['X'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error')
    
    return score.mean()

def launch_optim_random_forest():
    st.session_state['range_n_estimators'] = st.session_state['slider_range_n_estimators']
    st.session_state['range_max_depth'] = st.session_state['slider_range_max_depth']
    st.session_state['range_min_samples_split'] = st.session_state['slider_range_min_samples_split']
    st.session_state['range_min_samples_leaf'] = st.session_state['slider_range_min_samples_leaf']
    st.session_state['nb_trials'] = nb_trial

    with st.spinner('Optimizing hyperparameters...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_random_forest, n_trials=st.session_state['nb_trials'])
        st.session_state['best_params'] = study.best_params
        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
        st.session_state['afficher_radar_param_optim'] = True

def get_radar_random_forest_slider():
    min_a = st.session_state['slider_range_n_estimators'][0]
    max_a = st.session_state['slider_range_n_estimators'][1]
    min_b = st.session_state['slider_range_max_depth'][0]
    max_b = st.session_state['slider_range_max_depth'][1]
    min_c = st.session_state['slider_range_min_samples_split'][0]
    max_c = st.session_state['slider_range_min_samples_split'][1]
    min_d = st.session_state['slider_range_min_samples_leaf'][0]
    max_d = st.session_state['slider_range_min_samples_leaf'][1]

    min_a_scaled = (min_a -10)/(500-10)
    max_a_scaled = (max_a -10)/(500-10)
    min_b_scaled = (min_b -1)/(50-1)
    max_b_scaled = (max_b -1)/(50-1)
    min_c_scaled = (min_c -2)/(10-2)
    max_c_scaled = (max_c -2)/(10-2)
    min_d_scaled = (min_d -1)/(10-1)
    max_d_scaled = (max_d -1)/(10-1)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled,
                        min_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled,
                        max_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))

    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )

    st.plotly_chart(fig_radar)

def get_radar_random_forest_optim():
    a_scaled = (st.session_state['best_params']['n_estimators'] -10)/(500-10)
    b_scaled = (st.session_state['best_params']['max_depth'] -1)/(50-1)
    c_scaled = (st.session_state['best_params']['min_samples_split'] -2)/(10-2)
    d_scaled = (st.session_state['best_params']['min_samples_leaf'] -1)/(10-1)
    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled,
                        d_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)

    st.plotly_chart(fig_radar_optim)

### CatBoost related functions
def get_radar_catboost_optim():
    a_scaled = (st.session_state['best_params']['n_iterations'] -50)/(1000-50)
    b_scaled = (st.session_state['best_params']['learning_rate'] -0.001)/(0.5-0.001)
    c_scaled = (st.session_state['best_params']['depth'] -1)/(15-1)
    d_scaled = (st.session_state['best_params']['subsample'] -0.05)/(1-0.05)
    fig_radar_optim = go.Figure()

    fig_radar_optim.add_trace(go.Scatterpolar(
    r=[ a_scaled,
                        b_scaled,
                        c_scaled,
                        d_scaled],
    theta=st.session_state['categories'],
    fill='toself',
    name='Hyperparamètres optimales'))

    fig_radar_optim.update_layout(
                        title='Hyperparamètres optimales',
                    polar=dict(
                            radialaxis=dict(
                            visible=True,
                            range=[0, 1])),
                        showlegend=False)

    st.plotly_chart(fig_radar_optim)

def get_radar_catboost_slider():
    min_a = st.session_state['slider_range_n_iterations'][0]
    max_a = st.session_state['slider_range_n_iterations'][1]
    min_b = st.session_state['slider_range_learning_rate'][0]
    max_b = st.session_state['slider_range_learning_rate'][1]
    min_c = st.session_state['slider_range_depth'][0]
    max_c = st.session_state['slider_range_depth'][1]
    min_d = st.session_state['slider_range_subsample'][0]
    max_d = st.session_state['slider_range_subsample'][1]

    min_a_scaled = (min_a -50)/(1000-50)
    max_a_scaled = (max_a -50)/(1000-50)
    min_b_scaled = (min_b -0.001)/(0.5-0.001)
    max_b_scaled = (max_b -0.001)/(0.5-0.001)
    min_c_scaled = (min_c -1)/(15-1)
    max_c_scaled = (max_c -1)/(15-1)
    min_d_scaled = (min_d -0.05)/(1-0.05)
    max_d_scaled = (max_d -0.05)/(1-0.05)

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
                    r=[ min_a_scaled,
                        min_b_scaled,
                        min_c_scaled,
                        min_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne minimum'
                ))
    fig_radar.add_trace(go.Scatterpolar(
                    r=[ max_a_scaled,
                        max_b_scaled,
                        max_c_scaled,
                        max_d_scaled],
                    theta=st.session_state['categories'],
                    fill='toself',
                    name='borne max'
                ))

    fig_radar.update_layout(
                title = 'Hyperparamètres',
                polar=dict(
                    radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                    )),
                showlegend=False
                )

    st.plotly_chart(fig_radar)

def objective_catboost(trial):
    n_iterations = trial.suggest_int('n_iterations', st.session_state['range_n_iterations'][0], st.session_state['range_n_iterations'][1])
    learning_rate = trial.suggest_float('learning_rate', st.session_state['range_learning_rate'][0], st.session_state['range_learning_rate'][1])
    depth = trial.suggest_int('depth', st.session_state['range_depth'][0], st.session_state['range_depth'][1])
    subsample = trial.suggest_float('subsample', st.session_state['range_subsample'][0], st.session_state['range_subsample'][1])

    model = cb.CatBoostRegressor(iterations=n_iterations,
                              learning_rate=learning_rate,
                              depth=depth,
                              subsample=subsample,
                              random_state=123)

    score = cross_val_score(model, st.session_state['X'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error', verbose=0)
    
    return score.mean()

def launch_optim_catboost():
    st.session_state['range_n_iterations'] = st.session_state['slider_range_n_iterations']
    st.session_state['range_learning_rate'] = st.session_state['slider_range_learning_rate']
    st.session_state['range_depth'] = st.session_state['slider_range_depth']
    st.session_state['range_subsample'] = st.session_state['slider_range_subsample']
    #st.session_state['nb_trials'] = nb_trial

    with st.spinner('Optimizing hyperparameters...'):
        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
        study.optimize(objective_catboost, n_trials=st.session_state['nb_trials'])
        st.session_state['best_params'] = study.best_params
        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
        st.session_state['afficher_radar_param_optim'] = True





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




st.title('Training page')

#############################################
###            Onglets                    ###
#############################################

tab1, tab2, tab3 = st.tabs(["Aperçu des données de travail", "Entrainement d\'un modèle dédié", "Téléchargement du modèle"])


##############################################
###             Tab 1                      ###
##############################################

with tab1:
    
    col_1, col_2, col_3 = st.columns([1,1,1])

       
    if 'data' in st.session_state and st.session_state['data'] is not None:
        with col_1:
            st.subheader('Données d\'entrées ')
            st.dataframe(st.session_state['data'][st.session_state['inputs']])
        with col_2:
            st.subheader('Données de sorties ')
            st.dataframe(st.session_state['data'][st.session_state['outputs']])

        with col_3:
            st.write('ttttest')


            #with st.spinner('Optimizing hyperparameters...'):
             
                #study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())

                #study.optimize(objective_xgboost, n_trials=st.session_state['nb_trials'])


                #st.session_state['best_params'] = study.best_params
                
                #optuna.visualization.plot_parallel_coordinate(study).show()
                #st.write(study.best_params)
                
                

##############################################
###             Tab 2                      ###
##############################################

with tab2:
    st.subheader('Choix du modèle')
    model = st.selectbox('Choix du modèle', ['Random Forest', 'Neural Network', 'XGBoost', 'CatBoost'])
    st.session_state['afficher_radar_param_optim'] = False




    st.subheader('Choix des hyperparamètres')
     
    ###################################
    ##            XGBoost            ##
    ###################################

    if model == 'XGBoost':

        col_1, col_2 = st.columns([1,1])
        
        with col_1:
            st.session_state['slider_range_nbr_estimateurs'] = st.slider('Nombre d\'estimateurs', value=[50, 200], step=10, min_value=20, max_value=500)
            st.session_state['slider_range_max_depth'] = st.slider('Profondeur maximale', value=[3, 10], step=1, min_value=1, max_value=15)
            st.session_state['slider_range_eta'] = st.slider('Taux d\'apprentissage', value=[0.01, 0.2], step=0.001, min_value=0.001, max_value=0.1)
            st.session_state['slider_range_min_child_weight'] = st.slider('Poids minimal des feuilles', value=[1, 3], step=1, min_value=1, max_value=10)
            
            nb_trial = st.number_input('Nombre d\'essais e combinaisons', min_value=1, max_value=1000, value=20)
            st.session_state['mape_tolerance'] = st.number_input('Erreur relative acceptée pour un arrêt anticipé ', value=10)
            st.session_state['timeout_tolerance'] = st.number_input('Temps maximum d\'exécution (en secondes)', value=600)
            

            st.session_state['categories'] = ['nombre d\'estimateurs', 'profondeur maximale', 'taux d\'apprentissage', 'poids minimal des feuilles']

            if st.button('Valider les choix'):
                launch_optim_xgboost()
                best_param = st.session_state['best_params']
                best_nb_estimators = best_param['n_estimators']
                best_max_depth = best_param['max_depth']
                best_eta = best_param['eta']
                best_min_child_weight = best_param['min_child_weight']
                best_model = xgb.XGBRegressor(n_estimators=best_nb_estimators,
                                             max_depth=best_max_depth,
                                             eta=best_eta,
                                             min_child_weight=best_min_child_weight,
                                             random_state=123)
                best_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'XGBoost'

        with col_2:
            get_radar_xgboost_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_xgboost_optim()
                st.write(st.session_state['best_params'])

    ####################################
    ##        Neural Network          ##
    ####################################

    elif model == 'Neural Network':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            
            ## Saisie des hyperparamètres de l'utilisateur ##
            st.session_state['slider_range_first_layer'] = st.select_slider('Premiere couche', options=[2,4,8,16,32,64,128], value=(8,64))
            st.session_state['slider_range_second_layer'] = st.select_slider('Seconde couche', options=[2,4,8,16,32,64,128], value=(2,32))
            st.session_state['slider_range_batch_size'] = st.select_slider('Taille des batches', options=[1,10, 20, 30, 40, 50,60,70,80,100], value=(10, 50))
            st.session_state['nb_epoch'] = st.number_input('Nombre d\'itération parmi les données d\entrainement', min_value=50, max_value=1000, value=500)
    

            nb_trial = st.number_input('Nombre d\'essais', min_value=1, max_value=1000, value=10)

            st.session_state['categories'] = ['nombre de neurones première couche', 'nombre de neurones seconde couche', 'taille de batch'] 

            if st.button('Valider les choix'):
                st.session_state['nb_trials'] = nb_trial
             

                with st.spinner('Optimisation des hyperparamètres...'):
                    study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
                    study.optimize(objective_nn, n_trials=st.session_state['nb_trials'])
                    
                    st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
                    st.session_state['afficher_radar_param_optim'] = True

                    st.session_state['best_params'] = study.best_params
                    best_first_layer = st.session_state['best_params']['first_layer']
                    best_second_layer = st.session_state['best_params']['second_layer']
                    best_batch_size = st.session_state['best_params']['batch_size']
                    best_model = MLPRegressor(hidden_layer_sizes=[best_first_layer, best_second_layer],
                                            batch_size=best_batch_size,
                                            activation='relu',
                                            solver='adam',
                                            alpha=0.0001,
                                            learning_rate='adaptive',
                                            learning_rate_init=0.1,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            max_iter=st.session_state['nb_epoch'],
                                            validation_fraction=0.15,
                                            early_stopping=True,
                                            tol=0.0001,
                                            n_iter_no_change=50)
                    best_model.fit(st.session_state['X_scaled'], st.session_state['y'])
                    st.session_state['model'] = best_model
                    st.session_state['type_model'] = 'Neural_network'

                    
                    
                    
        with col_2:
            get_radar_nn_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_nn_optim()
                st.write(st.session_state['best_params'])
        
    ####################################
    ##        Random Forest           ##
    ####################################

    elif model == 'Random Forest':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            st.session_state['slider_range_n_estimators'] = st.slider('Number of estimators', value=[50,200], min_value=10, max_value=500)
            st.session_state['slider_range_max_depth'] = st.slider('Max depth', value=[3, 10], min_value=1, max_value=50)
            st.session_state['slider_range_min_samples_split'] = st.slider('Min samples split', value=[2,5], min_value=2, max_value=10)
            st.session_state['slider_range_min_samples_leaf'] = st.slider('Min samples leaf', value=[1,5], min_value=1, max_value=10)

            nb_trial = st.number_input('Number of trials', min_value=1, max_value=1000, value=20)

            st.session_state['categories'] = ['number of estimators', 'max depth', 'min samples split', 'min samples leaf']

            if st.button('Validate choices'):
                launch_optim_random_forest()
                best_param = st.session_state['best_params']
                best_n_estimators = best_param['n_estimators']
                best_max_depth = best_param['max_depth']
                best_min_samples_split = best_param['min_samples_split']
                best_min_samples_leaf = best_param['min_samples_leaf']
                best_model = RandomForestRegressor(n_estimators=best_n_estimators,
                                                    max_depth=best_max_depth,
                                                    min_samples_split=best_min_samples_split,
                                                    min_samples_leaf=best_min_samples_leaf,
                                                    random_state=123)
                best_model.fit(st.session_state['X_train'], st.session_state['y_train'])
                st.session_state['model'] = best_model
                st.session_state['type_model'] = 'Random Forest'

        with col_2:
            get_radar_random_forest_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_random_forest_optim()
                st.write(st.session_state['best_params'])

    ####################################
    ##        CatBoost                ##
    ####################################

    elif model == 'CatBoost':
        col_1, col_2 = st.columns([1,1])

        with col_1:
            st.session_state['slider_range_n_iterations'] = st.slider('Number of iterations', value=[100,500], min_value=50, max_value=1000)
            st.session_state['slider_range_learning_rate'] = st.slider('Learning rate', value=[0.01, 0.1], min_value=0.001, max_value=0.5)
            st.session_state['slider_range_depth'] = st.slider('Depth', value=[3, 10], min_value=1, max_value=15)
            st.session_state['slider_range_subsample'] = st.slider('Subsample', value=[0.5, 0.8], min_value=0.05, max_value=1.)

            st.session_state['nb_trials'] = st.number_input('Number of trials', min_value=1, max_value=1000, value=20)

            st.session_state['categories'] = ['number of iterations', 'learning rate', 'depth', 'subsample']

            if st.button('Validate choices'):
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
            get_radar_catboost_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_catboost_optim()
                st.write(st.session_state['best_params'])

    

##############################################
###             Tab 3                      ###
##############################################

with tab3:
    if st.button('Valider la saisie'):
        if 'X_scaled' in st.session_state and 'y' in st.session_state:
            with st.spinner('Training model...'):
                if st.session_state['type_model'] == 'Neural_network':
                    error = round(train_model(st.session_state['model'], st.session_state['X_scaled'], st.session_state['y']),2)
                else :
                    y_hat = st.session_state['model'].predict(st.session_state['X_test'])
                    err = mean_absolute_percentage_error(st.session_state['y_test'], y_hat)*100
                    error = round(err,2)
            st.success('Done!')
            st.success(f'Erreur relative :{error}%')
            def download_model_and_scaler(model,scaler, file_name):
                file = (model, scaler)
                output = pickle.dumps(file)
                b64 = base64.b64encode(output).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="neural_network_{file_name}.pkl">Download Trained {file_name} Model</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            st.write(f"Mean Absolute Percentage Error: {error}%")

            if st.session_state['type_model'] == 'Neural Network':
                download_model_and_scaler(st.session_state['model'], st.session_state['scaler'], 'model_name')
            else:
                download_model(st.session_state['model'], st.session_state['type_model'])

