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
from sklearn.model_selection import cross_val_score
import optuna
import xgboost as xgb
import plotly.graph_objects as go
import math

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
    st.write('taux d apprentissage :', round(st.session_state['best_params']['eta'],2))
    st.write('poids minimal des feuilles :', st.session_state['best_params']['min_child_weight'])

def get_radar_nn_slider():
    min_a = st.session_state['slider_range_first_layer'][0]
    max_a = st.session_state['slider_range_first_layer'][1]
    min_b = st.session_state['slider_range_second_layer'][0]
    max_b = st.session_state['slider_range_second_layer'][1]
    min_c = st.session_state['slider_range_batch_size'][0]
    max_c = st.session_state['slider_range_batch_size'][1]

    min_a_scaled = math.log2(min_a)-1/math.log2(128)-1
    max_a_scaled = math.log2(max_a)-1/math.log2(128)-1
    min_b_scaled = math.log2(min_b)-1/math.log2(128)-1
    max_b_scaled = math.log2(max_b)-1/math.log2(128)-1
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

    score = cross_val_score(model, st.session_state['X'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error')
    
    return score.mean()


st.title('Training page')

#if 'model' not in st.session_state:
    #st.session_state['model'] = create_model()
    #st.write(dir(st.session_state['model']))

tab1, tab2, tab3 = st.tabs(["importer les données", "choisir les entrées et les sorties", "performance"])


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

        if st.button("Confirmer les données d\'entrainement"):
            X, X_scaled, y, scaler = get_cleaned_data(st.session_state.data,st.session_state["inputs"] , st.session_state["outputs"])
            st.session_state['X_scaled'] = X_scaled
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['scaler'] = scaler
            st.write('Data preparation done')

            with st.spinner('Optimizing hyperparameters...'):
             
                study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())

                study.optimize(objective_xgboost, n_trials=st.session_state['nb_trials'])


                st.session_state['best_params'] = study.best_params
                
                optuna.visualization.plot_parallel_coordinate(study).show()

                st.write(study.best_params)
                
                

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
            st.session_state['slider_range_eta'] = st.slider('Taux d\'apprentissage', value=[0.01, 0.2], step=0.1, min_value=0.01, max_value=0.5)
            st.session_state['slider_range_min_child_weight'] = st.slider('Poids minimal des feuilles', value=[1, 3], step=1, min_value=1, max_value=10)
            
            nb_trial = st.number_input('Nombre d\'essais e combinaisons', min_value=1, max_value=1000, value=20)
            

            st.session_state['categories'] = ['nombre d\'estimateurs', 'profondeur maximale', 'taux d\'apprentissage', 'poids minimal des feuilles']

            if st.button('Valider les choix'):
                def launch_optim_xgboost():
                    st.session_state['range_nbr_estimateurs'] = st.session_state['slider_range_nbr_estimateurs']
                    st.session_state['range_max_depth'] = st.session_state['slider_range_max_depth']
                    st.session_state['range_eta'] = st.session_state['slider_range_eta']
                    st.session_state['range_min_child_weight'] = st.session_state['slider_range_min_child_weight']
                    st.session_state['nb_trials'] = nb_trial

                    with st.spinner('Optimisation des hyperparamètres...'):
                        study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
                        study.optimize(objective_xgboost, n_trials=20)
                        st.session_state['best_params'] = study.best_params
                        st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
                        st.session_state['afficher_radar_param_optim'] = True
                launch_optim_xgboost()

        with col_2:
            get_radar_xgboost_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_xgboost_optim()

    ####################################
    ##        Random Forest           ##
    ####################################

    elif model == 'Random Forest':
        pass

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
    

            nb_trial = st.number_input('Nombre d\'essais', min_value=1, max_value=1000, value=20)

            st.session_state['categories'] = ['nombre de neurones première couche', 'nombre de neurones seconde couche', 'taille de batch'] 

            if st.button('Valider les choix'):
                st.session_state['nb_trials'] = nb_trial
             

                with st.spinner('Optimisation des hyperparamètres...'):
                    study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())
                    study.optimize(objective_nn, n_trials=st.session_state['nb_trials'])
                    st.session_state['best_params'] = study.best_params
                    st.plotly_chart(optuna.visualization.plot_parallel_coordinate(study)) 
                    st.session_state['afficher_radar_param_optim'] = True
                    st.write('debug')
                    

        with col_2:
            get_radar_nn_slider()
            if st.session_state['afficher_radar_param_optim']:
                get_radar_nn_optim()
        pass
    elif model == 'CatBoost':
        pass

    

##############################################
###             Tab 3                      ###
##############################################

with tab3:
    if st.button('Valider la saisie'):
        if 'X_scaled' in st.session_state and 'y' in st.session_state:
            with st.spinner('Training model...'):
                error = round(train_model(st.session_state['model'], st.session_state['X_scaled'], st.session_state['y']),2)
            st.success('Done!')
            def download_model_and_scaler(model,scaler, file_name):
                file = (model, scaler)
                output = pickle.dumps(file)
                b64 = base64.b64encode(output).decode()
                href = f'<a href="data:file/output_model;base64,{b64}" download="neural_network_{file_name}.pkl">Download Trained {file_name} Model</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            st.write(f"Mean Absolute Percentage Error: {error}%")
            #download_model(st.session_state['model'], 'model_name')
            #download_scaler(st.session_state['scaler'])
            download_model_and_scaler(st.session_state['model'], st.session_state['scaler'], 'model_name')

