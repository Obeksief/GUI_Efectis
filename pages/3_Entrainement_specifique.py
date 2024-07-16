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

def objective(trial):
    first_hidden_layer = trial.suggest_categorical('first_hidden_layer', [2,4,8,16,32,64,128])
    second_hidden_layer = trial.suggest_categorical('second_hidden_layer', [2,4,8,16,32,64,128])
    alpha = trial.suggest_float('alpha', 0.0001, 0.01)

    
    model = MLPRegressor(hidden_layer_sizes=[first_hidden_layer, second_hidden_layer],
                         activation='relu',
                         solver='adam',
                         alpha=alpha,
                         learning_rate='adaptive',
                         max_iter=500,
                         validation_fraction=0.15,
                         early_stopping=True,)

    
    score = cross_val_score(model, st.session_state['X_scaled'], st.session_state['y'], cv=3, scoring='neg_mean_squared_error')
    
    return score.mean()


st.title('Training page')

if 'model' not in st.session_state:
    st.session_state['model'] = create_model()
    st.write(dir(st.session_state['model']))

tab1, tab2, tab3 = st.tabs(["importer les données", "choisir les entrées et les sorties", "performance"])


##############################################
###             Tab 1                      ###
##############################################

with tab1:
    st.subheader('Données utilisées :')
    if 'data' in st.session_state:
        st.dataframe(st.session_state.data)

    st.subheader('Définir les entrées et les sorties')
    if 'data' in st.session_state:
        features = st.session_state['data'].columns
        liste = [str(_) for _ in features]
        inputs = st.multiselect("Quel:", liste)
        outputs = st.multiselect('What are the outputs:', liste)

        # A Supprimer ############
        inputs = ['x1', 'x2', 'x3']
        outputs = ['y']
        st.write('Salut Kilian, valide simplement la saisie pour passer à l\'étape suivante et ne t\'embetes pas' )
        ##########################

        ###################################################
        ## Bric à brac ici si on permet plusieurs models ##
        ###################################################
        

        if st.button("valider la saisie"):
            X, X_scaled, y, scaler = get_cleaned_data(st.session_state.data, inputs, outputs)
            st.session_state['X_scaled'] = X_scaled
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['scaler'] = scaler
            st.write('Data preparation done')

            with st.spinner('Optimizing hyperparameters...'):
             
                study = optuna.create_study(direction='minimize', sampler= optuna.samplers.RandomSampler())

                study.optimize(objective, n_trials=5)
                
                st.session_state['best_params'] = study.best_params
                
##############################################
###             Tab 2                      ###
##############################################

with tab2:
    st.subheader('Choix des hyperparamètres')
    

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

