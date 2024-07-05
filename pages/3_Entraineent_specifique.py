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

def get_cleaned_data(input_data, inputs, outputs):
    X = input_data[inputs]
    y = input_data[outputs]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def create_model():
    mini_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='gelu'),
        tf.keras.layers.Dense(32, activation='gelu'),
        tf.keras.layers.Dense(1)
    ])
    mini_model.compile(loss=tf.keras.losses.mse,
                       optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.012),
                       metrics=['mse'])
    
    return mini_model

def train_model(model, X_scaled, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=3)
        
    # Debugging: Print shapes of train and test sets
    st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
    model.fit(X_train, y_train, epochs=100, validation_split=0.15, shuffle=True, verbose=0)
    
    
    y_pred = model.predict(X_test)
    st.write(f"y_pred shape: {y_pred.shape}")
        
    acc = mean_absolute_percentage_error(y_test, y_pred) * 100
    return acc

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="model.pkl">Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_scaler(scaler):
    output_scaler = pickle.dumps(scaler)
    b64 = base64.b64encode(output_scaler).decode()
    href = f'<a href="data:file/output_scaler;base64,{b64}" download="scaler.pkl">Download scaler .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)

st.title('Training page')

if 'model' not in st.session_state:
    st.session_state['model'] = create_model()
    st.write(dir(st.session_state['model']))

tab1, tab2, tab3 = st.tabs(["importer les données", "choisir les entrées et les sorties", "performance"])

with tab1:
    st.subheader('Données utilisées :')
    if 'data' in st.session_state:
        st.dataframe(st.session_state.data)

    st.subheader('Définir les entrées et les sorties')
    if 'data' in st.session_state:
        features = st.session_state['data'].columns
        liste = [str(_) for _ in features]
        inputs = st.multiselect("What are the inputs:", liste)
        outputs = st.multiselect('What are the outputs:', liste)

        ###################################################
        ## Bric à brac ici si on permet plusieurs models ##
        ###################################################
        

        if st.button("valider la saisie"):
            X_scaled, y, scaler = get_cleaned_data(st.session_state.data, inputs, outputs)
            st.session_state['X_scaled'] = X_scaled
            st.session_state['y'] = y
            st.session_state['scaler'] = scaler
            st.write('Data preparation done')

with tab2:
    st.write('yo')
    


with tab3:
    if st.button('Valider la saisie'):
        if 'X_scaled' in st.session_state and 'y' in st.session_state:
            error = train_model(st.session_state['model'], st.session_state['X_scaled'], st.session_state['y'])
            st.write(f"Mean Absolute Percentage Error: {error}%")
            download_model(st.session_state['model'])
            download_scaler(st.session_state['scaler'])

