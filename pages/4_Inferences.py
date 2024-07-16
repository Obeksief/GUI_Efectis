import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle
from utils import generate_excel_download_link
from sklearn.preprocessing import StandardScaler


def get_cleaned_data(input_data,scaler):
    X_scaled = scaler.transform(input_data)
    return X_scaled

scaler_file = None
model_file = None

st.title('Inferences')

tab1, tab2 = st.tabs(["Importer les données et le modèle", "Prédictions"])
goo = False

##################################
###          Tab 1              ##
##################################

with tab1:
    st.subheader('Upload des données à inférer')
    new_data = st.file_uploader("Uploader un fichier Excel (.xlsx, .csv)")
    st.session_state['new_data'] = None
    if new_data is not None:
            try:
                df = pd.read_excel(new_data)
                display_features = True
            except Exception as e:
                print(e)
                df = pd.read_csv(new_data)
                
            st.session_state['new_data'] = df

    st.subheader('Importer le modèle')
    model_file = st.file_uploader('Uploader votre modèle')
    if model_file is not None:
        try:
            #model_file.seek(0)
            model_bytes = model_file.read()
            if model_file.name.startswith('neural_network'):
                output = pickle.loads(model_bytes)
                reconstructed_model = output[0]
                reconstructed_scaler = output[1] 

                st.write("Modèle réseau de neurones chargé avec succès.")
        except Exception as e:
            st.write("Erreur lors du chargement du modèle :", e)
     
        if st.button('Valider'):
            goo = True
        

##################################
###          Tab 2              ##
##################################
            
with tab2:
    if st.button('Démarrer l\'inférence'):
        if reconstructed_scaler is not None:
            st.session_state['new_data'] = get_cleaned_data(st.session_state['new_data'],reconstructed_scaler)
        prediction = reconstructed_model.predict(st.session_state['new_data'])
        concatenated_df = pd.concat([pd.DataFrame(st.session_state['new_data']), pd.DataFrame(prediction)], axis=1)
        generate_excel_download_link(concatenated_df, 'prediction')
        st.dataframe(prediction)
        #st.dataframe(concatenated_df)
