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

tab1, tab2, tab3 = st.tabs(["Importer les données et le modèle", "Prédictions"])
goo = False

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
            model_file.seek(0)  
            reconstructed_model = pickle.load(model_file)
            goo = True
            st.write("Modèle chargé avec succès.")
        except Exception as e:
            st.write("Erreur lors du chargement du modèle :", e)
            try:
                model_file.seek(0)  
                reconstructed_model = pickle.load(model_file)
                st.write("Modèle chargé avec succès en utilisant pickle.")
                goo = True
            except Exception as e2:
                st.write("Erreur lors du chargement du modèle avec pickle:", e2)
    st.subheader('Si votre modèle est un réseau de neurones, veuillez importer le scaler utilisé pour l\'entrainement')
    scaler_file = st.file_uploader('Uploader votre scaler')

    if scaler_file is not None:
        try:
            scaler = pickle.load(scaler_file)
            st.write("Scaler chargé avec succès.")
        except Exception as e:
            st.write("Erreur lors du chargement du scaler:", e)
    
            
            
            
with tab2:
    if goo :
        if scaler_file is not None:
            st.session_state['new_data'] = get_cleaned_data(st.session_state['new_data'],scaler)
        prediction = reconstructed_model.predict(st.session_state['new_data'])
        concatenated_df = pd.concat([pd.DataFrame(st.session_state['new_data']), pd.DataFrame(prediction)], axis=1)
        generate_excel_download_link(concatenated_df, 'prediction')
        st.dataframe(prediction)
        #st.dataframe(concatenated_df)
