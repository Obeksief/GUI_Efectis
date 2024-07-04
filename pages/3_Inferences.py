import streamlit as st
from supervised import AutoML
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def get_cleaned_data(input_data,scaler):
    X_scaled = scaler.transform(input_data)
    return X_scaled

st.title('Inferences')

tab1, tab2, tab3 = st.tabs(["importer les données", "choisir le modèle ", "performance"])
goo = False

with tab1:
    st.subheader('Entrez les données d\'entraînement')
    new_data = st.file_uploader("Uploader un fichier Excel (.xlsx)")
    st.session_state['new_data'] = None
    if new_data is not None:
            try:
                df = pd.read_excel(new_data)
                display_features = True
            except Exception as e:
                print(e)
                df = pd.read_csv(new_data)
                
            st.session_state['new_data'] = df
                
with tab2:
    st.subheader(' Importer le modèle  ')
    if st.session_state['new_data'] is not None:
        features = st.session_state['new_data'].columns
        liste = [str(_) for _ in features._dir_additions_for_owner]
        st.write('en-tête de colonnes :', liste)
        
        model_file = st.file_uploader(' Uploader votre modèle ')
        
        if model_file is not None:
            try:
                # Try to load the model using TensorFlow
                reconstructed_model = load_model(model_file)
                goo = True
                st.write("Modèle chargé avec succès en utilisant TensorFlow.")
            except Exception as e:
                st.write("Erreur lors du chargement du modèle avec TensorFlow:", e)
                try:
                    # Fallback to load the model using pickle
                    model_file.seek(0)  # Reset file pointer to the beginning
                    reconstructed_model = pickle.load(model_file)
                    st.write("Modèle chargé avec succès en utilisant pickle.")
                    goo = True
                except Exception as e2:
                    st.write("Erreur lors du chargement du modèle avec pickle:", e2)
                    
            
            
            
with tab3:
    if goo :
        prediction = reconstructed_model.predict(st.session_state['new_data'])
        st.dataframe(prediction)
    st.write('test')
    
    
        