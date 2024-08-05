import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle
from utils import generate_excel_download_link
from sklearn.preprocessing import StandardScaler


def get_scaled_data(input_data,scaler):
    X_scaled = scaler.transform(input_data)
    return X_scaled

scaler_file = None
model_file = None

st.title('Inferences')

tab1, tab2 = st.tabs(["Importer les données et le modèle", "Prédictions"])


##################################
###          Tab 1              ##
##################################

with tab1:

    ### Upload des données à inférer ###

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
            st.dataframe(st.session_state['new_data'])
        
    ### Upload du modèle ###

    st.subheader('Importer le modèle')
    model_file = st.file_uploader('Uploader votre modèle')

    if model_file is not None:

        try:
            ### Reconstruction du model et des scalers éventuels ###
            
            #model_file.seek(0)
            model_bytes = model_file.read()
            
            
            ################################################################################################
            # A FAIRE : UNE FONCTION POUR UNPACK LE PICKLE EN FONCTION DU NOMBRE DE MODELES ET LEUR NATURE #
            ################################################################################################
            output = pickle.loads(model_bytes)
            
            if type(output) == tuple:
                model_name = type(output[0].estimators_[0]).__name__
                output = pickle.loads(model_bytes)
                st.session_state['reconstructed_model'] = output[0]
                st.session_state['reconstructed_scaler_X'] = output[1]
                st.session_state['reconstructed_scaler_y'] = output[2]
                
                st.success("Modèle chargé avec succès.")

            elif type(output) != tuple:
                st.session_state['reconstructed_model'] = output
                st.success("Modèle chargé avec succès.")

            else:
                st.error("Erreur lors du chargement du modèle : le fichier n'a ni un objet regresseur, ni un tuple.")

          
            
        except Exception as e:
            print(e)
            st.error('Erreur lors du chargement du modèle')
        
     

        

##################################
###          Tab 2              ##
##################################
            
with tab2:
    if st.button('Démarrer l\'inférence'):

        st.info(st.session_state["reconstructed_model"])
        st.info(type(st.session_state["reconstructed_model"]).__name__)

        st.info(st.session_state["reconstructed_model"].estimators_[0])
        st.info(type(st.session_state["reconstructed_model"].estimators_[0]).__name__)

        ### Mise à l'échelle des données d'entrée ###
        st.session_state['new_data_scaled'] = get_scaled_data(st.session_state['new_data'],st.session_state['reconstructed_scaler_X'])

        ### Prédiction ###
        prediction_scaled = st.session_state['reconstructed_model'].predict(st.session_state['new_data_scaled'])
        st.info(f"prediction_scaled shape: {prediction_scaled.ndim}")

        ### Mise à l'échelle inverse des prédictions ###
        if prediction_scaled.ndim == 1:
            st.info('ICI ICI ICI')
            prediction_scaled = prediction_scaled.reshape(-1,1)
     
        prediction = st.session_state['reconstructed_scaler_y'].inverse_transform(prediction_scaled)

        concatenated_df = pd.concat([pd.DataFrame(st.session_state['new_data']), pd.DataFrame(prediction)], axis=1)
        generate_excel_download_link(concatenated_df, 'prediction')
    
        st.dataframe(concatenated_df)
