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

def get_labeled_data(input_data, encoder):
    col_to_encode = input_data[encoder.feature_names_in_]
    X_encoded = encoder.transform(col_to_encode)
    X_encoded = X_encoded.toarray()
    return X_encoded

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
            st.success('Données chargées avec succès.')
        
    ### Upload du modèle ###

    st.subheader('Choix du modèle')
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
            ### Sortir les labels des inputs et outputs du modèle ###
            st.session_state['model_output_labels'] = output[-1]
            output = output[:-1]
            st.session_state['model_input_labels'] = output[-1]
            output = output[:-1]
            st.info(f"Ordre et noms des variables d\'entrées nécessaires sur le jeu de données à inférer :{st.session_state['model_input_labels']}")
            st.info(f"Ordre et noms des variables de sortie du modèle : {st.session_state['model_output_labels']}")



       
            if (type(output)==tuple):
                # Si il y a le OneHotEncoder dans le tuple
                
                if output[-1].__class__.__name__ == 'OneHotEncoder':
                    st.session_state['reconstructed_ohe'] = output[-1]
                    st.subheader('Les variables qualitatives et leurs états possibles connus de ce modèle sont :')
                    for i, category in enumerate(st.session_state['reconstructed_ohe'].categories_):
                        st.write(f"({st.session_state['reconstructed_ohe'].feature_names_in_[i]}): {category}")
                    output = output[:-1]
                else :
                    st.session_state['reconstructed_ohe'] = None
            
                if type(output[0]).__name__ == 'MLPRegressor':
                    model_name = 'MLPRegressor'
                elif type(output[0]).__name__ == 'Sequential':
               
                    model_name = 'Sequential' 
                    params = output[-1]                           # On récupère les paramètres du modèle
                    
                    st.subheader('Les hyperparamètres et configurations de ce modèle sont : ')
                    architecture = []
                    for i in range(len(params["nb_neurons"])):
                        architecture.append(params["nb_neurons"][i])
                        architecture.append(params["nb_activation"][i])
                    st.write(f"Nombre de neurones et fonction d\'activation par couche: {architecture}")
                    st.write(f"Optimiseur et taux d'apprentissage initial: {params['optimizer']} - {params['learning_rate_initiale']}")
                    st.write(f"fonction de perte et métriques: {params['loss']} - {params['metrics']}")
                    
                    st.write(f"Pourcentage de données de validation et de test: {params['valid_size']*100} - {params['test_size']*100} ")
                    st.write(f"scaler utilisé pour les données d'entrée: {params['scaler']}")
                    st.write(f"Nombre d'epochs pendant l\entraînement: ", params['epochs'])
                    output = output[:-1]
                else :
                    
                    model_name = type(output[0].estimators_[0]).__name__
                
                st.session_state['reconstructed_model'] = output[0]
                if model_name != 'RandomForestRegressor':
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

    ### Bouton pour choisir un modèle entraîné dans Modèle_Optimisé ###
    #if st.session_state['model'] is not None :
        #st.button('Choisir le modèle entraîné précédemment')
        
     

        

##################################
###          Tab 2              ##
##################################

            
with tab2:

    if st.button('Démarrer l\'inférence'):
        st.info(model_name)


        if model_name != 'RandomForestRegressor':

            ### Encodage des variables catégorielles ###
            if st.session_state['reconstructed_ohe'] is not None:
                st.session_state['X_ohe'] = get_labeled_data(st.session_state['new_data'],st.session_state['reconstructed_ohe'])
                st.dataframe(pd.DataFrame(st.session_state['X_ohe']))
                # Retrirer les colonnes catégorielles
                st.session_state['new_data_num'] = st.session_state['new_data'].drop(st.session_state['reconstructed_ohe'].feature_names_in_, axis=1)
            else:
                st.session_state['new_data_num'] = st.session_state['new_data']

            ### Mise à l'échelle des données d'entrée ###
            st.session_state['new_data_scaled'] = get_scaled_data(st.session_state['new_data_num'],st.session_state['reconstructed_scaler_X'])

            ### Prédiction ###
            if st.session_state['reconstructed_ohe'] is not None:
                st.session_state['new_data_pred'] = np.concatenate((st.session_state['new_data_scaled'],st.session_state['X_ohe']),axis=1)
            else:
                st.session_state['new_data_pred'] = st.session_state['new_data_scaled']
            prediction_scaled = st.session_state['reconstructed_model'].predict(st.session_state['new_data_pred'])
            st.info(f"prediction_scaled shape: {prediction_scaled.ndim}")

            ### Mise à l'échelle inverse des prédictions ###
            if prediction_scaled.ndim == 1:
                st.info('ICI ICI ICI')
                prediction_scaled = prediction_scaled.reshape(-1,1)
        
            prediction = st.session_state['reconstructed_scaler_y'].inverse_transform(prediction_scaled)
            st.dataframe(pd.DataFrame(prediction_scaled))

        else:
            prediction = st.session_state['reconstructed_model'].predict(st.session_state['new_data'])
        
        concatenated_df = pd.concat([pd.DataFrame(st.session_state['new_data']), pd.DataFrame(prediction)], axis=1)
        generate_excel_download_link(concatenated_df, 'prediction')
    
        st.dataframe(concatenated_df)
