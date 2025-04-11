import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ut import *

def split_input_output(input_data, inputs, outputs):
    X = input_data[inputs]
    y = input_data[outputs]
    return X, y

def get_labeled_data(input_data, labels):
    return input_data[labels]

def colorize_multiselect_options(color: str) -> None:
    rule = f".stMultiSelect div[data-baseweb='select'] span[data-baseweb='tag']{{background-color: {color};}}"
    st.markdown(f"<style>{rule}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Jeu de données")
st.header("Choix des données d\'entrainement")
            
#if 'data' in st.session_state:
        #del st.session_state.data

if 'data' not in st.session_state:
    st.session_state.data = None

if "drop_col" not in st.session_state:
    st.session_state.drop_col = ""
if "col_to_time" not in st.session_state:
    st.session_state.col_to_time = ""
if "col_to_float_money" not in st.session_state:
    st.session_state.col_to_float_money = ""
if "col_to_float_coma" not in st.session_state:
    st.session_state.col_to_float_coma = ""
if "separateur" not in st.session_state:
    st.session_state.separateur = ""
if "file_details" not in st.session_state:
    st.session_state.file_details = ""
    

##########################################
####    En tête de la page Dataset    ####
##########################################

st.session_state['separateur_de_champs_data'] = st.text_input(label = 'Caractère de séparateur de champs dans le dataset', value=";")
st.session_state['encoding'] = st.text_input(label = 'Encoding', value="latin1")
uploaded_file = st.file_uploader("", type=['csv','xlsx'],accept_multiple_files=False)

df = None
if uploaded_file is not None :
    try:
        st.session_state['data'] = pd.read_csv(uploaded_file,
                                                   encoding=st.session_state['encoding'], 
                                                   sep = st.session_state['separateur_de_champs_data']
                                                  )
    except:
        try:
            st.session_state['data'] = pd.read_excel(uploaded_file, 
                                                     engine='openpyxl', 
                                                     sep = st.session_state['separateur_de_champs_data'])
        except:
            
            st.error("Erreur lors du chargement du fichier")
            


st.session_state['uploaded'] = 0  
if st.session_state['data'] is not None:
    st.session_state.file_details = {
                                        "FileType": st.session_state['data'].attrs,
                                        "FileSize": st.session_state['data'].size
                                        }
    st.success('Fichier chargé avec succès !')
    st.session_state['uploaded'] = 1

    
    
    if sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) > 0:
        st.warning("Attention, il y a des valeurs manquantes dans le dataset. Elles ont été retirées.") #retirés / retirées ?
        df = st.session_state['data'].dropna()
        st.session_state['data'] = df
        st.write(' - Taille après suppression des valeurs manquantes:', st.session_state['data'].shape[0], 'lignes')
        #st.info(sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()))
    else:
        st.write(' - Taille:', st.session_state['data'].shape[0], 'lignes')

    st.markdown('<p class="section">Aperçu du jeu de données</p>', unsafe_allow_html=True)
    st.dataframe(st.session_state['data'].head())


col1_1, b_1, col2_1 = st.columns((1, 0.1, 1))
col1, b, col2 = st.columns((2.7, 0.2, 1))


dataset_choix = "Choisir un dataset personnel"
if dataset_choix == "Choisir un dataset personnel":
    

    #####################################
    ##     Choix Inputs / Outputs      ##
    #####################################


    #st.markdown("<p class='petite_section'>Choix des entrées et sorties : </p>", unsafe_allow_html=True)

    if st.session_state['uploaded'] == 1:
        with col1_1:
            ### Récupérer les labels des input et outputs 
            features = st.session_state['data'].columns
            liste = [str(_) for _ in features]

            # Background color for multiselect options
            colorize_multiselect_options("darkcyan")
            
            st.subheader("Choix des variables d'entrées")
            # inputs
            inputs = st.multiselect(label="Quelles sont les variables quantitatives:", options=liste, default=liste, placeholder='Chosir les variables d\'entrées')
            
            # Variables qualitatives
            if inputs is not None:
                one_hot_labels = st.multiselect(label="Quelles sont les variables qualitatives ?", options=liste, placeholder='Laissez vide si aucune')

                
            st.subheader("Choix des variables de sorties")        
            # outputs
            outputs = st.multiselect('Qulles sont les variables à prédire :', liste, placeholder='Chosir les variables à prédire')

    
            ##########################################
            ##    Initialisation des variables      ##
            ##########################################

            if st.button("Valider la saisie"):
                
                st.session_state['inputs'] = inputs
                st.session_state['outputs'] = outputs
                
                st.session_state['one_hot_labels'] = one_hot_labels
                #st.info('Variables qualitatives :'+str(st.session_state['one_hot_labels']))
                if len(st.session_state['one_hot_labels']) > 0:
                    st.session_state['all_inputs'] = inputs + one_hot_labels

                else :
                    st.session_state['all_inputs'] = inputs

                # Séparer X et y
                st.session_state['X_num'], st.session_state['y'] = split_input_output(st.session_state.data, 
                                                                                st.session_state['inputs'], 
                                                                                st.session_state['outputs'])
                

                
                # Créer des données scalées ( Créer les variables seesion_state 'scaler_X' et 'scaler_y')
                st.session_state['X_scaled'], st.session_state['y_scaled'] = scale_data(st.session_state['X_num'], 
                                                                                        st.session_state['y'])

                def get_labeled_data_fit_transform(input_data, labels):
                    encoder = OneHotEncoder()
                    X_encoded = encoder.fit_transform(input_data[labels])
                    X_encoded = X_encoded.toarray()
                    l = []
                    for _ in encoder.categories_:
                        l.append(np.array(_))

                    col = np.concatenate(l, axis=0)
                    X_encoded_df = pd.DataFrame(X_encoded, columns=col)
                    st.session_state['encoder'] = encoder
                    return X_encoded_df

                
                

                if len(st.session_state['one_hot_labels']) > 0:
                    st.session_state['X_labeled'] = get_labeled_data_fit_transform(st.session_state['data'], st.session_state['one_hot_labels'])
                    st.session_state['X_scaled'] = np.concatenate((st.session_state['X_scaled'], st.session_state['X_labeled']), axis=1)
                    st.session_state['X'] = np.concatenate((st.session_state['X_num'], st.session_state['X_labeled']), axis=1)

                elif len(st.session_state['one_hot_labels']) == 0:
                    st.session_state['X'] = st.session_state['X_num']


                
                X_train, X_test, y_train, y_test = train_test_split(st.session_state['X'], 
                                                                    st.session_state['y'], 
                                                                    test_size=0.15, 
                                                                    random_state=3)
                
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(st.session_state['X_scaled'],
                                                                                                st.session_state['y_scaled'],
                                                                                                test_size=0.15,
                                                                                                random_state=3)
                
                
                
                # st.session_state['X_scaled'] généré dans ut.py
                # st.session_state['y_scaled'] généré dans ut.py
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['X_train_scaled'] = X_train_scaled
                st.session_state['X_test_scaled'] = X_test_scaled
                st.session_state['y_train_scaled'] = y_train_scaled
                st.session_state['y_test_scaled'] = y_test_scaled



                st.success('Data preparation done')

        

        with col2_1:

            st.subheader('Explication :')
            st.write('Pour réaliser des entraînements de modèles de machine learning, il est essentiel de diviser les variables en deux catégories :')
            st.write('  - Les variables d\'entrée (ou features) qui servent à guider le modèle en lui fournissant les informations nécessaires pour apprendre les relations entre les données et les résultats attendus.')
            st.write('  - Les variables de sortie (ou target) qui représentent les réponses ou les valeurs à prédire par le modèle, constituant l’objectif de l’apprentissage supervisé.')
            st.write('\n')
            st.write('On peut diviser les variables d\'entrée en deux catégories :')
            st.write('  - Les variables quantitatives, qui sont numériques et permettent de représenter des valeurs mesurables (comme l\'âge, le salaire, ou la température).')
            st.write('  - Les variables qualitatives, qui sont catégorielles et servent à représenter des informations discrètes ou des classes (comme le genre, la catégorie d\'un produit, ou l\'état civil).')
                    
    

    if uploaded_file is None:
        with col1_1:
            st.info("Veuillez charger un dataset")

    #####################################
    ##    Options de preprocessing     ##
    #####################################

    if "data" in st.session_state:
        my_expander = st.expander(label="Options de preprocessing")
        with my_expander:
            with col1_1:
                st.write("##")
                st.write("##")
                st.write("##")
            st.write("##")

            



            ######################################
            ##     Transformation des colonnes  ##
            ######################################

            st.markdown("<p class='petite_section'>Modifications du dataset : </p>", unsafe_allow_html=True)
            col1_1, b_1, col2_1, c_1, col3_1 = st.columns((1, 0.2, 1, 0.2, 1)) 
            st.write("##")
            if st.session_state.data is not None:
                option_col_update = st.session_state.data.columns.tolist()

            with col1_1:
                #st.write('col1_1')
                pass

            with col2_1:
                pass
                #st.write('col2_1')
            with col3_1:
                pass
                #st.write('col3_1')
            with col1_1:
                if st.session_state.data is not None:
                    st.session_state.drop_col = st.multiselect(label='Retirer des colonnes',
                                                           options=option_col_update,
                                                           )

            with col1_1:
                for col in st.session_state["col_to_time"]:
                    try:
                        st.session_state.data[col] = pd.to_datetime(st.session_state.data[col])
                        st.success("Transformation de " + col + " effectuée !")
                    except:
                        st.error("Transformation impossible ou déjà effectuée")

            with col3_1:
                for col in st.session_state.col_to_float_coma:
                    try:
                        st.session_state.data[col] = st.session_state.data[col].apply(
                            lambda x: float(str(x).replace(',', '.')))
                        st.success("Transformation de " + col + " effectuée !")
                    except:
                        st.error("Transformation impossible ou déjà effectuée")
            with col1_1:
                for col in st.session_state["drop_col"]:
                    try:
                        st.session_state.data = st.session_state.data.drop(columns=col, axis=1)
                        st.success("Colonnes " + col + " supprimée !")
                        st.dataframe(st.session_state.data)
                    except:
                        st.error("Transformation impossible ou déjà effectuée")
            
            ######################################
            ##                EDA               ##
            ######################################

            with col1:
                pass
            

            ############################
            ##         EDA            ##
            ############################

            with col2:
                if st.session_state.data is not None:
                    pass
                    
                    
            ############################
            ##         Download       ##
            ############################

            if st.session_state.data is not None:
                generate_excel_download_link(st.session_state.data, 'dataset modifié')

    st.session_state.choix_dataset = "Vous avez choisi de selectionner votre dataset"

        

            
