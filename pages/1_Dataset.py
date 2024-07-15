import streamlit as st
import pandas as pd
from utils import generate_excel_download_link


st.set_page_config(layout="wide", page_title="Dataset")
st.header("Choix des données d\'entrainement")
            
if 'data' in st.session_state:
        del st.session_state.data

if 'data' not in st.session_state:
    st.session_state.data = None

if "drop_col" not in st.session_state:
    st.session_state.drop_col = ""
if "col_to_time" not in st.session_state:
    st.session_state.col_to_time = ""
#if "col_to_float_money" not in st.session_state:
    #st.session_state.col_to_float_money = ""
if "col_to_float_coma" not in st.session_state:
    st.session_state.col_to_float_coma = ""
if "separateur" not in st.session_state:
    st.session_state.separateur = ""
if "file_details" not in st.session_state:
    st.session_state.file_details = ""
    
col1_1, b_1, col2_1 = st.columns((1, 0.1, 1))
col1, b, col2 = st.columns((2.7, 0.2, 1))

##########################################
####    En tête de la page Dataset    ####
##########################################


dataset_choix = "Choisir un dataset personnel"
if dataset_choix == "Choisir un dataset personnel":
    with col1_1:

        uploaded_file = st.file_uploader("", type=['xlsx'])

        df = None
        if uploaded_file is not None :
            st.session_state['data'] = pd.read_excel(uploaded_file)
            
        # uploaded_file = 0
        if st.session_state['data'] is not None:
            st.session_state.file_details = {"FileName": uploaded_file.name,
                                             "FileType": uploaded_file.type,
                                             "FileSize": uploaded_file.size}
            st.success('Fichier ' + st.session_state.file_details['FileName'] + ' chargé avec succès !')
        

    

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

            #####################################
            ##     Choix Inputs / Outputs      ##
            #####################################

            st.markdown("<p class='petite_section'>Choix des entrées et sorties : </p>", unsafe_allow_html=True)



            ######################################
            ##     Transformation des colonnes  ##
            ######################################

            st.markdown("<p class='petite_section'>Modifications du dataset : </p>", unsafe_allow_html=True)
            col1_1, b_1, col2_1, c_1, col3_1 = st.columns((1, 0.2, 1, 0.2, 1)) 
            st.write("##")
            if st.session_state.data is not None:
                option_col_update = st.session_state.data.columns.tolist()

            with col1_1:
                st.write('col1_1')

            with col2_1:
                st.write('col2_1')
            with col3_1:
                st.write('col3_1')
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

            with col1:
                if st.session_state.data is not None:
                    st.write("##")
                    st.markdown('<p class="section">Aperçu</p>', unsafe_allow_html=True)
                    st.write(st.session_state.data.head(50))
                    st.write("##")

            ############################
            ##         EDA            ##
            ############################

            with col2:
                if st.session_state.data is not None:
                    st.write("##")
                    st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
                    st.write(' - Taille:', st.session_state.data.shape)
                    st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
                    st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
                    st.write(' - Pourcentage de valeurs manquantes:', round(
                        sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                            ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
                    
            ############################
            ##         Download       ##
            ############################

            if st.session_state.data is not None:
                generate_excel_download_link(st.session_state.data, 'dataset modifié')

    st.session_state.choix_dataset = "Vous avez choisi de selectionner votre dataset"

        

            
