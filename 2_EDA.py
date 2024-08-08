import streamlit as st
from ut import *

st.set_page_config(page_title='Exploratory Data Analysis', layout='wide')
st.header('Exploratory Data Analysis')

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
    
else :
    st.info('Veuillez charger un jeu de données')




    