import streamlit as st
from ut import *

st.set_page_config(page_title='Exploratory Data Analysis', layout='wide')
st.header('Exploratory Data Analysis')

tab1, tab2, tab3 = st.tabs(['Apercu général','Distribution des variables','matrice de correlation '])

##################################
##   tab1 : Caractéristiques    ##
##################################

with tab1:
    col_1, col_2 = st.columns([1,1])
    if st.session_state.data is not None:
        with col_1:
            st.write("##")
            st.markdown('<p class="section">Caractéristiques</p>', unsafe_allow_html=True)
            st.write(' - Taille:', st.session_state.data.shape)
            st.write(' - Nombre de valeurs:', st.session_state.data.shape[0] * st.session_state.data.shape[1])
            st.write(' - Type des colonnes:', st.session_state.data.dtypes.value_counts())
            st.write(' - Pourcentage de valeurs manquantes:', round(
                                sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()) * 100 / (
                                        st.session_state.data.shape[0] * st.session_state.data.shape[1]), 2),
                                    ' % (', sum(pd.DataFrame(st.session_state.data).isnull().sum(axis=1).tolist()), ')')
        with col_2:  
            st.subheader('Statistiques descriptives')
            st.write(st.session_state.data.describe())
        
    else :
        st.info('Veuillez charger un jeu de données')

##################################
##   tab2 : Distribution        ##
##################################


with tab2:
    if st.session_state.data is not None:
        
        sns.set_theme(style="darkgrid")
        for i, column in enumerate(st.session_state.data.columns):
            with st.container():
                fig =plt.figure(figsize=(8, 6))
                sns.histplot(data=st.session_state.data, x=st.session_state.data[column], kde=False, bins=100) 
                plt.title(f'Distribution de {column}')
                plt.xlabel(column)
                plt.ylabel('Fréquence')
                st.pyplot(fig)

    else :
        st.info('Veuillez charger un jeu de données')

###################################
##   tab3 : Correlation          ##
###################################

with tab3:
    if st.session_state.data is not None:
        
        


        # Filtrer uniquement les colonnes numériques
        numeric_data = st.session_state['data'].select_dtypes(include=['float64', 'int64'])

        # Calculer la matrice de corrélation
        corr_matrix = numeric_data.corr()

        st.subheader("Matrice de corrélation des variables numériques")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={'shrink': .8}, ax=ax)
        st.pyplot(fig)

    else :
        st.info('Veuillez charger un jeu de données')

        

    





    