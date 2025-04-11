import streamlit as st
from ut import *
import plotly.express as px

st.set_page_config(page_title='Clustering', layout='wide')
st.header('Etude de Clustering')

tab1, tab2, tab3 = st.tabs(['Données','Méthodes de clustering général','Méthode'])

##################################
##   tab1 : Données             ##
##################################

col_1, col_2 = st.columns([1, 3])



with tab1:
    ### Colonne 1 ###
    with col_1:
        if st.session_state['uploaded'] == 1:
            st.subheader('Données')
            st.write(st.session_state['data'].head(10))
            st.write(f"Nombre de lignes : {st.session_state['data'].shape[0]}")
            st.write(f"Nombre de colonnes : {st.session_state['data'].shape[1]}")
            st.write("Types de données :")
            st.write(st.session_state['data'].dtypes)
            st.write("Statistiques descriptives :")
            st.write(st.session_state['data'].describe())

    ### Colonne 2 ###

    ### Ajouter affichages des données classique ET affichage de série temporelle ###
    with col_2:
        if st.session_state['uploaded'] == 1:
            display_type = st.selectbox('Sélectionner le type d\'affichage', ['Classique','2'],index = None, placeholder="Choisir une option")
            st.subheader('Visualisation des données')

            if display_type == 'Classique':
                st.write("Sélectionner les colonnes à afficher :")
                cols = st.multiselect('Sélectionner les colonnes', st.session_state['data'].columns.tolist())
                if len(cols) > 0:
                    data = st.session_state['data']

                    if len(cols) == 1:
                        fig = px.histogram(data, x=cols[0],
                            width=900,
                            height=600)
                        st.plotly_chart(fig, use_container_width=True)

                    elif len(cols) == 2:
                        fig = px.scatter(
                            data,
                            x=cols[0],
                            y=cols[1],
                            title="Graphique 2D interactif",
                            width=900,
                            height=900
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif len(cols) == 3:
                        fig = px.scatter_3d(
                            data,
                            x=cols[0],
                            y=cols[1],
                            z=cols[2],
                            title="Graphique 3D interactif",
                            opacity=0.7,
                            width=900,
                            height=900)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.warning("Trop de dimensions sélectionnées. Veuillez en choisir 3 ou moins.")

            elif display_type == '2':
                data = st.session_state['outputs']

                st.write("Sélectionner les colonnes à afficher :")
                
                
               
                st.write(series_cols)
                st.dataframe(data)

                if len(series_cols) > 0:
                    fig = px.line(
                        data,
                        x=time_col,
                        y=data,
                        title="Série(s) temporelle(s)",
                        width=900,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Veuillez sélectionner au moins une série à afficher.")


            else : 
                st.warning("Sélectionner une option valide.")


##############################################
##   tab2 : Méthodes de clustering général  ##
##############################################