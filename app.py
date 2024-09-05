import streamlit as st

def main():
    st.set_page_config(

        page_title ="Efectis AI GUI",
        page_icon=":fire:",
    )


    st.title('Main page')
    st.sidebar.info('Choisir une page')
    st.markdown('Bienvenue sur notre application Streamlit dédiée à la création de modèles de machine learning pour les problèmes de régression.  Cette plateforme vous permet de développer et d évaluer facilement des modèles de prédiction à partir de vos propres jeux de données.')
    st.markdown('### Expliques le projet et les pages gros nigaud:')
    st.write('-- Description du projet')
    st.write('-- Indications ')
    st.header('Pages')
    
    ### Import data
    st.subheader('Importation du jeu de données')
    st.write('-- Première étape si l\'on veut entraîner un modèle par la suite : il s\'agit d\'importer le jeu de données que l\'on souhaite utiliser sur la plateforme' )

    ### EDA
    st.subheader('EDA')
    st.write('-- EDA')

    ### Dummy model
    st.subheader(' Dummy Model')
    st.write('-- Dummy Model, modèles  (2eme étape)')

    ### Model Optimisé using Optuna
    st.subheader(' Model Optimisé')
    st.write('-- Model Optimisé')
    
    ### Sandbox
    st.subheader(' Sandbox')
    st.write('-- Sandbox')

    ### Inférence
    st.write('-- Inférences')
    st.write('-- Support')

if __name__ == '__main__':
    main()