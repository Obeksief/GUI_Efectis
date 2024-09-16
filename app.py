import streamlit as st

def main():
    st.set_page_config(

        page_title ="Efectis AI GUI",
        page_icon=":fire:",
    )


    st.title('Main page')
    st.sidebar.info('Choisir une page')
    st.markdown('Bienvenue sur notre application Streamlit dédiée à la création de modèles de machine learning pour les problèmes de régression.  Cette plateforme vous permet de développer et d\'évaluer facilement des modèles de prédiction à partir de vos propres jeux de données.')
   

    st.write('-- Indications ')
    st.write('L\'application se divise en 3 parties :')
    st.write('1) Importer les données d\'entraînement dans l\'onglet "Importation des données"')
    st.write('2) Créer et entraîner un modèle dans les onglets "Entraînement préliminaire", "Modèle Optimisé" et "Bac à sable"')
    st.write('3) Faire des prédictions à l\'aide d\'un modèle téléchargé dans la partie 2) sur de nouvelles données dans l\'onglet "Inférences"')

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