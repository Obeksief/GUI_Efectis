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
    st.write('\n')
    st.write('2) Créer et entraîner un modèle dans les onglets "Entraînement préliminaire", "Modèle Optimisé" et "Bac à sable"')
    st.write('\n')
    st.write('3) Faire des prédictions à l\'aide d\'un modèle téléchargé dans la partie 2) sur de nouvelles données dans l\'onglet "Inférences"')
    st.write('\n')

    st.header('Pages')
    
    ### Import data
    st.subheader('Importation du jeu de données')
    st.write('-- Première étape si l\'on veut entraîner un modèle par la suite : il s\'agit d\'importer le jeu de données que l\'on souhaite utiliser sur la plateforme' )

    ### EDA
    st.subheader('Analyse de données exploratoire') 
    st.write('-- Exploratory Data Analysis, quelques graphes d\'analyse statistique du jeu de données importé')

    ### Dummy model
    st.subheader('Entraînements préliminaires')
    st.write('-- Quelques modèles préconfigurés vont s\'entraîner sur le jeu de données importé pour avoir une idée des performances de base de chaque modèle')

    ### Model Optimisé using Optuna
    st.subheader(' Modèle Optimisé')
    st.write('-- Page dédié à l\entraînement d\'un modèle avec optimisation de ses paramètres')
    
    ### Sandbox
    st.subheader('Sandbox')
    st.write('-- Onglet 1 : Permet d\'affiner la recherche de paramètres d\'un modèle')
    st.write("-- Onglet 2 : Permet de construire un réseau de neurones de manière personnalisée")

    ### Inférence
    st.subheader('Inférences')
    st.write('-- Page dédiée à la prédiction de nouvelles données à l\'aide d\'un modèle téléchargé')
    
    st.subheader('Tickets')
    st.write('-- Si vous avez des questions, des problèmes ou bien même des propositions ( exemple : Il serait bien d\'ajouter telle fonctionnalité), n\'hésitez pas à me contacter par mail : kilian.cheix@efectis.com')

if __name__ == '__main__':
    main()