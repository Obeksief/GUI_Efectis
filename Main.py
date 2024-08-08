import streamlit as st

def main():
    st.set_page_config(

        page_title ="Efectis AI GUI",
        page_icon=":fire:",
    )


    st.title('Main page')
    st.sidebar.info('Choisir une page')
    st.markdown('### Expliques le projet et les pages gros nigaud:')
    st.write('-- Description du projet')
    st.write('-- Indications ')
    st.write('-- Importation du jeu de données ( 1ere étape)')
    st.write('-- EDA')
    st.write('-- Dummy Model, modèles  (2eme étape)')
    st.write('-- Model Optimisé')
    st.write('-- Sandbox')
    st.write('-- Inférences')
    st.write('-- Support')

if __name__ == '__main__':
    main()