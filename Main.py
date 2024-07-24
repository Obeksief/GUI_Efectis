import streamlit as st

def main():
    st.set_page_config(

        page_title ="Efectis AI GUI",
        page_icon=":fire:",
    )


    st.title('Main page')
    st.sidebar.success('Choisir une page')
    st.markdown('### To Do List :')
    st.write('-- Description du projet')
    st.write('-- Indications ')
if __name__ == '__main__':
    main()