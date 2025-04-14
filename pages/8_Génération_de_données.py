import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
from utils import generate_excel_download_link

from typing import List

def generate_cumulative_adjusted_normal_sequence(moyenne: float, ecart_type: float, taille: int) -> List[float]:
    sequence = [0]
    for _ in range(taille - 1):
        tirage = np.random.normal(moyenne, ecart_type)
        tentative = sequence[-1] + tirage
        if tentative < 0:
            rebond = abs(tentative)
            prochaine_valeur = rebond
        else:
            prochaine_valeur = tentative
        sequence.append(prochaine_valeur)
    return sequence

if 'data_généré' not in st.session_state:
    st.session_state['data_généré'] = False


col1, col2 = st.columns([1, 1])

with col1:
    with st.container(border=True):
        st.write("Création des colonnes indépendantes")
        nbr_colonnes = st.number_input("Nombre de colonnes", min_value=1, value=1, step=1)

        col_config = []

        for i in range(nbr_colonnes):
            with st.expander(f"Colonne {i+1}", expanded=False):
                col_name = st.text_input(f"Nom de la colonne {i+1}", key=f"col_name_{i}")
                col_type = st.selectbox(f"Type de données de la colonne {i+1}", ["Entier", "Flottant", "Catégorielle", "Série temporelle"], key=f"col_type_{i}")

                col_info = {"name": col_name, "type": col_type}

                if col_type == "Catégorielle":
                    categories = st.text_input(f"Valeurs possibles (séparées par des virgules)", key=f"categories_{i}")
                    col_info["categories"] = [x.strip() for x in categories.split(",") if x.strip()]
                elif col_type == "Série temporelle":
                    col_info["series_length"] = st.number_input("Longueur de la série", min_value=2, value=10, key=f"series_len_{i}")
                    col_info["mean"] = st.number_input("Moyenne", key=f"mean_series_{i}")
                    col_info["std"] = st.number_input("Écart-type", value=1.0, key=f"std_series_{i}")
                else:
                    distribution = st.selectbox(f"Loi de distribution", ["Uniforme", "Normale"], key=f"dist_{i}")
                    col_info["distribution"] = distribution

                    if distribution == "Normale":
                        col_info["mean"] = st.number_input(f"Moyenne", key=f"mean_{i}")
                        col_info["std"] = st.number_input(f"Écart-type", value=1.0, key=f"std_dev_{i}")
                    else:
                        col_info["min"] = st.number_input(f"Valeur min", key=f"min_val_{i}")
                        col_info["max"] = st.number_input(f"Valeur max", value=1.0, key=f"max_val_{i}")

                col_config.append(col_info)

    with st.container(border=True):
        st.write("Création des colonnes dépendantes")

    with st.container(border=True):
        st.subheader("Génération du dataset")

        nbr_lignes = st.number_input("Nombre de lignes", min_value=1, value=100, step=1)

        if st.button("Générer le dataset"):
            st.session_state["data_généré"] = True
            df = pd.DataFrame()

            for col in col_config:
                if col["type"] == "Catégorielle":
                    df[col["name"]] = np.random.choice(col["categories"], size=nbr_lignes)

                elif col["type"] == "Série temporelle":
                    for j in range(col["series_length"]):
                        df[f"{col['name']}_{j+1}"] = [
                            generate_cumulative_adjusted_normal_sequence(col["mean"], col["std"], col["series_length"])[j]
                            for _ in range(nbr_lignes)
                        ]

                else:
                    if col["distribution"] == "Uniforme":
                        if col["type"] == "Entier":
                            df[col["name"]] = np.random.randint(col["min"], col["max"] + 1, size=nbr_lignes)
                        else:  # Flottant
                            df[col["name"]] = np.random.uniform(col["min"], col["max"], size=nbr_lignes)
                    else:  # Normale
                        vals = np.random.normal(loc=col["mean"], scale=col["std"], size=nbr_lignes)
                        if col["type"] == "Entier":
                            df[col["name"]] = vals.astype(int)
                        else:
                            df[col["name"]] = vals

            st.success("✅ Dataset généré avec succès !")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.write("Télécharger le dataset au format excel")
            generate_excel_download_link(df, 'data')

            st.write("Télécharger le dataset au format CSV")
            st.download_button("📅 Télécharger en CSV", csv, "dataset.csv", "text/csv")

with col2:
    if st.session_state["data_généré"] == True:
        st.subheader("Visualisation des distributions")
        for col in col_config:
            if col["type"] == "Série temporelle":
                st.caption(f"Exemple de série pour '{col['name']}'")
                series = generate_cumulative_adjusted_normal_sequence(col["mean"], col["std"], col["series_length"])
                st.line_chart(series, height=200, use_container_width=True)
                
            else:
                st.caption(f"Distribution de '{col['name']}'")
                st.bar_chart(df[col["name"]].value_counts().sort_index(), height=200, use_container_width=True)
                

