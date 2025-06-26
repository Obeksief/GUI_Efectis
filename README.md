# Interface Graphique de Regression

## Description
Ce projet consiste en une interface graphique (GUI) intuitive qui permet aux utilisateurs, même sans compétences avancées en programmation, de créer et entraîner facilement des modèles de machine learning pour des tâches de régression. Grâce à cette interface, l'utilisateur peut charger ses données, configurer divers paramètres d'entraînement (choix du modèle, sélection des features, etc.), visualiser les résultats et évaluer les performances du modèle.

## To do list

- [x] Ajouter des architectures de réseaux de neurones dans la partie "Entraînements préliminaires"
- [ ] Rattacher au GPU les calculs éligibles 
- [ ] Faire un graphique par output car les graphiques d'erreurs sont surchargés et illisibles
- [ ] Faire une video tutoriel
- [ ] Ajouter méthode de clustering pour faire un modèle par cluster
- [ ] Ajout de traitement des données temporelles
- [ ] Ajout d'outils de feature importance (SHAP, explicabilité de model)
- [ ] Ajouter Kolmogorov Arnorld Network 
- [ ] Regression Symobolique
- [ ] Ajouter transformation de série temporelle en données de regression
- [ ] Ajouter un outil permettant de créer un jeu de données artificielles (quantitative, qualitative, série temporelle)



## Lancer localement : Installation d'un environnement virtuel avec venv

### Prérequis
- Python 3.11 

Pour les systèmes Unix/MacOS :  
```python3 -m venv env```  
Pour Windows  
```python -m venv env```  

### Pour les systèmes Unix/MacOS
```source env/bin/activate```

### Pour Windows

```.\env\Scripts\activate```

### Installer les packages
```pip install -r requirements.txt```


### Utilisation (perso)
Ouvrir le terminal et entrer la commande suivante :
```conda activate tf_env```
```Streamlit run [chemin\jusqu'au\fichier]\app.py --server.port 8088``` 