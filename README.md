# Interface Graphique de Regression

## Description
Ce projet consiste en une interface graphique (GUI) intuitive qui permet aux utilisateurs, même sans compétences avancées en programmation, de créer et entraîner facilement des modèles de machine learning pour des tâches de régression. Grâce à cette interface, l'utilisateur peut charger ses données, configurer divers paramètres d'entraînement (choix du modèle, sélection des features, etc.), visualiser les résultats et évaluer les performances du modèle.

## To do list

- [x] Ajouter des architectures de réseaux de neurones dans la partie "Entraînements préliminaires"
- [ ] Rattacher au GPU les calculs éligibles 
- [ ] Faire un graphique par output car les graphiques d'erreurs sont surchargés et illisibles
- [ ] Par défaut ne mettre aucune variable dans la case "Variable quantiatives"
- [ ] Faire une video tutoriel
- [ ] Ajouter méthode de clustering pour faire un modèle par cluster
- [ ] Ajout de traitement des données temporelles 



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


### Utilisation 
Ouvrir le terminal et entrer la commande suivante :
```Streamlit run [chemin\jusqu'au\fichier]\app.py```
