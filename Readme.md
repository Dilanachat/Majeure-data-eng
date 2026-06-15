## Contexte et problématique

Ce projet s’inscrit dans le cadre de l’analyse prédictive appliquée au sport. L’objectif est d’exploiter des données historiques sur les performances de joueurs afin d’anticiper le nombre de points qu’un joueur pourrait marquer lors de son prochain match. Ce type de prédiction peut être utile pour l’aide à la décision, l’analyse de performance et l’automatisation d’indicateurs dans un environnement orienté data.

Le dataset utilisé provient d’une compétition Kaggle de prédiction, avec une structure classique composée de fichiers d’entraînement, de test et de soumission, ce qui permet de construire un workflow machine learning complet allant de la préparation des données jusqu’à la génération de prédictions exploitables.

La problématique de ce projet est la suivante : **comment prédire le nombre de points qu’un joueur réalisera lors de son prochain match à partir de ses performances passées et des variables disponibles dans le dataset ?**

Pour répondre à cette question, le projet met en place un pipeline de machine learning permettant de préparer les données, entraîner un modèle prédictif, évaluer ses performances, puis générer des prédictions sur de nouvelles données. L’enjeu est de construire un processus fiable, reproductible et facilement orchestrable dans une logique MLOps.

L’objectif de ce projet est donc de concevoir un workflow de prédiction automatisé capable de produire des estimations fiables à partir de données sportives, tout en respectant les bonnes pratiques d’orchestration, de traçabilité et de réutilisabilité des traitements ML.