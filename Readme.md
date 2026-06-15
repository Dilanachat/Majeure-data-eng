## Contexte et problématique

Ce projet s’inscrit dans le cadre de l’analyse prédictive appliquée au sport automobile. L’objectif est d’exploiter des données issues de stratégies de course en Formule 1 afin d’anticiper les arrêts au stand d’un pilote. Ce type de prédiction peut être utile pour l’analyse de performance, l’aide à la décision en course et l’automatisation d’indicateurs dans un environnement orienté data.

Le dataset utilisé provient d’une compétition Kaggle intitulée **“Predicting F1 Pit Stops”**, dans le cadre de la série **Playground Series - Season 6 Episode 5**. La compétition fournit une structure classique composée de trois fichiers principaux — `train.csv`, `test.csv` et `sample_submission.csv` — ce qui permet de construire un workflow machine learning complet allant de la préparation des données jusqu’à la génération de prédictions exploitables.

La problématique de ce projet est la suivante : **comment prédire si un pilote effectuera un arrêt au stand au prochain tour à partir des variables disponibles dans le dataset ?**

Pour répondre à cette question, le projet met en place un pipeline de machine learning permettant de préparer les données, entraîner un modèle prédictif, évaluer ses performances, puis générer des prédictions sur de nouvelles données. L’enjeu est de construire un processus fiable, reproductible et facilement orchestrable dans une logique MLOps.

L’objectif de ce projet est donc de concevoir un workflow de prédiction automatisé capable de produire des estimations fiables sur les arrêts au stand en Formule 1, tout en respectant les bonnes pratiques d’orchestration, de traçabilité et de réutilisabilité des traitements ML.