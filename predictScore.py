from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from utilities import recup_dataframe
# Exécution de la requête et chargement des données
df = recup_dataframe("""SELECT "primaryTitle" , "isAdult" , "startYear", "runtimeMinutes", 
             genres, "averageRating","primaryName" ,"category","characters","job",
             "primaryProfession","deathYear"
             FROM datanetfloox.predictscore LIMIT 50000""") 



# Exécution de la requête et chargement des données



df_clean = df.dropna(subset=['averageRating'])


df_clean['combined'] =  (df_clean['primaryTitle'].fillna('').astype(str) + ' ' + 
                        df_clean['genres'].fillna('').astype(str)+ ' ' + 
                        df_clean['primaryName'].fillna('').astype(str)+ ' ' + 
                        df_clean['category'].fillna('').astype(str) + ' ' + 
                        df_clean['characters'].fillna('').astype(str)+ ' ' +
                        df_clean['job'].fillna('').astype(str) + ' '+
                        df_clean['primaryProfession'].fillna('').astype(str))
                        
#print(df_clean['combined'])

categorical_features = df_clean['combined']
df_clean['combined']= df_clean['combined'].str.replace(',', ' ')
print(df_clean['combined'])


# Nettoyage des données, y compris la colonne cible pour les valeurs NaN
#df['averageRating'] = df['averageRating'].dropna
#print(df['averageRating'])
# Séparation des données en variables indépendantes (X) et dépendante (y)
y = df_clean["averageRating"]
X = df_clean.drop(["averageRating"], axis=1)
#print(X)
# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Sélection des types de colonnes
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#print(numeric_features)
#print(f" numeric_features {numeric_features}")

#categorical_features = X.select_dtypes(include=['object']).columns
#print(f" categorical_features :{categorical_features}")
#categorical_features['genres'] = categorical_features['genres'].str.replace(',', ' ')


# Pipelines de prétraitement pour les données numériques et catégorielles
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    #('imputer', SimpleImputer(strategy='most_frequent')),
    ('Vectorizer',CountVectorizer())
])

# Préprocesseur qui applique les transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, 'combined')
    ])

# Pipeline final incluant le préprocesseur et le modèle
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])

# Paramètres pour GridSearchCV
param_grid = [
    
    {
        'model': [RandomForestRegressor()],  
        'model__max_depth': [None, 10, 20, 30, 40, 50],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__n_estimators':[50,100,200]
    }, 
    {
        'model' : [AdaBoostRegressor()],
        'model__estimator' : [None, 10, 20],
        'model__n_estimators' : [50, 100, 200]
    }, 
    {
        'model' : [GradientBoostingRegressor()],
        'model__n_estimators' : [50, 100, 200]
        
    }
]

# Création et exécution de GridSearchCV
#grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
#grid_search.fit(X_train, y_train)
search = HalvingGridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='r2',
    n_jobs=-1,
    min_resources='exhaust',
    max_resources=200,# Ajustez selon le maximum de ressources que vous êtes prêt à allouer
    cv=5, 
    random_state=42
).fit(X_train, y_train)

# Meilleur modèle et prédictions
#best_model = grid_search.best_estimator_

#y_pred = best_model.predict(X_test)
y_pred = search.predict(X_test)    
print(search.best_params_)
# Évaluation du modèle
print('r2_score:', r2_score(y_test, y_pred))
print('mean_absolute_error:', mean_absolute_error(y_test, y_pred))
print('mean_squared_error:', mean_squared_error(y_test, y_pred))
print('root_mean_squared_error:', sqrt(mean_squared_error(y_test, y_pred)))

