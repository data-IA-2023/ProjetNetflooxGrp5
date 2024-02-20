# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:23:17 2024

@author: naouf
"""
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from math import sqrt
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge,RidgeCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,RobustScaler, Binarizer,PolynomialFeatures,LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier


def load_data():
    DATABASE_URI = "postgresql+psycopg2://citus:floox2024!@c-groupe5.ljbgwobn4cx2bv.postgres.cosmos.azure.com:5432/netfloox?sslmode=require"


    engine = create_engine(DATABASE_URI)

    sql_queries = text('SELECT * FROM datanetfloox.predictscore LIMIT 50000')

    df = pd.read_sql(sql_queries, engine)
    return  df

def recommandation(df,film,valeur):
    #combinaisons des colomnes
    df['combined_features'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            
    #recherche de l'index du film proposé
    
    
    index = df[df["primaryTitle"] == film].index[0]
    
    
    #vectorisation
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])
    
   
    
    #matrice de distance methode cosinus
    cosine_sim= cosine_similarity(count_matrix , count_matrix[index])
    
    
    
    
    #triage des distances correspondant à l'index du film en leur creant un index
    trie =sorted(list(enumerate(cosine_sim)),key=lambda x:x[1],reverse=True)
    
    trie.pop(0)
    
    liste=[]
    #on remplie une liste des valeurs des films
    for nb, element in trie:
        liste.append(df["primaryTitle"].iloc[nb])
    
    
    liste_sans_doublons=[]
    for x in liste:
        if x not in liste_sans_doublons:
            liste_sans_doublons.append(x)
    
    
    resultat = liste_sans_doublons[0:valeur]
    if film in resultat:
        resultat= liste_sans_doublons[0:valeur+1]
        resultat.remove(film)
    
    liste_recommandation=[]
    for loop in range(len(resultat)):
        liste_recommandation.append(resultat[loop])
    return list(liste_recommandation)