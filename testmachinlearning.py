# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:26:55 2024

@author: naouf
"""
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor,RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
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


DATABASE_URI = "postgresql+psycopg2://citus:floox2024!@c-groupe5.ljbgwobn4cx2bv.postgres.cosmos.azure.com:5432/netfloox?sslmode=require"


engine = create_engine(DATABASE_URI)

sql_queries = text('SELECT "primaryTitle" , "isAdult" , "startYear", "runtimeMinutes", genres, "averageRating", "numVotes" FROM datanetfloox.viewnaoufel LIMIT 1000')
    
   
df = pd.read_sql(sql_queries, engine)



df = df.dropna(subset=["averageRating"])

y = df["averageRating"]
X = df.drop(["averageRating"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

numeric_features = X.select_dtypes(exclude=['object']).columns
categorial_features = X.select_dtypes(include=['object']).columns




pipe_num = Pipeline([
        ('imputer',SimpleImputer(strategy="median")),
        ('scaler', MinMaxScaler())
       ])

 
pipe_text = Pipeline([
        ('imputer',SimpleImputer(strategy="most_frequent")),
        ("vectorizer",OneHotEncoder(handle_unknown='ignore'))
      ])



preprocessor = ColumnTransformer(
    transformers=[
        ('scaler',pipe_num,numeric_features),
        ('text_encodeur',pipe_text,categorial_features)
        
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Model',LinearRegression())
    ])

params =[
        {
           
            'preprocessor__scaler': [MinMaxScaler(), RobustScaler(),Binarizer(),PolynomialFeatures()],
            'Model': [Ridge()],
            'Model__alpha': [1.5, 9.5, 7.5]
        },
        {   
            'Model':[RandomForestRegressor()]
         
         },
        {
          
            'preprocessor__scaler': [MinMaxScaler(), RobustScaler(),Binarizer(),PolynomialFeatures()],
            'Model': [LinearRegression()],
            'Model__fit_intercept': [True, False]
        },
        {
            
            'preprocessor__scaler': [MinMaxScaler(), RobustScaler(),Binarizer(),PolynomialFeatures()],
            'Model':[DecisionTreeRegressor()],
            'Model__max_depth':[2,9]
        }
        ]

#cv=cross_val_score(model,X_train,y_train, cv =10)



grid = GridSearchCV(pipeline, param_grid=params, cv=10 , refit='neg_mean_absolute_error',n_jobs=14)

grid.fit(X_train, y_train)
y_pred=grid.predict(X_test) 

best_model = grid.best_estimator_
print('Modele retenu:',best_model)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)


print('r2_score',r2_score(y_test, y_pred))
print('mean_absolute_error',mean_absolute_error(y_test, y_pred))
print('mean_squared_error',mean_squared_error(y_test, y_pred))
print('root_mean_squared_error',sqrt(mean_squared_error(y_test, y_pred)))



engine.dispose()
