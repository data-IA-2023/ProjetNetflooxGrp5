# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:40:25 2024

@author: naouf
"""

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeRegressor
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

sql_queries = text("SELECT * FROM datanetfloox.viewnaoufel LIMIT 1000000")
    
   
df = pd.read_sql(sql_queries, engine)

df.drop_duplicates()

df = df.dropna(subset=["averageRating"])


df['numVotes'] = df['numVotes'].replace(np.nan,0)
df['averageRating'] = df['averageRating'].replace(np.nan,0)
df['startYear'] = df['startYear'].replace(np.nan,df['startYear'].mean())
df['runtimeMinutes'] = df['runtimeMinutes'].replace(np.nan,df['runtimeMinutes'].mean())



y = df["averageRating"]
X = df.drop(["averageRating"], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


numeric_features = X.select_dtypes(exclude=['object']).columns


categorial_features = X.select_dtypes(include=['object']).columns






pipe_num = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy="median")),
        ('scaler', MinMaxScaler())
       ])

 
pipe_text = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy="most_frequent")),
        ("vectorizer",OneHotEncoder(handle_unknown='ignore'))
    ])


#OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler',pipe_num,numeric_features),
        ('text_encodeur',pipe_text,categorial_features)
        
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Model',LinearRegression())
    ])
print(pipeline)
scoring_metrics = ['r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

params =[
        {
           
            'preprocessor__scaler': [MinMaxScaler(), RobustScaler(),Binarizer(),PolynomialFeatures()],
            'Model': [Ridge()],
            'Model__alpha': [1.5, 9.5, 7.5]
        },
        {   
            'Model':[DecisionTreeRegressor()],
            'Model__max_depth':[2,5,6,7,8,9,15],
            'Model__min_samples_split':[3,4,5,6,7,8,9,10],
            'Model__criterion':["squared_error", "friedman_mse", "absolute_error", "poisson"]
         },
        {
          
            'preprocessor__scaler': [MinMaxScaler(), RobustScaler(),Binarizer(),PolynomialFeatures()],
            'Model': [LinearRegression()],
            'Model__fit_intercept': [True, False]
        },
        {
            
            'preprocessor__scaler': [MinMaxScaler(), RobustScaler(),Binarizer(),PolynomialFeatures()],
            'Model': [RidgeCV()],
            'Model__alphas': [1.5, 9.5, 7.5]
        }
        ]


#cv=cross_val_score(model,X_train,y_train, cv =10)



grid = GridSearchCV(pipeline, param_grid=params, cv=5 , scoring=scoring_metrics, refit='neg_mean_absolute_error',n_jobs=14)

grid.fit(X_train, y_train)
y_pred=grid.predict(X_test) 



print()
best_model = grid.best_estimator_
print('Modele retenu:',best_model)


best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print('r2_score',r2_score(y_test, y_pred))
print('mean_absolute_error',mean_absolute_error(y_test, y_pred))
print('mean_squared_error',mean_squared_error(y_test, y_pred))
print('root_mean_squared_error',sqrt(mean_squared_error(y_test, y_pred)))



engine.dispose()

