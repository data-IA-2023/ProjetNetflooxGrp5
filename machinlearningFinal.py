# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:16:01 2024

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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


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
    sql_queries = text('SELECT * FROM datanetfloox.predictscore LIMIT 10000')
    df = pd.read_sql(sql_queries, engine)
    return  df
#TitreBasic, TitreRating = load_data()
#df = pd.merge(TitreBasic, TitreRating, on='tconst',how='left')


df= load_data()

with open('datas.pkl', 'wb') as f:
    pickle.dump(df, f)
    
df.isna().sum().plot(kind='bar')



# Ajoutez un titre au graphique
plt.title('Distribution des valeurs manquantes')

# Affichez le graphique
plt.show()
"""
df = df.replace('\\N', np.nan)
df = df.dropna(subset=["averageRating"])


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


preprocessor = ColumnTransformer(
    transformers=[
        ('num_encodeur',pipe_num,numeric_features),
        ('text_encodeur',pipe_text,categorial_features)
        
    ])


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('Model',LinearRegression())
    ])



scoring_metrics = ['r2','neg_mean_squared_error','neg_mean_absolute_error','neg_root_mean_squared_error']

params =[
        {   
            'preprocessor__num_encodeur__scaler': [MinMaxScaler(), RobustScaler()],
            'preprocessor__text_encodeur__vectorizer': [OneHotEncoder(handle_unknown='ignore')],
            'Model':[RandomForestRegressor()],
            'Model__max_depth':[10,30,50],
            'Model__min_samples_split':[3,5,7,10],
            'Model__n_estimators':[50,100,200]
         }
    
        ]


#cv=cross_val_score(model,X_train,y_train, cv =10)



grid = GridSearchCV(pipeline, param_grid=params, cv=5 , scoring=scoring_metrics, refit='neg_mean_absolute_error', n_jobs=14)

grid.fit(X_train, y_train)
y_pred=grid.predict(X_test) 



print()
best_model = grid.best_estimator_
print('Modele retenu:',best_model)

with open('ma_variable.pkl', 'wb') as file:
    pickle.dump(best_model, file)


best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print('r2_score',r2_score(y_test, y_pred))
print('mean_absolute_error',mean_absolute_error(y_test, y_pred))
print('mean_squared_error',mean_squared_error(y_test, y_pred))
print('root_mean_squared_error',sqrt(mean_squared_error(y_test, y_pred)))
"""