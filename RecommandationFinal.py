# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:46:40 2024

@author: naouf
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np


#connexion base de donnée
DATABASE_URI = "postgresql+psycopg2://citus:floox2024!@c-groupe5.ljbgwobn4cx2bv.postgres.cosmos.azure.com:5432/netfloox?sslmode=require"


engine = create_engine(DATABASE_URI)


#requetes sql stockées sous forme de liste de dataframes
sql_queries = [
    text("SELECT * FROM datanetfloox.predictscore LIMIT 50000")
    
   
]

dataframes = []

#stockage des requetes 
for query in sql_queries:
    df = pd.read_sql(query, engine)
    dataframes.append(df)
    
       

#extraction des dataframes voulus
df = dataframes[0]

#input du film a regarder et du nombre de films proches souhaités
#film = input("Quel film avez vous regardé ? ")
film = input("Rentrez un film: ")
valeur = int(input("Rentrez le nombre de films similaires que vous voulez trouver: "))

def recommandation():
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


resultat =recommandation()


for loop in range(len(resultat)):
    print(resultat[loop])
engine.dispose()


"""


runfile('C:/Users/naouf/Documents/Naoufel/projet/Netfloox/Codes/Recommandation.py', wdir='C:/Users/naouf/Documents/Naoufel/projet/Netfloox/Codes')
Rentrez un film: 

return resultat

resultat=recommandation(df,film,valeur)
for loop in range(len(resultat)):
    print(resultat[loop])

for loop in range(len(resultat)):
    print(resultat[loop]) 
print(resultat)
#fermeture de la connexion
def Recommandation(film,valeur):
sorted_value =sorted(cosine_sim[index])


print(sorted_value[::-1][1:6])

for nb, element in enumerate(sorted_value[::-1][1:6]):
 
    print((df["primaryTitle"].iloc[nb+1]))
    if valeur == nb+1:
        break 
  
    
  
    
 
  df_new=pd.DataFrame(new,dtype={'primaryTitle': 'object',
                                 'runtimeMinutes': 'int32',
                                 'isAdult': 'int32',
                                 'startYear':'int32',
                                 'genres': 'object',
                                 'numVotes': 'int32',
                                 'category': 'object',
                                 'characters':'object',
                                 'job': 'object',
                                 'primaryProfession':'object',
                                 'primaryName': 'object',
                                 'deathYear':'int32'})
  
  for column in df_new.columns:
      saisie = st.sidebar.text_input(f'Rentrez le {column} de votre film :')
      df_new.loc[0, column] = saisie


  # Affichage des données saisies dans la barre latérale
  if st.sidebar.button("Ajouter"):
      st.table(df_new.head(1))

  newpred = y_pred = best_model.predict(df_new)
  st.write(newpred)

"""