from sqlalchemy import create_engine, text
import pandas as pd
import dotenv
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utilities import recup_dataframe

df = recup_dataframe("SELECT * FROM datanetfloox.filmrecommander LIMIT 50000") 
df = df.drop_duplicates(subset='primaryTitle' , keep='first')

# Préparation des données: ici, vous devez définir comment vous combinez vos caractéristiques pour la vectorisation
# Pour l'exemple, supposons que nous combinons simplement toutes les colonnes textuelles en une seule chaîne
# Assurez-vous que toutes les colonnes soient de type string
df['combined_features'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
"""
# Vectorisation des caractéristiques combinées
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

# Calcul des similarités cosinus
cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)
print(f" shape : {cosine_sim.shape}")
"""


   

# Fonction de recommandation
def recommend(title):
    df['combined_features'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    column_values = df['primaryTitle'].values
    #print(column_values)
    if title not in column_values:
        return "Film non trouvé."
    
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])
    
    idx = df[df['primaryTitle'] == title].index[0]
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix[idx])
    #print(cosine_sim)
    sim_scores = list(enumerate(cosine_sim))
    #print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: 6]  # Les trois films les plus similaires
    movie_indices = [i[0] for i in sim_scores]
    return df['primaryTitle'].iloc[movie_indices]

# Exemple d'utilisation
#print(recommend("Carmencita"))


