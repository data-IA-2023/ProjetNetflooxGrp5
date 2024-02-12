from sqlalchemy import create_engine, text
import pandas as pd
import dotenv
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utilities import recup_dataframe

df = recup_dataframe() 

# Préparation des données: ici, vous devez définir comment vous combinez vos caractéristiques pour la vectorisation
# Pour l'exemple, supposons que nous combinons simplement toutes les colonnes textuelles en une seule chaîne
# Assurez-vous que toutes les colonnes soient de type string
df['combined_features'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Vectorisation des caractéristiques combinées
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

# Calcul des similarités cosinus
cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)
print(f" shape : {cosine_sim.shape}")



# Créer un mapping entre les titres de films et leurs indices
indices = pd.Series(df.index, index=df['primaryTitle']).drop_duplicates()
print(indices)



# Fonction de recommandation
def recommend(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Film non trouvé."
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Les trois films les plus similaires
    movie_indices = [i[0] for i in sim_scores]
    return df['primaryTitle'].iloc[movie_indices]

# Exemple d'utilisation
print(recommend("Le clown et ses chiens"))