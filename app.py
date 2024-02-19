import datetime
import streamlit as st
from film_recommender import recommend, df  # Importez les fonctions et données nécessaires depuis votre fichier principal
from utilities import recup_dataframe
import pickle
from joblib import dump, load
import pandas as pd
import numpy as np

# Interface utilisateur avec Streamlit
st.title("Recommandation de films")

# Sélection du film à partir d'un menu déroulant
selected_movie = st.selectbox("Choisissez un film:", df['primaryTitle'].values)

# Affichage des films recommandés
st.write("Les films recommandés sont:")
recommended_movies = recommend(selected_movie)
st.write(recommended_movies)

st.title("prediction du score")


df_predict = recup_dataframe("""SELECT "primaryTitle" , "startYear", "runtimeMinutes", 
             genres, "averageRating","primaryName" ,"category","characters","job",
             "primaryProfession","deathYear"
             FROM datanetfloox.predictscore LIMIT 50000""") 



reload_model = load('bestparam.joblib')


new ={
    'primaryTitle': [None],
    'runtimeMinutes': [np.nan],
    'isAdult': [np.nan],
    'startYear': [np.nan],
    'genres': [None],
    'category': [None],
    'characters': [None],
    'job': [None],
    'primaryProfession': [None],
    'primaryName': [None],
    'deathYear': [np.nan],
}

primaryTitle = st.text_input('entrer le nom du nouveau film :' )


job = st.text_input('entrer le nom du nouveau film :' )

df_predict = df_predict.drop_duplicates(subset=['primaryName'] , keep='first')
selected_movie = st.selectbox("Vous pouvez chosir un nom dans notre base de donnée:",df_predict['primaryName'].values)
primaryProfession = st.text_input('entrer la professions :' )
job = st.text_input('entrer le nom du nouveau film :' )

characters = st.text_input('entrer  :' )
startYear = 2024
runtimeMinutes = st.number_input('entrer la durée de votre film:')

is_adulte_activated =  st.toggle('isadulte')
print(is_adulte_activated)

df_predict['genres'] = df_predict['genres'].str.replace(',', ' ')
df_predict = df.drop_duplicates(subset='genres' , keep='first')
genres = st.multiselect('liste des genres :',df_predict['genres'].values) 

def x_text_transform(nameTittle=None,nameCharacter1=None, nameCharacter2=None, selected_movie=None, genres=None):
    array_text = [nameTittle,nameCharacter1,nameCharacter2,selected_movie, genres]
    df= pd.DataFrame(array_text)
    df['combined'] =  (df['nameTittle'].fillna('').astype(str) + ' ' + 
                        df['nameCharacter1'].fillna('').astype(str)+ ' ' + 
                        df['nameCharacter2'].fillna('').astype(str)+ ' ' + 
                        df['selected_movie'].fillna('').astype(str) + ' ' + 
                        df['genres'].fillna('').astype(str))
    print(df)









"""df_clean['primaryTitle'].fillna('').astype(str) + ' ' + 
                        df_clean['genres'].fillna('').astype(str)+ ' ' + 
                        df_clean['primaryName'].fillna('').astype(str)+ ' ' + 
                        df_clean['category'].fillna('').astype(str) + ' ' + 
                        df_clean['characters'].fillna('').astype(str)+ ' ' +
                        df_clean['characters'].fillna('').astype(str)+ ' ' +
                        df_clean['job'].fillna('').astype(str) + ' '+
                        df_clean['primaryProfession'].fillna('').astype(str))"""