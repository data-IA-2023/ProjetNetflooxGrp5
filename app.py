import streamlit as st
from film_recommender import recommend, df  # Importez les fonctions et données nécessaires depuis votre fichier principal

# Interface utilisateur avec Streamlit
st.title("Recommandation de films")

# Sélection du film à partir d'un menu déroulant
selected_movie = st.selectbox("Choisissez un film:", df['primaryTitle'].values)

# Affichage des films recommandés
st.write("Les films recommandés sont:")
recommended_movies = recommend(selected_movie)
st.write(recommended_movies)