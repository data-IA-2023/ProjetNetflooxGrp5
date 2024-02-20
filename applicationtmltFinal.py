# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:26:11 2024

@author: naouf
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from Resources import recommandation


st.title("Films recommandés ")
st.sidebar.title("Recherche de film ")



with open('datas.pkl', 'rb') as f:
    df = pickle.load(f)

# Sélection du film à partir d'un menu déroulant
film = st.sidebar.selectbox("Quel film avez vous vu? ", df['primaryTitle'].unique())

valeur2 = st.sidebar.slider("Combien de films recherchez-vous? ", 1, 100,10)


# Affichage des films recommandés
resultat = recommandation(df,film,valeur2)
st.write("Voici les films que nous vous recommandons!")
for loop in range(len(resultat)):
    st.write(resultat[loop])
st.sidebar.write("Vos paramètres pour la recherche: ")
st.sidebar.write("Film sélectionnée: ", film)

st.sidebar.write("Nombre de films à rechercher: ", valeur2)


st.title("Prédiction de la note d'un nouveau film ")
st.sidebar.title("Prédiction de la note d'un nouveau film ")





with open('ma_variable.pkl', 'rb') as file:
    best_model = pickle.load(file)
r2_score =0.9048499654201048
mean_absolute_error= 0.09132096546864109
mean_squared_error= 0.08220164838481708
root_mean_squared_error= 0.2867082984233576

new ={
    'primaryTitle': [None],
    'runtimeMinutes': [np.nan],
    'isAdult': [np.nan],
    'startYear': [np.nan],
    'genres': [None],
    'numVotes': [np.nan],
    'category': [None],
    'characters': [None],
    'job': [None],
    'primaryProfession': [None],
    'primaryName': [None],
    'deathYear': [np.nan],
}


df_new = pd.DataFrame(new)

ISADULTE = st.sidebar.checkbox('Film pour adulte?')
if ISADULTE:
    df_new['isAdult']=1
else:
    df_new['isAdult']=0
saisie1 = st.sidebar.text_input('Rentrez le primaryTitle de votre film :',value=None, placeholder="Rentrez votre titre")
df_new['primaryTitle']=saisie1
saisie2 = st.sidebar.number_input('Rentrez le la durée de votre film  :',value=None,placeholder="Rentrez votre chiffre")
df_new['runtimeMinutes']=saisie2 
#saisie3 = st.sidebar.number_input('Rentrez le isAdult de votre film :')
#df_new['isAdult']=saisie3
saisie4 = st.sidebar.number_input("Rentrez l'année de votre film :",value=None,placeholder="Rentrez votre chiffre")
df_new['startYear']=saisie4
saisie5 = st.sidebar.text_input('Rentrez le genres de votre film :',value= np.nan,placeholder="Rentrez votre genre")
df_new['genres']=saisie5
saisie6 = st.sidebar.number_input('Rentrez le nombre de vues potentiel de votre film :',value=None,placeholder="Rentrez votre chiffre")
df_new['numVotes']=saisie6
saisie7 = st.sidebar.text_input('Rentrez la categorie de votre film :',value= np.nan,placeholder="Rentrez votre catégorie")
df_new['category']=saisie7
saisie8 = st.sidebar.text_input("Rentrez les rôles vos acteurs :",value= np.nan,placeholder="Rentrez votre caractère")
df_new['characters']=saisie8
saisie9 = st.sidebar.text_input('Rentrez le poste de vos acteurs :',value= np.nan,placeholder="Rentrez votre poste")
df_new['job']=saisie9
saisie10 = st.sidebar.text_input('Rentrez la profession de vos acteurs :',value= np.nan,placeholder="Rentrez votre premiere profession")
df_new['primaryProfession']=saisie10
saisie11 = st.sidebar.text_input('Rentrez le nom de vos acteurs :',value= np.nan,placeholder="Rentrez votre nom")
df_new['primaryName']=saisie11
saisie12 = st.sidebar.number_input('Rentrez la date de mort de vos acteurs :',value=None,placeholder="Rentrez votre chiffre")
df_new['deathYear']=saisie12

st.write(df_new)
prediction = best_model.predict(df_new)




if st.button("Reset", type="primary"):
   st.write(prediction)

