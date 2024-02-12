from sqlalchemy import create_engine, text
import pandas as pd
import dotenv
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recup_dataframe(): 
    # Charger les variables d'environnement
    dotenv.load_dotenv()

    # Connexion à la base de données
    DATABASE_URI = os.environ['DATABASE_URI']
    engine = create_engine(DATABASE_URI)

    # Exécuter la requête SQL et stocker le résultat dans un DataFrame
    sql_query = text("SELECT * FROM datanetfloox.testalgo LIMIT 300")
    df = pd.read_sql(sql_query, engine)
    return df 