from sqlalchemy import create_engine, text
import pandas as pd
import dotenv
import os



def recup_dataframe(resquest): 
    # Charger les variables d'environnement
    dotenv.load_dotenv()

    # Connexion à la base de données
    DATABASE_URI = os.environ['DATABASE_URI']
    engine = create_engine(DATABASE_URI)

    # Exécuter la requête SQL et stocker le résultat dans un DataFrame
    sql_query = text(resquest)
    df = pd.read_sql(sql_query, engine)
    engine.dispose()
    return df 
