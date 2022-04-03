
from pymongo import MongoClient  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.stem import PorterStemmer
from tqdm import tqdm
import string
import re
import numpy as np
from textblob import TextBlob
import pandas as pd

tqdm.pandas()


myclient = MongoClient("mongodb://user:pass@ec2-13-40-36-157.eu-west-2.compute.amazonaws.com:27017/")  

ruta=r'C:\ruta'
   
# Definición de BBDD
db = myclient["nebulova"] 

# Definición de las colecciones

usuarios = db["users"]
tweets = db["tweets"]


# Leemos los datos de la BBDD MongoDB

pipeline=[
    # {
    #     '$sort': {
    #         'created_at': 1
    #     }
    # }, 
    # {
    #     '$limit': 10000
    # }, 
    {
        '$project': {
            'text': 1, 
            'created_at': 1,
            'user_id_str': 1,
            # 'retweeted': 1,
            '_id': 0
        }
    }
]

# Creamos el dataframe con los tweets

df = pd.DataFrame(tweets.aggregate(pipeline))

df.info()

# Generamos un fichero con las stopwords que aplican en este caso
# BTC y similares

stopwords_propias = (pd.read_csv(ruta + r"\stopwords.csv",names=["words"]))['words'].to_list()

# Juntamos todas las stopwords en un único set

STOP = set(stopwords.words("english")+stopwords_propias)
PUNCT = set(string.punctuation)
DIGITS = set(string.digits)
MIN_LENGTH = 2

# REGEX para filtrar caracteres extraños, multiples espacios, enlaces a URL,
# referencias a usuarios de twitter y direcciones de correo electrónico,
# eliminamos también las cadenas BTC y BITCOIN si forman parte de una palabra

weird_chars = re.compile(r"[^#a-z\s]")
multi_space = re.compile(r"\s{2,}")
url = re.compile(r'http\S+')
arroba = re.compile(r'[0-9a-z.]*@[0-9a-z.]*')
btc = re.compile(r'\s(btc)')
bitcoin = re.compile(r'\s(bitcoin)')
multichar = re.compile(r'([a-z])\1{2,}')

# Normalización de texto según su significado

lemmatizer = WordNetLemmatizer()

# Función para limpieza de texto. Se aplican las condiciones indicadas
# anteriormente

def clean(text):
    clean_text = text.lower()  # normalizar texto
    clean_text = " ".join([i for i in clean_text.split() if i not in STOP])
    clean_text = url.sub("", clean_text)
    clean_text = weird_chars.sub("", clean_text)
    clean_text = arroba.sub("", clean_text)
    clean_text = "".join([i for i in clean_text if i not in PUNCT])
    clean_text = " ".join([i for i in clean_text.split() if i not in STOP])
    clean_text = btc.sub(" ", clean_text)
    clean_text = bitcoin.sub(" ", clean_text)
    clean_text = "".join([i for i in clean_text if i not in DIGITS])
    clean_text = multi_space.sub(" ", clean_text)
    # Integrar Regex para convertir de "gooooooooooooooooooal" a "goal"
    clean_text = multichar.sub(r"\1",clean_text)
    clean_text = " ".join([lemmatizer.lemmatize(i) for i in clean_text.split()])
    clean_text = " ".join([i for i in clean_text.split() if (len(i) > MIN_LENGTH)])
    # stemmer = PorterStemmer()
    # clean_text = " ".join([stemmer.stem(i) for i in clean_text.split()])
    # clean_text = " ".join(set(clean_text.split()))
    return clean_text

# Se aplica la función

df["clean_text"] = df["text"].progress_apply(clean)

# Bag of Words, para ver volumen que tiene cada palabra dentro del corpus

BoW = dict()
for doc in df["clean_text"]:
    for token in doc.split():
        BoW[token] = BoW.get(token, 0) + 1
BoW_df = pd.Series(BoW).sort_values(ascending=False)

# Función sentimiento. Para determinar la polaridad de cada tweet
# Polaridad < 0: sentimiento negativo
# Polaridad > 0: sentimiento positivo
# Polaridad = 0: sentimiento neutro
# Se devuelve también el valor del sentimiento

def sentiment(text):
    sentimiento = TextBlob(text).sentiment.polarity
    if sentimiento>0:
        polaridad = 1
    elif sentimiento<0:
        polaridad = -1
    else:
        polaridad=0
    return (sentimiento,polaridad)

# 

df[["sentimiento","polaridad"]]=df.progress_apply(lambda row: sentiment(row.clean_text),
                                    axis="columns", result_type="expand")



data=df.drop(["text","clean_text","polaridad"], axis=1)

data.set_index("created_at", inplace=True)
data.replace(0,np.nan, inplace=True)

data.to_csv(ruta + r"\datos_tweets.csv")

pipeline_mentions=[
    # {
    #     '$sort': {
    #         'created_at': 1
    #     }
    # }, 
    # {
    #     '$limit': 10000
    # }, 
    {
        '$project': {
            # 'text': 1, 
            # 'created_at': 1,
            # 'user_id_str': 1,
            # 'retweeted': 1,
            'user_mentions': 1,
            '_id': 0
        }
    }
]

# Listado de menciones a un usuario, para darle mayor importancia a los
# usuarios más mencionados

df_mentions = pd.DataFrame(tweets.aggregate(pipeline_mentions))

lista_menciones= list()
for i in df_mentions.iterrows():
    if len(i[1][0])>0:
        for j in range(len(i[1][0])):
            lista_menciones.append(i[1][0][j]["id_str"])

lista_menciones = pd.DataFrame(lista_menciones, columns=["user_id_str"])

conteo_menciones = lista_menciones.groupby("user_id_str")["user_id_str"].agg(["count"])

conteo_menciones = conteo_menciones.rename(columns={'count':'count_mentions'})

# Calculamos el máximo número de menciones para ponderar los conteos respecto
# al usuario más mencionado

# Se divide también entre 2 porque se suma la ponderación de menciones y la
# ponderación del número de tweets emitidos

max_mentions=conteo_menciones["count_mentions"].max()
conteo_menciones["peso_menciones"]=conteo_menciones["count_mentions"]/max_mentions/2
conteo_menciones.reset_index(inplace=True)

conteo_menciones.to_csv(ruta + r"\usuarios_mencionados.csv")

pipeline_users=[
    # {
    #     '$sort': {
    #         'created_at': 1
    #     }
    # }, 
    # {
    #     '$limit': 10000
    # }, 
    {
        '$project': {
            # 'text': 1, 
            # 'created_at': 1,
            'user_id_str': 1,
            # 'retweeted': 1,
            # 'user_mentions': 1,
            '_id': 0
        }
    }
]

# Listado de usuarios que han escrito tweets relacionados con la temática, 
# para darle mayor importancia a los usuarios más activos escribiendo

df_users = pd.DataFrame(tweets.aggregate(pipeline_users))

conteo_users = df_users.groupby("user_id_str")["user_id_str"].agg(["count"])
conteo_users = conteo_users.sort_values("count", ascending=False)
max_tweets= conteo_users["count"].max()
conteo_users["peso_tweets"] = conteo_users["count"]/max_tweets/2
conteo_users.reset_index(inplace=True)

conteo_users.to_csv(ruta + r"\usuarios.csv")

data.reset_index(inplace=True)

# Cualificamos los datos con los datos de peso de tweets y peso menciones

data = pd.merge(data,conteo_menciones, how="left")
data = pd.merge(data,conteo_users, how="left")
data = data.drop(["count","count_mentions"], axis=1)

# Reemplazamos los nan por valor mínimo

data["peso_menciones"] = data["peso_menciones"].replace(np.nan, conteo_menciones["peso_menciones"].min())
data["sentimiento_ponderado"] = data["sentimiento"]*(data["peso_tweets"]+data["peso_menciones"])

data.set_index("created_at", inplace=True)

# Agrupamos datos para cada 5 min

agrupado = data.groupby(pd.Grouper(level=0, freq="5Min")).mean()

agrupado["sentimiento_ponderado"].interpolate(method="linear",inplace=True)

# Calculamos las medias móviles con ventanas de 10 y 30 periodos

agrupado["MM_10p"] = agrupado["sentimiento_ponderado"].rolling(window=10).mean()
agrupado["MM_30p"] = agrupado["sentimiento_ponderado"].rolling(window=30).mean()
agrupado.to_csv(r"\agrupado.csv")

# agrupado = pd.read_csv(ruta + r"\agrupado.csv")
# fig = px.line(agrupado, x="created_at", y="sentimiento")
# fig.show("browser")

