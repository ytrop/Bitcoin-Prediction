from tweepy import OAuthHandler, API, Stream
from datetime import datetime
from pymongo import MongoClient  
  
# Conexión a MongoDB en local 
myclient = MongoClient("mongodb://localhost:27017/")  
   
# Definición de BBDD
db = myclient["nebulova"] 

# Definición de las colecciones

usuarios = db["users"]
tweets = db["tweets"]

# Claves de Twitter
  
TWITTER_APP_KEY = "*****************"
TWITTER_APP_SECRET = "*****************"

TWITTER_KEY = "*****************"
TWITTER_SECRET = "*****************"

auth = OAuthHandler(TWITTER_APP_KEY, TWITTER_APP_SECRET)
auth.set_access_token(TWITTER_KEY, TWITTER_SECRET)

api = API(auth)

# Definición del streamer

claves_usuario = ["id_str","screen_name","location","url","verified",
                  "followers_count","friends_count","listed_count",
                  "favourites_count","statuses_count","withheld_in_countries"]
claves_tweet = ["id_str","source","in_reply_to_status_id",
                "in_reply_to_status_id_str","in_reply_to_user_id",
                "in_reply_to_user_id_str","in_reply_to_screen_name","geo",
                "coordinates","place","extended_tweet","full_text",
                "quote_count","reply_count","retweet_count","favorite_count",
                "favorited","retweeted"]


class StreamListener(Stream):

    def on_status(self, status):
        
        # Entrada del usuario en la BBDD
        
        usuario_t = dict()
        for key,value in status._json["user"].items():
            if key in claves_usuario:
                usuario_t[key] = value
        usuario_t["created_at"] = datetime.strptime(
            status._json["user"]["created_at"],'%a %b %d %H:%M:%S +0000 %Y')
        if not status._json["user"]["profile_image_url"]:
            usuario_t["profile_image_url"] = False
        else:
            usuario_t["profile_image_url"] = True
        
        a = usuarios.find_one_and_replace({'id_str' : 
                                             status._json["user"]["id_str"]},
                                            usuario_t)
        if a == None:
            usuarios.insert_one(usuario_t)
            
        # Entrada del tweet en la BBDD
        
        tweet = dict()
        
        for key,value in status._json.items():
            if key in claves_tweet:
                tweet[key] = value
        if status._json["truncated"] == True:
            tweet["text"] = status._json["extended_tweet"]["full_text"]
        else:
            tweet["text"] = status._json["text"]
            
        tweet["created_at"] = datetime.strptime(
            status._json["created_at"],'%a %b %d %H:%M:%S +0000 %Y')
        tweet["user_id_str"]= status._json["user"]["id_str"]
        
        hashtags = list()
        for i in range(len(status._json["entities"]["hashtags"])):
            hashtags.append(status._json["entities"]["hashtags"][i]["text"])
        
        urls = list()
        for j in range(len(status._json["entities"]["urls"])):
            urls.append(status._json["entities"]["urls"][j]["url"])
            
        user_mentions = list()
        for k in range(len(status._json["entities"]["user_mentions"])):
            user_mentions.append(
                {"screen_name" : 
                  status._json["entities"]["user_mentions"][k]["screen_name"],
                  "id_str" : 
                      status._json["entities"]["user_mentions"][k]["id_str"]})
        
        symbols = list()
        for l in range(len(status._json["entities"]["symbols"])):
            symbols.append(status._json["entities"]["symbols"][l]["text"])
        
        tweet["hashtags"] = hashtags
        tweet["urls"] = urls
        tweet["user_mentions"] = user_mentions
        tweet["symbols"] = symbols
        
        tweet["inserted_at"] = datetime.now()
        
        tweets.insert_one(tweet)
            
    def on_error(self, status_code):
        if status_code == 420:
            return False
    
# Ponemos el script a la escucha

stream_listener = StreamListener(TWITTER_APP_KEY,TWITTER_APP_SECRET,
                                 TWITTER_KEY,TWITTER_SECRET)

# Lo mantenemos a la escucha hasta que haya una interrupción de teclado
# o haya una excepción de la ejecución

try:
    stream_listener.filter(track=["BTC"], languages=['en'])
except KeyboardInterrupt:
    stream_listener.disconnect()
except Exception as e:
    print(e)
    
