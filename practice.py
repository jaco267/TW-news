from sklearn.feature_extraction.text import CountVectorizer

texts = ["dog cat fish","dog cat cat","fish bird","bird"]

cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.get_feature_names_out())
print(cv_fit.toarray())
print(cv_fit.toarray().sum(axis=0))




label_mapping = {'國際':0,'生活':1,'政治':2,'體育':3,'社會':4,'財經':5}
label_mapping = {y: x for x, y in label_mapping.items()}
print(label_mapping)
category=[0,1,2]
def cat(x):
    return label_mapping[x]
num = map(cat,category)
print(list(num))




import utils.mongo_function as mongo_f
import pymongo
from pymongo import MongoClient
from dotenv import dotenv_values
config = dotenv_values(".env")


my_cluster = MongoClient(config["local_DB"])
my_db  = my_cluster["TW_news"]
my_collection  = my_db["TW_news_100"]

agg_list=mongo_f.agg_category(my_collection)
for cat in agg_list:
    print(cat["_id"],cat["count"])

mongo_f.agg_media(my_collection)