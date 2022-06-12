#coding=utf-8

import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

import json
# from jieba.analyse import *
import jieba.analyse as jieba_analayse
import jieba
import utils.mongo_function as mongo_f

train_percentage=0.7

from dotenv import dotenv_values
config = dotenv_values(".env")

db_name="TW_news_10"  #TW_news
db_name="TW_news_100" 
db_name="TW_news"
# my_cluster = MongoClient(config["API_KEY_MINE"])
my_cluster = MongoClient(config["local_DB"])
my_db          = my_cluster[db_name]
# my_collection  = my_db["news"]
my_collection  = my_db["TW_news_100"]

# my_collection.delete_many({"content":""})


agg_list=mongo_f.agg_category(my_collection)
for cat in agg_list:
    print(cat["_id"],cat["count"])
# my_collection.update_many({"category":"運動"},{"$set":{"category":"體育"}})
# my_collection.update_many({"category":"全球"},{"$set":{"category":"國際"}})
# mongo_f.target_train_document(my_collection)

mongo_f.count_num(my_collection)



import pandas as pd 
news_object={"id":[],"label":[],"title":[],"content":[]}
for mongo in my_collection.find({"train_target":True}):
    seg_list2= jieba.cut(mongo["content"], cut_all=False)
    news_object["id"].append(mongo["_id"])
    news_object["content"].append("|".join(seg_list2))#
    news_object["label"].append(mongo['category'])
    news_object["title"].append(mongo['title'])

news_df = pd.DataFrame(news_object)



from sklearn.model_selection import train_test_split
train_news_df, test_news_df = train_test_split( news_df , test_size=1-train_percentage, random_state=4)


db_name="TW_news_100" 
import os
directory_name=f"./files/{db_name}/csv"
os.makedirs(directory_name, exist_ok = True)

pd.DataFrame(news_df).to_csv(  f"{directory_name}/news.csv",encoding="utf_8_sig")
pd.DataFrame(train_news_df).to_csv(  f"{directory_name}/train_news.csv",encoding="utf_8_sig")
pd.DataFrame(test_news_df).to_csv(  f"{directory_name}/test_news.csv",encoding="utf_8_sig")


directory_name=f"./files/{db_name}/info"
os.makedirs(directory_name, exist_ok = True)
with open( directory_name + "/some_info.txt",'w',encoding= 'utf-8') as f:
    f.write(f"all news df shape   : {news_df.shape}\n")
    f.write(f"train news df shape : {train_news_df.shape}\n")
    f.write(f"test news df shape  : {test_news_df.shape}\n")
