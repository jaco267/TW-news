#coding=utf-8

import numpy as np
import pandas as pd 

import utils.train_func as u
import math

db_name="TW_news_100"
# directory_name=f"./files/csv"
# db_name="TW_news_100" 
directory_name=f"./files/{db_name}/csv"

# ratio = 0.055
ratio=0.47
train_news_df = pd.read_csv( f"{directory_name}/train_news.csv",nrows=math.ceil(447057*ratio))
test_news_df = pd.read_csv(f"{directory_name}/test_news.csv",nrows=math.ceil(191597*ratio))

print(train_news_df.shape)
print(test_news_df.shape)
print(train_news_df.shape[0]+test_news_df.shape[0])
print(test_news_df.head())
# print(train_news_df.head())

# train_news_df.drop(["Unnamed: 0.3"], axis=1, inplace=True)
# train_news_df.drop(["Unnamed: 0.2"], axis=1, inplace=True)
# train_news_df.drop(["Unnamed: 0.1"], axis=1, inplace=True)

train_news_df.drop(["Unnamed: 0"], axis=1, inplace=True)
train_news_df.drop(["id"], axis=1, inplace=True)
train_news_df.drop(["title"], axis=1, inplace=True)
print(train_news_df.head())
test_news_df.drop(["Unnamed: 0"], axis=1, inplace=True)
test_news_df.drop(["id"], axis=1, inplace=True)
test_news_df.drop(["title"], axis=1, inplace=True)
print(test_news_df.head())




db_name="TW_news_50" 
import os
directory_name=f"./files/{db_name}/csv"
os.makedirs(directory_name, exist_ok = True)

pd.DataFrame(train_news_df).to_csv(  f"{directory_name}/train_news.csv",encoding="utf_8_sig",index=False)
pd.DataFrame(test_news_df).to_csv(  f"{directory_name}/test_news.csv",encoding="utf_8_sig",index=False)

