#coding=utf-8

import numpy as np
import pandas as pd 

import utils.train_func as u
import math
# db_name="TW_news_100"
# directory_name=f"./files/csv"
db_name="TW_news_50" 
directory_name=f"./files/{db_name}/csv"




train_news_df = pd.read_csv( f"{directory_name}/train_news.csv")
test_news_df = pd.read_csv(f"{directory_name}/test_news.csv")

print(train_news_df.head())
print(train_news_df.shape)
print(test_news_df.shape)
# print(train_news_df.head())

print(train_news_df["label"].unique())

label_mapping = {'國際':0,'生活':1,'政治':2,'體育':3,'社會':4,'財經':5}
train_news_df["label"] = train_news_df['label'].map(label_mapping)
# print(train_news_df.head())
test_news_df["label"] = test_news_df['label'].map(label_mapping)


def df_to_array(news_df):
    words=[]
    y_mat=[]
    # print(math.ceil(len(news_df)*0.09))
    # for index in range(math.ceil(len(news_df)*0.09)):
    for index in range(len(news_df)):
        content_list = news_df["content"][index].split("|")

        words.append(" ".join(content_list))
        y_mat.append(news_df["label"][index])
    # return np.array(words), np.array(y_mat)
    return words, np.array(y_mat)

x_train_words, y_train = df_to_array(train_news_df)
x_test_words, y_test = df_to_array(test_news_df)


print("\n\n\n\n".join(x_train_words[:3]))
# print(x_train_words.shape)
# print(x_test_words.shape)
#Doc2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def label_sentences(corpus,label_type):
    labeled = []
    for index, words in enumerate(corpus):
        label = label_type + '____' +str(index)
        labeled.append(TaggedDocument(words.split(),[label]))
    return labeled


x_train = label_sentences(x_train_words,"Train")
# print(x_train[0])
x_test = label_sentences(x_test_words,"Test")
print("\n\n")
print(x_test[0])
all_data = x_train + x_test

model_dbow = Doc2Vec(dm=0,vector_size=300,negative=5,min_count=1,alpha=0.065,min_alpha=0.065)


model_dbow.build_vocab([x for x in all_data])  #x_train




from sklearn import utils

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in all_data]),total_examples=len(all_data),epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha



# print(labeled["Test____0"])
def get_vectors(model,corpus_size, vectors_size,vectors_type):
    vectors = np.zeros((corpus_size,vectors_size))
    for index in range(0,corpus_size):
        prefix = vectors_type + '____' + str(index)
        vectors[index] = model.dv[prefix]   #docvecs
    return vectors

train_vectors_dbow = get_vectors(model_dbow,len(x_train), 300, 'Train')

test_vectors_dbow = get_vectors(model_dbow,len(x_test), 300, 'Test')

# ivec = model.infer_vector(doc_words=tokens_list, steps=20, alpha=0.025)
# print(model.most_similar(positive=[ivec], topn=10))


print(train_vectors_dbow.shape)

print(train_vectors_dbow[:5])

print(train_vectors_dbow[-5:])
###

import os


directory_name=f"./files/{db_name}/csv/doc2vec"

os.makedirs(directory_name, exist_ok = True)

train_vec_df = pd.DataFrame(train_vectors_dbow)
pd.DataFrame(train_vec_df).to_csv(  f"{directory_name}/train_vec.csv",encoding="utf_8_sig")

test_vec_df = pd.DataFrame(test_vectors_dbow)
pd.DataFrame(test_vec_df).to_csv(  f"{directory_name}/test_vec.csv",encoding="utf_8_sig")


y_train_df = pd.DataFrame(y_train)
pd.DataFrame(y_train_df).to_csv(  f"{directory_name}/y_train.csv",encoding="utf_8_sig")
y_test_df = pd.DataFrame(y_test)
pd.DataFrame(y_test_df).to_csv(  f"{directory_name}/y_test.csv",encoding="utf_8_sig")



#Logistic regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model = model.fit(train_vectors_dbow,y_train)
yhat = model.predict(test_vectors_dbow)


from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test, yhat, labels=[0,1,2,3,4,5])

print(conf_mat)

res=model.score(test_vectors_dbow,y_test)
print(res)