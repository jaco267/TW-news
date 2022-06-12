#coding=utf-8

import numpy as np
import pandas as pd 
import os
import utils.train_func as u

db_name="TW_news_50"  #accuracy  0.84
# db_name="TW_news"   #accuracy  0.85
# db_name="TW_news_100" #accuracy  0.80
# directory_name=f"./files/csv"
# db_name="TW_news"
directory_name=f"./files/{db_name}/csv"

train_news_df = pd.read_csv( f"{directory_name}/train_news.csv")
test_news_df = pd.read_csv(f"{directory_name}/test_news.csv")

feature_size=4500
print(train_news_df.head())

print(train_news_df["label"].unique())

label_mapping = {'國際':0,'生活':1,'政治':2,'體育':3,'社會':4,'財經':5}
train_news_df["label"] = train_news_df['label'].map(label_mapping)
# print(train_news_df.head())
test_news_df["label"] = test_news_df['label'].map(label_mapping)



stop_string="1100  1111  115  116  117  118  1203  121  122  123  125  126  127  128  129  1300  131  132  133  134  135  136  137  138  139  1400  141  142  143  144  145  146  148  149  151  152  154  155  157  158  1600  162  164  165  166  167  168  169  1700  171  174  175  177  178  1800  182  185  188  189  190  1925  195  1980  199  1990  1996  1998  1999  2001  2002  2003  2004  202  2026  2050  206  210  211  219  220  225  230  2300  2330  240  2400  257  258  260  270  272  280  286  290  303  320  321  322  325  327  328  329  343  346  350  3500  364  370  371  376  380  396256  3C  3D  3c4cToo  408  450  4500  460  520  586  590  607  650  674  699  701  737  750  838  839  888  9000  911  999  103  170  001922  0800  102  104  111  1922  37gsay1  140  160  1995  2006  311  499  4000  539  7000  76  79  114  124  130  2005  2008  2010  2023  2025  2500  00  000  01  02  03  04  05  06  07  08  09  10  100  1000  101  105  106  107  108  109  11  110  1100216  1100217  1100218  1100219  112  113  119  12  120  1200  13  14  15  150  1500  16  17  18  180  19  1px  20  200  2000  2007  2009  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2024  2030  21  22  228  23  24  25  250  26  27  28  29  30  300  3000  31  32  33  34  35  36  360  37  38  39  40  400  41  42  43  44  45  46  47  48  49  50  500  5000  51  52  53  54  55  56  57  58  59  5G  60  600  6000  61  62  63  64  65  66  67  68  69  70  700  71  72  73  74  75  77  78  80  800  8000  81  82  83  84  85  86  87  88  89  90  900  91  92  93  94  95  96  97  98  99  A9"
stop_list = stop_string.split("  ")

def df_to_array(news_df):
    words=[]
    y_mat=[]
    for index in range(len(news_df)):
        content_list = news_df["content"][index].split("|")
        words.append(" ".join(content_list))
        y_mat.append(news_df["label"][index])
    return words, y_mat

x_train_words, y_train = df_to_array(train_news_df)
x_test_words, y_test = df_to_array(test_news_df)

from sklearn.feature_extraction.text import CountVectorizer
#########
cv = CountVectorizer(analyzer='word', max_features=feature_size, lowercase = False,stop_words=stop_list)
cv.fit(x_train_words)
#########
print("feature words : ",cv.get_feature_names_out())
for word_ in cv.get_feature_names_out():
    print(word_,end="  ")
print("\n\n")
#turn bags of words into matrix
x_train=cv.transform(x_train_words).toarray()  
x_test=cv.transform(x_test_words).toarray()
# print(x_train,"\n",x_train.sum(axis=0))    # print(len(train_news_df), np.array(x_train).shape, len(y_train))
# print(x_test,"\n",x_test.sum(axis=0))      # print(len(test_news_df), np.array(x_test).shape, len(y_test))


###################################################
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

yhat = classifier.predict(x_test)
res=classifier.score(x_test,y_test)
print(res)
count=0
# for i in range(len(yhat)):
#     if(yhat[i]==0 and y_test[i]==5):
#         count+=1
# print(count)
from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test, yhat, labels=[0,1,2,3,4,5])

print("\n\n\n")
print(conf_mat)


doc_size = u.map_file_name(db_name)
filename0="bayers_size_"+doc_size+"_feature_"+str(feature_size)
print(filename0)





u.plot_confusion_matrix(conf_mat, classes=[0,1,2,3,4,5],normalize= True,  title=filename0)
u.plot_confusion_matrix(conf_mat, classes=[0,1,2,3,4,5],normalize= False,  title=filename0)

directory_name=f"./files/result/info"
os.makedirs(directory_name, exist_ok = True)
with open( directory_name + f"/{filename0}.txt",'w',encoding= 'utf-8') as f:
    f.write(f"{filename0} \n")
    f.write(f"feature_size: {feature_size} \n")
    f.write(f"all news size   : {train_news_df.shape[0]+test_news_df.shape[0]}\n")
    f.write(f"train news df shape : {train_news_df.shape}\n")
    f.write(f"test news df shape  : {test_news_df.shape}\n")
    f.write(f"accuracy   : {res}\n")
