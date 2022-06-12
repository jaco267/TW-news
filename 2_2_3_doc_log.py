#coding=utf-8

import numpy as np
import pandas as pd 
import os
import utils.train_func as u

# db_name="TW_news_10"    #accuracy 0.87
db_name="TW_news_50"     #accuracy 0.84
# db_name="TW_news_15"    #accuracy 0.87
# db_name="TW_news" 
# directory_name=f"./files/csv"
directory_name=f"./files/{db_name}/csv/doc2vec_infer_new"
# directory_name=f"./files/{db_name}/csv/doc2vec_infer_new"



train_vec_df = pd.read_csv( f"{directory_name}/train_vec.csv")
test_vec_df = pd.read_csv(f"{directory_name}/test_vec.csv")

train_vec = train_vec_df.to_numpy()[:,1:]
test_vec  = test_vec_df.to_numpy()[:,1:]

print(train_vec.shape)
print(test_vec.shape)

y_train_df = pd.read_csv( f"{directory_name}/y_train.csv")
y_test_df = pd.read_csv(f"{directory_name}/y_test.csv")

y_train = y_train_df.to_numpy()[:,1:].ravel()
y_test = y_test_df.to_numpy()[:,1:].ravel()
print(y_train.shape)
print(y_test.shape)



#Logistic regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)
model = model.fit(train_vec,y_train)
yhat = model.predict(test_vec)


from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test, yhat, labels=[0,1,2,3,4,5])

print(conf_mat)

res=model.score(test_vec,y_test)
print(res)



doc_size = u.map_file_name(db_name)
# filename0="doc2vec_size_"+doc_size
filename0="doc2vec_infer_size_"+doc_size
print(filename0)


u.plot_confusion_matrix(conf_mat, classes=[0,1,2,3,4,5],normalize= True,  title=filename0)
u.plot_confusion_matrix(conf_mat, classes=[0,1,2,3,4,5],normalize= False,  title=filename0)





directory_name=f"./files/result/info"
os.makedirs(directory_name, exist_ok = True)
with open( directory_name + f"/{filename0}.txt",'w',encoding= 'utf-8') as f:
    f.write(f"{filename0} \n")
    f.write(f"all news size   : {train_vec.shape[0]+test_vec.shape[0]}\n")
    f.write(f"train news df shape : {train_vec.shape}\n")
    f.write(f"test news df shape  : {test_vec.shape}\n")
    f.write(f"accuracy   : {res}\n")