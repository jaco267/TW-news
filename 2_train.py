#coding=utf-8

import numpy as np
import pandas as pd 

directory_name=f"./files/csv"

train_news_df = pd.read_csv( f"{directory_name}/train_news.csv")
test_news_df = pd.read_csv(f"{directory_name}/test_news.csv")


print(train_news_df.head())

print(train_news_df["label"].unique())

label_mapping = {'國際':0,'生活':1,'政治':2,'體育':3,'社會':4,'財經':5}
train_news_df["label"] = train_news_df['label'].map(label_mapping)
print(train_news_df.head())
test_news_df["label"] = test_news_df['label'].map(label_mapping)



stop_string="4000  539  7000  76  79  114  124  130  2005  2008  2010  2023  2025  2500  00  000  01  02  03  04  05  06  07  08  09  10  100  1000  101  105  106  107  108  109  11  110  1100216  1100217  1100218  1100219  112  113  119  12  120  1200  13  14  15  150  1500  16  17  18  180  19  1px  20  200  2000  2007  2009  2011  2012  2013  2014  2015  2016  2017  2018  2019  2020  2021  2022  2024  2030  21  22  228  23  24  25  250  26  27  28  29  30  300  3000  31  32  33  34  35  36  360  37  38  39  40  400  41  42  43  44  45  46  47  48  49  50  500  5000  51  52  53  54  55  56  57  58  59  5G  60  600  6000  61  62  63  64  65  66  67  68  69  70  700  71  72  73  74  75  77  78  80  800  8000  81  82  83  84  85  86  87  88  89  90  900  91  92  93  94  95  96  97  98  99  A9"
stop_list = stop_string.split("  ")
from sklearn.feature_extraction.text import CountVectorizer
def df_to_array(news_df):
    words=[]
    y_mat=[]
    for index in range(len(news_df)):
        content_list = news_df["content"][index].split("|")
        words.append(" ".join(content_list))
        y_mat.append(news_df["label"][index])
    return words, y_mat

x_train_words, y_train = df_to_array(train_news_df)

#########
cv = CountVectorizer(analyzer='word', max_features=4500, lowercase = False,stop_words=stop_list)
cv.fit(x_train_words)
#########

for feature in cv.get_feature_names_out():
    print(feature,end="  ")
print()    


x_train=cv.transform(x_train_words).toarray()
print(x_train,"\n",x_train.sum(axis=0))
print(len(train_news_df), np.array(x_train).shape, len(y_train))



x_test_words, y_test = df_to_array(test_news_df)
x_test=cv.transform(x_test_words).toarray()
print(x_test,"\n",x_test.sum(axis=0))
print(len(test_news_df), np.array(x_test).shape, len(y_test))





###################################################
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

yhat = classifier.predict(x_test)
res=classifier.score(x_test,y_test)
print(res)
count=0
for i in range(len(yhat)):
    if(yhat[i]==0 and y_test[i]==5):
        count+=1
print(count)
from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test, yhat, labels=[0,1,2,3,4,5])

print("\n\n\n")
print(conf_mat)

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import classification_report
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    label_mapping = {'國際0':0,'生活1':1,'政治3':2,'體育3':3,'社會4':4,'財經5':5}
    label_mapping = {y: x for x, y in label_mapping.items()}  #把 key value 相反過來˙
    def cat(x):
        return label_mapping[x]
    classes = list(map(cat,classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    fmt = 'd'# '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=[0,1,2,3,4,5],normalize= False,  title='Confusion matrix')
plt.show()