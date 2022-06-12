import numpy as np
import os
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import classification_report
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig=plt.figure(figsize=(8,7),dpi=150)
    fig.set_figheight(5.6);
    fig.set_figwidth(7);
    
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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(left=0.152,bottom=0.176,right=0.97,top=0.939)
    


    num="0"
    if normalize:
        num="1"
    directory_name=f"./files/result/image/{num}"
    os.makedirs(directory_name, exist_ok = True)
    plt.savefig(f'{directory_name}/{title+"_"+num}.png')
    # plt.show()



def map_file_name(db_name):
    if db_name=="TW_news":
        doc_size="6k"
    elif db_name == "TW_news_10":
        doc_size="60k"
    elif db_name == "TW_news_15":
        doc_size="100k"
    elif db_name == "TW_news_50":
        doc_size="300k"
    elif db_name == "TW_news_100":
        doc_size="600k"
    return doc_size