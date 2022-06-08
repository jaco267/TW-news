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