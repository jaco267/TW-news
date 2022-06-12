import os
directory_name=f"./files/result/info"
import re

files_names = os.listdir(directory_name)


bayers_files = []
doc2vec_infer_files = []
doc2vec_files = []
for file in files_names:
    print(file)
    if "bayers" and "4500" in file:
        bayers_files.append(file)
    elif "doc2vec_infer" in file:
        doc2vec_infer_files.append(file)
    elif "doc2vec" in file:
        doc2vec_files.append(file)

print("\n\n")
print(bayers_files)
print(doc2vec_infer_files)
print(doc2vec_files)





bayers_list = []
for file in bayers_files:
    numbers = re.findall(r'\d+',file)
    doc_size = numbers[0]
    feature_size = numbers[1]
    file_info = {"doc_size":0,"accuracy":0}
    with open( directory_name + "/" +file,'r',encoding= 'utf-8') as rf:
        for line in rf:
            if "all news size" in line:
                doc_size = line.split(":")
                print(doc_size[1].strip())
                file_info["doc_size"] = doc_size[1].strip()
            if "accuracy" in line:
                accuracy = (float(line.split()[2]))
                file_info["accuracy"] = accuracy 
    bayers_list.append(file_info)

print(bayers_list)



doc2vec_infer_list = []
for file in doc2vec_infer_files:
    # print(file)
    numbers = re.findall(r'\d+',file)
    doc_size = numbers[1]
    file_info = {"doc_size":0,"accuracy":0}
    with open( directory_name + "/" +file,'r',encoding= 'utf-8') as rf:
        for line in rf:
            if "all news size" in line:
                doc_size = line.split(":")
                print(doc_size[1].strip())
                file_info["doc_size"] = doc_size[1].strip()
            if "accuracy" in line:
                accuracy = (float(line.split()[2]))
                file_info["accuracy"] = accuracy 
    doc2vec_infer_list.append(file_info)

print(doc2vec_infer_list)


doc2vec_list = []
for file in doc2vec_files:
    print(file)
    numbers = re.findall(r'\d+',file)
    doc_size = numbers[1]
    file_info = {"doc_size":0,"accuracy":0}
    
    with open( directory_name + "/" +file,'r',encoding= 'utf-8') as rf:
        for line in rf:
            if "all news size" in line:
                doc_size = line.split(":")
                print(doc_size[1].strip())
                file_info["doc_size"] = doc_size[1].strip()
            if "accuracy" in line:
                accuracy = (float(line.split()[2]))
                file_info["accuracy"] = accuracy 
    doc2vec_list.append(file_info)

print(doc2vec_list)











import matplotlib.pyplot as plt
import numpy as np


def plot_array (algor_list,label):
    x_list = [int(o["doc_size"]) for o in algor_list]
    y_list = [o["accuracy"] for o in algor_list]
    x_list, y_list = zip(*sorted(zip(x_list, y_list)))
    print(x_list,y_list)
    xpoints = np.array(x_list)
    ypoints = np.array(y_list)

    plt.plot(xpoints, ypoints,linestyle='-',marker='o',label=label)
    # plt.plot(xpoints, ypoints,label=label)
  
plot_array(bayers_list,"bayers")
plot_array(doc2vec_infer_list,"doc2vec_infer_new")
plot_array(doc2vec_list,"doc2vec")
plt.legend()
plt.show()

