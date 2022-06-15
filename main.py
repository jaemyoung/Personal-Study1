# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:55:27 2022

@author: user
"""
#%% 데이터 불러오기
## load data
import pandas as pd
import numpy as np
import pickle

file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/0502_repo_71.pickle"
with open(file_path,"rb") as fr:
    data = pickle.load(fr)

file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(doc2vec).pickle"
with open(file_path,"rb") as fr:
    doc_embedding_vector = pickle.load(fr)
    
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(codebert).pickle"
with open(file_path,"rb") as fr:
    code_embedding_vector = pickle.load(fr)   
    
#file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(roberta).pickle"
#with open(file_path,"rb") as fr:
    #roberta_embedding_vector = pickle.load(fr)   
    
# concat data  
data["doc_embedding_vector"] = doc_embedding_vector
data["code_embedding_vector"] = code_embedding_vector
#data["roberta_embedding_vector"] = roberta_embedding_vector

total_embedding_vector = []
for index, row in data.iterrows():
    total_embedding_vector.append(np.concatenate((row["code_embedding_vector"],row["doc_embedding_vector"]),axis =None))
data["total_embedding_vector"] =total_embedding_vector
'''
data["total_embedding_vector"][39] =data["total_embedding_vector"][7]
data["total_embedding_vector"][63] =data["total_embedding_vector"][7]
data["total_embedding_vector"][69] =data["total_embedding_vector"][7]
'''
"""
total_embedding_vector = []
for index, row in data.iterrows():
    total_embedding_vector.append(np.concatenate((row["code_embedding_vector"],row["roberta_embedding_vector"]),axis =None))
data["total_embedding_vector"] =total_embedding_vector
"""
#%% 시각화 함수 정의
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def euclidean_sim(A, B):
    return distance.euclidean(A,B)
    

def make_tSNE(one_di_array,n_cluster,owner):
    simliarity_vector = []
    for vector in one_di_array:
        s= []
        for v in one_di_array:
            s.append(cos_sim(vector,v))
        simliarity_vector.append(s)
            
    E = pd.DataFrame(simliarity_vector)
    kmeans = KMeans(n_clusters=n_cluster).fit(simliarity_vector)
    TSNE_vector = TSNE(n_components=2).fit_transform(simliarity_vector)  # component = 차원
    Q = pd.DataFrame(TSNE_vector)  # dataframe으로 변경하여 K-means cluster lavel 열 추가
    Q["clusters"] = kmeans.labels_
   # Q["owner"] = owner["owner"]# label 추가
   # Q["repo_name"] =owner["repo_name"]
    Q["repo_name"] =owner.index
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.scatterplot(data=Q, x=0, y=1, hue= hier["Cluster"], palette='deep')
    plt.show()
    print("visualizer complete!")
    return Q

def cos_similarity_vector(one_di_array):
    simliarity_vector = []
    for vector in one_di_array:
        s= []
        for v in one_di_array:
            s.append(cos_sim(vector,v))
        simliarity_vector.append(s)
    return simliarity_vector

# Hierarchical clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def make_hierarchi_clustering(vector_sum, cut_off):
    vector_list = list(vector_sum)
    linked = linkage(vector_list, method = "ward", metric = "euclidean") #single, complete, average, weighted, centroid
    labelList = list(vector_sum.index)
    predict = pd.DataFrame(fcluster(linked, cut_off, criterion='distance'))
    predict["repo_name"] = labelList
    plt.figure(figsize=(30, 24))
    dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=False, leaf_rotation = 90,  leaf_font_size=20)
    plt.show()
    return predict

#%% 시각화 적용
hier.columns = ["Cluster","repo_name"]
hier = make_hierarchi_clustering(data["total_embedding_vector"], 150)
tsne1 = make_tSNE(data.reset_index()["total_embedding_vector"],4,data.reset_index())

data = data.drop(["aws/aws-neuron-sdk","aws/sagemaker-inference-toolkit"])
#bar 그래프

bar = plt.bar(data["owner"].value_counts().index,data["owner"].value_counts(),color = "navy")
plt.ylim(0, 40)
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%1.f' % height, ha='center', va='bottom', size = 12)
    
#%% 파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/cluster_4.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(hier, fw)

hier = hier.reset_index()
hier["owner"] = hier["repo_name"].apply(lambda x: x.split("/")[0])
data = data.reset_index()

hier[hier["Cluster"] != 1] =2

#%% 평가지표 계산
#전체 평균 포크수
32
#cluster 2 = 461.97
#cluster 3 = 422.33
#cluster 4 = 668.86
data.loc[hier["Cluster"]==1,"forks_count"].sum()/len(data.loc[hier["Cluster"]==1,"forks_count"])
#기업 평균 포크수

#2 IBM 45 m 747.43 g 256.66
#3 115 460.75
#4 59 924 3

data.loc[hier[(hier["Cluster"]==2)&(hier["owner"]=="IBM")].index,"forks_count"].sum()/len(data.loc[hier[(hier["Cluster"]==2)&(hier["owner"]=="IBM")].index,"forks_count"])

data.loc[hier[(hier["Cluster"]==1)&(hier["owner"]=="microsoft")].index,"forks_count"].sum()/len(data.loc[hier[(hier["Cluster"]==1)&(hier["owner"]=="microsoft")].index,"forks_count"])

data.loc[hier[(hier["Cluster"]==1)&(hier["owner"]=="google")].index,"forks_count"].sum()/len(data.loc[hier[(hier["Cluster"]==1)&(hier["owner"]=="google")].index,"forks_count"])

    
#전체 평균 스타수
#cluster 2 = 2539.36
#cluster 3 = 2904.33
#cluster 4 = 2625.14
data.loc[hier["Cluster"]==2,"stargazers_count"].sum()/len(data.loc[hier["Cluster"]==2,"stargazers_count"])
#기업 평균 스타수

#2 1583 3964.625 1912.666
#3 1220 3114
#4 355 3597  36

data.loc[hier[(hier["Cluster"]==2)&(hier["owner"]=="IBM")].index,"stargazers_count"].sum()/len(data.loc[hier[(hier["Cluster"]==2)&(hier["owner"]=="IBM")].index,"stargazers_count"])

data.loc[hier[(hier["Cluster"]==2)&(hier["owner"]=="microsoft")].index,"stargazers_count"].sum()/len(data.loc[hier[(hier["Cluster"]==2)&(hier["owner"]=="microsoft")].index,"stargazers_count"])

data.loc[hier[(hier["Cluster"]==1)&(hier["owner"]=="google")].index,"stargazers_count"].sum()/len(data.loc[hier[(hier["Cluster"]==1)&(hier["owner"]=="google")].index,"stargazers_count"])



1*(59/668.86)*(355/2625.14)
5*(924/668.86)*(3597/2625.14)
1*(3/668.86)*(36/2625.14)














