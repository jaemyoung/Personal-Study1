# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:33:07 2022

@author: user
"""

## load data
data_owner = pd.read_csv("C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/filttered_data_0424.csv")
owner_list = ["microsoft","IBM","google","aws"] #microsoft-Azure, google-tensorflow 등 같은 회사나타내는 것 표시
owner_4 = data[data['owner'].isin(owner_list)].reset_index()

## load contents
import pickle
import pandas as pd
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/0502_repo_71.pickle"
with open(file_path,"rb") as fr:
    data = pickle.load(fr)
    

data_concat= [str(row['repo']) +" "+ str(row['topics']).replace("#"," ")+" "+ str(row['description']) for idx, row in data.iterrows()]# 토픽 주석표시 제거
processed_data = text_preprocessing(data_concat)

#roberta
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base')

#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)


## load contents
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_embedding_vector.pickle"
with open(file_path,"rb") as fr:
    embedding_vector = pickle.load(fr)
    
#repo별 embedding vector 생성
embedding_vector = []
for repo in data_concat:
    meta_tokens=tokenizer.tokenize(repo ,max_length = 80,padding = "max_length",truncation =True)
    tokens=[tokenizer.cls_token]+meta_tokens+[tokenizer.sep_token]
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
    embedding_vector.append(context_embeddings)


#array 변환
one_di_array= []
for vector in embedding_vector:
    one_di_array.append(vector.view(-1).detach().numpy()) #1차원으로 변환
    

 #파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(roberta).pickle"
with open(file_path,"wb") as fw:
    pickle.dump(one_di_array, fw)   
    