# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:47:33 2022

@author: user
"""


## load contents
import pickle
import pandas as pd
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_version4.pickle"
with open(file_path,"rb") as fr:
    contents_owner_4 = pickle.load(fr)


#데이터 전처리(docstring)
import nltk
for idx, dic in enumerate(contents_owner_4):
    doc = " ".join(dic["docstring"])
    token = nltk.regexp_tokenize(doc.lower(),'[A-Za-z]+')
    word = []
    for tok in token:
        if len(tok) >2:
            word.append(tok)
    contents_owner_4[idx]["docstring"] = word


#CodeBert
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoTokenizer, AutoModel

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
#model.to(device)

## load contents
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_embedding_vector.pickle"
with open(file_path,"rb") as fr:
    embedding_vector= pickle.load(fr)

#repo별 embedding vector 생성
#embedding_vector = []
for repo in contents_owner_3[2400:2600]:
    nl_tokens=tokenizer.tokenize(" ".join(repo["docstring"]),max_length = 16,padding = "max_length",truncation =True)
    code_tokens=tokenizer.tokenize(repo["code"],max_length = 32,padding = "max_length",truncation =True) #차원 같게해주려고
    tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+ code_tokens+[tokenizer.sep_token]
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
    embedding_vector.append(context_embeddings)

 #파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(codebert)_owner_3.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(embedding_vector, fw)   
    

#array 변환
one_di_array= []
for vector in embedding_vector:
    one_di_array.append(vector.view(-1).detach().numpy()) #1차원으로 변환
    
#repo 단위 vector 값 합치기
pd_contents_owner_4 = pd.DataFrame(contents_owner_4)
pd_contents_owner_4["vector"] = one_di_array
code_embedding_vector = list(pd_contents_owner_4.groupby(["repo_name"])["vector"].sum())
##
pd_contents_owner_3 = pd.DataFrame(contents_owner_3)
test = pd_contents_owner_3[:4000]
test["vector"] = one_di_array
test_vector = list(test.groupby(["repo_name"])["vector"].mean())

test_repo = list(test["repo_name"].unique())
test_repo = pd.DataFrame(test_repo)
test_repo["vector"] = test_vector
test_repo.set_index(0,inplace =True) #repo_name으로 인데스 변경

 #파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/embedding_vector(codebert)_sum.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(code_embedding_vector, fw)   
    
