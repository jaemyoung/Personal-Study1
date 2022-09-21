# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 20:47:33 2022

@author: user
"""


#%% load contents
import pickle
import pandas as pd
file_path = "C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/contents_owner_4_version4.pickle"
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

# average pooling
def avg_pool(hidden_states, mask):
    length = torch.sum(mask, 1, keepdim=True).float()
    mask = mask.unsqueeze(2)
    hidden = hidden_states.masked_fill(mask == 0, 0.0)
    avg_hidden = torch.sum(hidden, 1) / length
        
    return avg_hidden


#%%CodeBert
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoTokenizer, AutoModel
"""
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
"""
tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")

    
linear_layer = torch.nn.Linear(768,1)
embedding_vector = []
for repo in contents_owner_4[1200:1460]:
    torch.cuda.empty_cache() #캐시 삭제
    nl_tokens=tokenizer.tokenize(" ".join(repo["docstring"]),max_length = 16,padding = "max_length",truncation =True)
    code_tokens=tokenizer.tokenize(repo["code"],max_length = 32,padding = "max_length",truncation =True)
    tokens= [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+ code_tokens+[tokenizer.sep_token]
    tokens_ids= tokenizer.convert_tokens_to_ids(tokens)
    context_vector=model(torch.tensor(tokens_ids)[None,:])[0]
    output = linear_layer(context_vector)
    embedding_vector.append(output)

"""
## load contents
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_embedding_vector.pickle"
with open(file_path,"rb") as fr:
    embedding_vector= pickle.load(fr)
    
#repo별 embedding vector 생성
embedding_vector = []
for repo in contents_owner_4[:1]:
    nl_tokens=tokenizer.tokenize(" ".join(repo["docstring"]),max_length = 16,padding = "max_length",truncation =True)
    code_tokens=tokenizer.tokenize(repo["code"],max_length = 32,padding = "max_length",truncation =True) #차원 같게해주려고
    tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+ code_tokens+[tokenizer.sep_token]
    tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
    embedding_vector.append(context_embeddings)
    break
"""

 #파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/embedding_vector(codebert+layer)_owner4.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(embedding_vector, fw)   
    

#array 변환
one_di_array= []
for vector in embedding_vector:
    one_di_array.append(vector.view(-1).detach().numpy()) #1차원으로 변환
    
#repo 단위 vector 값 합치기
pd_contents_owner_4 = pd.DataFrame(contents_owner_4)
pd_contents_owner_4["vector"] = one_di_array
code_embedding_vector = list(pd_contents_owner_4.groupby(["repo_name"])["vector"].mean())

#파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/embedding_vector(codebert+layer)_owner4_mean.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(code_embedding_vector, fw)   
    

#%% encoder만 사용 
from torch import nn
from transformers import T5Tokenizer, T5EncoderModel
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
model = T5EncoderModel.from_pretrained("Salesforce/codet5-small")

embedding_vector = []
for repo in contents_owner_4[100:500]:
    torch.cuda.empty_cache() #캐시 삭제
    input_ids = tokenizer(repo["code"], return_tensors="pt",max_length =256 ,padding ="max_length", truncation = True).input_ids
    attention_mask = tokenizer(repo["code"], return_tensors="pt",max_length =256 ,padding ="max_length", truncation = True).attention_mask  # Batch size 1
    outputs = model(input_ids=input_ids)
    last_hidden_states = outputs.last_hidden_state
    output = avg_pool(last_hidden_states, attention_mask)
    embedding_vector.append(output)

"""
# linear layer 추가
m = torch.nn.Linear(512, 256)
output = m(last_hidden_states).size()
"""
one_di_array= []
for vector in output:
    one_di_array.append(vector.view(-1).detach().numpy()) #1차원으로 변환
    
    
    