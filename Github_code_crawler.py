# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:07:10 2022

@author: user
"""
#%% data 
import pandas as pd
data = pd.read_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/filttered_data_0424.csv",)
owner = data["owner"].value_counts()

microsoft_list = ["microsoft","Azure","Azure-Samples"]
IBM_list = ["IBM","IBM-Cloud","IBM-Watson-APIs"]
aws_list = ["aws","awslabs","aws-samples","aws-solutions"]
google_list = ["google","googleapis","deepmind","tensorflow","google-research"]
meta_list = ["Meta","pytorch","PyTorchLightning"]

microsoft_owner = data[data['owner'].isin(microsoft_list)]
IBM_owner = data[data['owner'].isin(IBM_list)]
aws_owner = data[data['owner'].isin(aws_list)]
google_owner = data[data['owner'].isin(google_list)]
meta_owner = data[data['owner'].isin(meta_list)]

microsoft_owner["big_owner"] = "microsoft"
IBM_owner["big_owner"] = "IBM"
aws_owner["big_owner"] = "aws"
google_owner["big_owner"] = "google"
meta_owner["big_owner"] = "meta"

owner5 = pd.concat([microsoft_owner,IBM_owner,aws_owner,google_owner,meta_owner])

#%% load contents
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_version1.pickle"
with open(file_path,"rb") as fr:
    repo_contents_owner_4 = pickle.load(fr)
#%% code crowler
from github import Github  
import re
# Github Enterprise with custom hostname
g = Github(login_or_token="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") #토큰 입력
#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 새롬누나
#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 민찬
#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 재명
#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 현호
#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 예빈

#repo 에서 dictionary 중 __init__을 가진 dictionary만 뽑아서 경로와 content 저장 -> total data : 93
#reset_owner_4 = []
for repo in owner5["full_name"][62:]:
    repos = g.get_repo(repo)
    contents = repos.get_contents("")
    content ={}
    path_list = []
    content_list =[]
    break
    while contents:

        file_content = contents.pop(0)# setup.py에서 find_packages(exclude = ?) 확인 후 제거
        if file_content.type == "dir":
            contents.extend(repos.get_contents(file_content.path))
        elif file_content.name == "__init__.py":
            path = "/".join(file_content.path.split("/")[:-1])
            content["repo_name"] = repo
            path_list.append(path)
            content["file_path"] = path_list
            content_list.extend(repos.get_contents(path))
            content["contents"] = content_list #이전 경로
    reset_owner_4.append(content)
    
repos.description

#파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/reset_owner_4.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(reset_owner_4, fw)   


##load data
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/reset_owner_4.pickle"
with open(file_path,"rb") as fr:
    reset_owner_4 = pickle.load(fr)


reset_owner_4 = list(filter(None, reset_owner_4)) # 빈 dic 제거 -> total : 93

#주석 있는 것들 docstring으로 저장
import re
p1= re.compile('.*\.py$|.*\.java$|.*\.js$|.*\.php$|.*\.rb$|.*\.go$')# 6가지 PL만 추출
p2= re.compile('(def .*? return)',re.DOTALL) #주석만 추출
p3= re.compile('(""".*?""")',re.DOTALL) #주석만 추출
p4= re.compile('(# .*?\n)',re.DOTALL) #주석만 추출


contents_owner_3 =[] #IBM, Microsoft, Google만 사용 -> total 69개 데이터만 사용
for repo in reset_owner_4:
    if repo["repo_name"] in owner_3:# total 69개 데이터만 사용
        
        for content in repo["contents"]: # contens 리스트 
            if len(p1.findall(content.name)) >= 1 : # 6가지 PL만 사용
                print(content.raw_data)
                for function in p2.findall(content.decoded_content.decode('utf-8')):
                    if  len(p3.findall(function)+p4.findall(function)) >= 1:
                        dic={}
                        dic["repo_name"] = repo["repo_name"]
                        dic["file_path"] = repo["file_path"]
                        dic["file_name"] = content.name
                        #dic["contents"] = content.decoded_content.decode('utf-8')
                        dic["code"] = function#.replace("".join(p3.findall(function)[0]),"") #주석 빼고 저장
                        dic["docstring"] = p3.findall(function) + p4.findall(function)
                        contents_owner_3.append(dic)

#파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_3.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(contents_owner_3, fw)

###############################################################################################

pd_reset_owner_4= pd.DataFrame(contents_owner_4)
pd_reset_owner_4["owner"] = pd_reset_owner_4["repo_name"].apply(lambda x: x.split("/")[0])
pd_reset_owner_4["owner"].value_counts()

#owner 별 갯수세기
pd_reset_owner_4_repo = pd.DataFrame(reset_owner_4)        
pd_reset_owner_4_repo["owner"] =  pd_reset_owner_4_repo["repo_name"].apply(lambda x: x.split("/")[0])
pd_reset_owner_4_repo["owner"].value_counts()

#file별 중복 확인
pd_owner_4 = pd.DataFrame(li)
pd_owner_4["owner"] = pd_owner_4["repo_name"].apply(lambda x: x.split("/")[0])
a = pd_owner_4[pd_owner_4["owner"]=="microsoft"]["file_name"].apply(lambda x: x.split(".")[0]).value_counts()
b = pd_owner_4[pd_owner_4["owner"]=="IBM"]["file_name"].apply(lambda x: x.split(".")[0]).value_counts()
c = pd_owner_4[pd_owner_4["owner"]=="google"]["file_name"].apply(lambda x: x.split(".")[0]).value_counts()
d = pd_owner_4[pd_owner_4["owner"]=="aws"]["file_name"].apply(lambda x: x.split(".")[0]).value_counts()

aa = pd_owner_4[(pd_owner_4["owner"]=="microsoft")&(pd_owner_4["file_name"].apply(lambda x : x.split(".")[0]=="version"))]
bb= pd_owner_4[(pd_owner_4["owner"]=="IBM")&(pd_owner_4["file_name"].apply(lambda x : x.split(".")[0]=="setup"))]
cc = pd_owner_4[(pd_owner_4["owner"]=="google")&(pd_owner_4["file_name"].apply(lambda x : x.split(".")[0]=="utils"))]
dd = pd_owner_4[(pd_owner_4["owner"]=="aws")&(pd_owner_4["file_name"].apply(lambda x : x.split(".")[0]=="utils"))]

data_owner[data_owner["language"]=="Python"]
#파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_version1.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(owner_4, fw)
    
 #파일 저장
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_version4.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(reset_owner_4, fw)   



