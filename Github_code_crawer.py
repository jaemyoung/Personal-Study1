# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:07:10 2022

@author: user
"""
#data
"""
#파일 못받음 용량초과(5) -> total data : 144
owner_4["full_name"][12] #'microsoft/iot-curriculum'
owner_4["full_name"][61] #'microsoft/DirectML'
owner_4["full_name"][72] #'microsoft/onnxruntime'
owner_4["full_name"][117] #'IBM/MAX-Sports-Video-Classifier'
owner_4["full_name"][123] #'aws/sagemaker-python-sdk'
"""
data_owner = pd.read_csv("C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/filttered_data_0424.csv")
owner_list = ["microsoft","IBM","google","aws"] #microsoft-Azure, google-tensorflow 등 같은 회사나타내는 것 표시
owner_4 = data_owner[data_owner['owner'].isin(owner_list)].reset_index()
owner_4 = owner_4.drop(owner_4.index[[12,61,72,117,123]])
owner_4 = owner_4.reset_index() 



## load contents
import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/contents_owner_4_version1.pickle"
with open(file_path,"rb") as fr:
    repo_contents_owner_4 = pickle.load(fr)
    
from github import Github  
import re
# Github Enterprise with custom hostname
g = Github(login_or_token="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") #토큰 입력
#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 새롬누나
#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 민찬
#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 재명

###전처리
# repo에 코드 정보 가져오기
p1= re.compile('.*.py$|.*.java$|.*.js$|.*.php$|.*.rb$|.*.go$')# 6가지 
p_= re.compile('.*.py$')# 1가지 

#repo 에서 dictionary 중 __init__을 가진 dictionary만 뽑아서 경로와 content 저장 -> total data : 93
#reset_owner_4 = []
for repo in owner_4["full_name"][103:]:
    repos = g.get_repo(repo)
    contents = repos.get_contents("")
    content ={}
    path_list = []
    content_list =[]
    while contents:

        file_content = contents.pop(0)
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



