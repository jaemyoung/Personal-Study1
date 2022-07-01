#load data
import pandas as pd
data = pd.read_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/filttered_data_0424.csv")

owner_list = ["microsoft","IBM","google"] #microsoft-Azure, google-tensorflow 등 같은 회사나타내는 것 표시
owner_4 = data[data['owner'].isin(owner_list)].reset_index()
data.set_index("full_name",inplace =True) #repo_name으로 인데스 변경
owner_4= data.loc[owner_3]# 찾고자하는 repo_name을 index로 지정

import pickle
file_path = "C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/0502_repo_69.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(owner_4, fw)   
    
owner_4["owner"].value_counts()

#%% owner 통합
#load data
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

#bar 그래프
import matplotlib.pyplot as plt
bar = plt.bar(["Microsoft","IBM","AWS","Google","Meta"],[108,60,41,66,13])
