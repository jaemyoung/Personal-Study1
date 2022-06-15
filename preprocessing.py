#load data
import pandas as pd
data = pd.read_csv("C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/filttered_data_0424.csv")

owner_list = ["microsoft","IBM","google"] #microsoft-Azure, google-tensorflow 등 같은 회사나타내는 것 표시
owner_4 = data[data['owner'].isin(owner_list)].reset_index()
data.set_index("full_name",inplace =True) #repo_name으로 인데스 변경
owner_4= data.loc[owner_3]# 찾고자하는 repo_name을 index로 지정

import pickle
file_path = "C:/Users/user/Documents/GitHub/오픈소스 기반 기업 역량평가/data/0502_repo_69.pickle"
with open(file_path,"wb") as fw:
    pickle.dump(owner_4, fw)   
    
owner_4["owner"].value_counts()
