
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,norm
from tqdm import tqdm  
import re


def Preprocessing():
	# ### 导入数据Y 并处理异常值

	Y=pd.read_csv('../data/meinian_round1_train_20180408.csv',engine='python',encoding="gbk")
	Y_pred=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',engine='python',encoding="gbk")

	Y.columns=["vid","Systolic",'Diastolic','Glycerin','HDC','LDC']
	Y_pred.columns=["vid","Systolic",'Diastolic','Glycerin','HDC','LDC']

	m_test=Y_pred.shape[0]
	m_train=Y.shape[0]


	columns=['Systolic','Diastolic','Glycerin']
	for col in columns:
		temp=[]
		for i in range(m_train):
		    pattern = re.compile(r'\d+\.{0,1}\d+')   ##数值中间最多允许出现一个小数点
		    try:
		        temp.append(pattern.findall(Y[col][i])[0])
		    except:
		        temp.append(np.nan)
		Y[col]=temp
		Y[col]=Y[col].astype("float32")
		Y[col]=Y[col].fillna(Y[col].mean())

	Y=Y[Y["Diastolic"]<200]        ## 删除异常值
	Y["LDC"]=np.abs(Y["LDC"])      ## 将负值取绝对值
	Y=Y.set_index("vid")
	Y_pred=Y_pred.set_index("vid")


	# ### 导入X  并生成数据透视表并合并


	X=[]
	with open("../data/meinian_round1_data_part1_20180408.txt","r") as f:
		for line in f.readlines():
		    x=line.strip().split("$")
		    X.append(x)

	X1=pd.DataFrame(X[1:],columns=["vid","table_id","field_results"])

	X=[]
	with open("../data/meinian_round1_data_part2_20180408.txt","r") as f:
		for line in f.readlines():
		    x=line.strip().split("$")
		    X.append(x)
	X2=pd.DataFrame(X[1:],columns=["vid","table_id","field_results"])

	X=pd.concat([X1,X2],axis=0)

	### 生成数据透视表
	X_all=pd.pivot_table(X,index='vid',
		                   columns='table_id', 
		                   values='field_results',
		                   fill_value=np.nan,
		                   aggfunc=lambda x: ' '.join(x))


	# ### 只提取train和test中有的数据并将其合并,一起进行数值处理

	X_train=X_all.loc[Y.index]
	X_test=X_all.loc[Y_pred.index]

	X_train.to_csv('../data/X_train.csv',index=True)
	X_test.to_csv('../data/X_test.csv',index=True)
	Y.to_csv("../data/Y_train.csv",index=True)


