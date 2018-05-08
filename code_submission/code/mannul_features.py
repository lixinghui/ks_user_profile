

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,norm
from tqdm import tqdm  
import re


def Extract_mannul_features():

	X_train=pd.read_csv('../data/X_train.csv',low_memory=False)
	X_test=pd.read_csv('../data/X_test.csv',low_memory=False)
	Y_train=pd.read_csv('../data/Y_train.csv',low_memory=False)
	Y_pred=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',low_memory=False,encoding="gbk")
	
	print("The shape of training data is {}".format(X_train.shape))
	print("The shape of testing data is {}".format(X_test.shape))	
	
	assert np.sum(X_train["vid"]!=Y_train["vid"])==0
	assert np.sum(X_test["vid"]!=Y_pred["vid"])==0
	
	
	all_data=pd.concat([X_train,X_test],axis=0)

	# ### 删除缺失值超过97%的feature

	to_drop=all_data.isnull().sum().sort_values(ascending=False)/len(all_data)*100
	to_drop=to_drop[to_drop>97.0]
	all_data.drop(to_drop.index,axis=1,inplace=True)


	# ### 根据其他项信息提取性别Feature

	gender=[]
	all_data["0101"]=all_data["0101"].astype(str)
	all_data["0102"]=all_data["0102"].astype(str)
	all_data["0539"]=all_data["0539"].astype(str)
	all_data["0120"]=all_data["0120"].astype(str)
	all_data["0121"]=all_data["0121"].astype(str)
	all_data["0929"]=all_data["0929"].astype(str)
	for i in range(all_data.shape[0]):

		if "乳腺" in all_data["0101"].values[i]:
		    gender.append("F")
		elif "乳房" in all_data["0101"].values[i]:
		    gender.append("F")
		elif "乳腺" in all_data["0102"].values[i]:
		    gender.append("F")
		elif "子宫" in all_data["0102"].values[i]:
		    gender.append("F")
		elif "乳腺" in all_data["0121"].values[i]:
		    gender.append("F")
		elif "子宫" in all_data["0121"].values[i]:
		    gender.append("F")
		elif "阴道" in all_data["0539"].values[i]:
		    gender.append("F")
		elif "妇科" in all_data["0539"].values[i]:
		    gender.append("F")
		elif "宫颈" in all_data["0539"].values[i]:
		    gender.append("F")
		elif "乳腺" in all_data["0929"].values[i]:
		    gender.append("F")
		elif "小叶增生" in all_data["0929"].values[i]:
		    gender.append("F")
		elif "前列腺" in all_data["0102"].values[i]:
		    gender.append("M")
		elif "前列腺" in all_data["0120"].values[i]:
		    gender.append("M")
		else:
		    gender.append("unkown")
	all_data["gender"]=gender


	##根据一些疾病发病情况推断年龄情况


	age=[]
	all_data["3601"]=all_data["3601"].astype(str)
	all_data["0102"]=all_data["0102"].astype(str)
	all_data["0709"]=all_data["0709"].astype(str)
	all_data["0730"]=all_data["0730"].astype(str)
	all_data["0120"]=all_data["0120"].astype(str)
	all_data["A202"]=all_data["A202"].astype(str)
	all_data["0409"]=all_data["0409"].astype(str)
	all_data["1102"]=all_data["1102"].astype(str)
	all_data["1103"]=all_data["1103"].astype(str)
	all_data["1308"]=all_data["1308"].astype(str)
	all_data["0546"]=all_data["0546"].astype(str)
	all_data["0984"]=all_data["0984"].astype(str)
	for i in tqdm(range(all_data.shape[0])):

		if "骨质增生" in all_data["1102"].values[i]:  ###骨质增生与疏松
		    age.append("old")
		elif "退行性变" in all_data["1102"].values[i]:
		    age.append("old")
		elif "骨质增生" in all_data["1103"].values[i]:
		    age.append("old")
		elif "骨质疏松" in all_data["3601"].values[i]:
		    age.append("old")
		elif "减少" in all_data["3601"].values[i]:
		    age.append("old")
		elif "骨密度降低" in all_data["3601"].values[i]:
		    age.append("old")
		elif "绝经" in all_data["0546"].values[i]:    ###绝经情况
		    age.append("old")
		elif "闭经" in all_data["0546"].values[i]:
		    age.append("old")
		elif "停经" in all_data["0546"].values[i]:
		    age.append("old")
		elif "绝经" in all_data["0102"].values[i]:
		    age.append("old")
		elif "高血压" in all_data["0409"].values[i]:   ####三高病史
		    age.append("old")
		elif "糖尿病" in all_data["0409"].values[i]:
		    age.append("old")
		elif "冠心病" in all_data["0409"].values[i]:
		    age.append("old")
		    
		elif "增大" in all_data["0120"].values[i]:   ###前列腺增大
		    age.append("old")
		elif "义齿" in all_data["0709"].values[i]:   ###是否有义齿
		    age.append("old")
		    
		elif "老年环" in all_data["1308"].values[i]:   ###老年眼科病
		    age.append("old")
		elif "白内障" in all_data["1308"].values[i]:   
		    age.append("old")
		elif "玻璃体浑浊" in all_data["1308"].values[i]:   
		    age.append("old")
		elif "增生" in all_data["0984"].values[i]:   
		    age.append("old")
		elif "lmp" in all_data["0546"].values[i].lower():
		    age.append("young")
		elif "月经" in all_data["0546"].values[i]:
		    age.append("young")
		elif "哺乳" in all_data["0546"].values[i]:
		    age.append("young")
		else:
		    age.append("unknown")
	all_data["age"]=age


	### 提取每个人的检查项目数量

	all_data["num_items"]=all_data.isnull().sum(axis=1)




	##将正常数据（表达不同），均映射为形同的值
	all_data=all_data.astype(str)
	for col in tqdm(all_data.columns):
		all_data[col]=all_data[col].replace({"弃查":np.nan,           ## 将result数据做基本处理.相同意义的数据替换
		                                     "正常 正常":'正常',
		                                     "未见异常 未见异常":'正常',
		                                     "未触及 未触及":"正常",
		                                         "未见异常":"正常",
		                                         "未见明显异常":"正常",
		                                        "未见异常，活动自如":"正常",
		                                        "健康":"正常",
		                                        "整齐":"正常",
		                                        "详见纸质报告":np.nan,
		                                        "未查":np.nan,
		                                        "未触及":"正常",
		                                        "正常心电图":"正常",
		                                         "窦性心律正常心电图 ":"正常",
		                                        "骨量正常":"正常",
		                                        "耳鼻喉检查未见异常":"正常",
		                                        "外科检查未发现明显异常":"正常",
		                                        "内科检查未发现明显异常":"正常",
		                                        "右附件区未见明显异常回声":"正常",
		                                        "胰腺大小、形态正常，边缘规整，内部回声均匀，胰管未见扩张。":"正常",
		                                        "右肾大小、形态正常，包膜光滑，肾实质回声均匀，集合系统未见明显分离。":"正常",
		                                        "左肾大小、形态正常，包膜光滑，肾实质回声均匀，集合系统未见明显分离。":"正常",
		                                        "胆囊大小、形态正常，囊壁光整，囊腔内透声好，胆总管无扩张。":"正常",
		                                        "膀胱充盈良好，壁光滑，延续性好，其内透声性良好，未见明显占位性病变。":"正常",
		                                        "脾脏大小、形态正常，包膜光整，回声均匀。":"正常",
		                                        "脾脏大小、形态正常，包膜光整，内光点均匀。":"正常",
		                                        "右附件区未见明显异常回声。":"正常",
		                                        "左附件区未见明显异常回声。":"正常",
		                                        "肝、胆、胰、脾、左肾、右肾未发现明显异常":"正常",
		                                        "肝脏大小、形态正常，包膜光整，肝内血管走行较清晰，回声均匀。":"正常",
		                                        "前列腺大小、形态正常，包膜光滑完整，两侧对称，内部回声均匀。":"正常",
		                                        "甲状腺形态大小正常，边界清晰，内部回声分布均匀，未见明显异常回声。":"正常",
		                                        "双侧甲状腺大小形态正常，包膜光整，实质回声均匀，未见明显异常回声。CDFI：血流显示未见异常。":"正常",
		                                        "胸廓对称，双肺纹理清晰，走行自然，未见异常实变影，双肺门不大。纵隔窗示纵隔无偏移，心影及大血管形态正常，纵隔内未见肿块及肿大淋巴结。胸腔内未见积液。":"正常",
		                                        "脾脏大小测值正常，回声均匀，脾静脉测值正常。":"正常",
		                                        "甲状腺彩超未发现明显异常":"正常",
		                                        "肝脏大小、形态正常，包膜光整，肝内血管走行较清晰，光点分布尚均匀，其内未见明显异常光团。":"正常",
		                                        "无特殊记载":"正常",
		                                        "胰腺头、体、尾大小测值正常，内回声均匀。":"正常",
		                                        "前列腺未发现明显异常":"正常",
		                                        "双侧颈总动脉管径对称，内中膜不增厚,血流速度正常。双侧颈总动脉分叉处管径对称，内中膜不增厚，血流速度正常。双侧颈内、外动脉管径对称，管壁回声正常，血流速度正常。":"正常",
		                                        "回声正常，血流速度正常。":"正常",
		                                        "胆囊大小正常，壁光滑，腔内暗区清晰，胆总管测值正常范围。":"正常"})



	all_data.to_csv("../data/all_data_pivot.csv",index=False)


