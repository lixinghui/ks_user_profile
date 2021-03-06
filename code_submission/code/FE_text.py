

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,norm
from tqdm import tqdm  
import re

def FE_text_data():

	all_data=pd.read_csv("../data/all_data_outlier_done.csv",low_memory=False)
	all_data=all_data.set_index("vid")
	
	Y_pred=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',low_memory=False,encoding="gbk")
	Y_pred=Y_pred.set_index("vid")
	
	m_test=Y_pred.shape[0]


	# 用replace 及map函数处理简单（category少）离散变量

	all_data["30007"]=all_data["30007"].replace({"Ⅱ":2,"Ⅲ":3,"Ⅰ":1,"Ⅳ":4,"Ⅱ":2,"Ⅲ":3,"Ⅰ":1,"Ⅳ":4,"Ⅱ度":2,
		                                        "II":2,"Ⅲ度":3,"正常":0,"中度":2,"III":3,"ii°":2,"iii°":3,"Ⅰ°":1,
		                                        "Ⅱ°":2,"Ⅰ度":1,"Ⅳ度":4,"见TCT":0,"yellow":0,"-":0,"结果见TCT":0,
		                                        "Ⅳ°":4,"阴性":0,"微混":0,"Ⅲ°":3,"+":2,"I":1,"见刮片":0,"Ⅱv":2,
		                                        "Ⅰ Ⅰ":1,"iv°":4,"i°":1}).astype("float")

	all_data["0431"]=all_data["0431"].replace({"无":"正常","无 无":"正常","无压痛点":"正常","未见异常 未见异常":"正常",np.nan:None})

	all_data["0976"]=all_data["0976"].replace({"无":"正常","无 无":"正常",np.nan:None})
	all_data["3400"]=all_data["3400"].map({"透明":"透明","浑浊":"浑浊","混浊":"浑浊","微混":"浑浊"})
	all_data["3400"].fillna("透明",inplace=True)
	all_data["0215"]=all_data["0215"].map({np.nan:None,"正常":"正常"})
	all_data["0215"]=all_data["0215"].fillna("异常")
	all_data["0216"]=all_data["0216"].map({np.nan:None,"正常":"正常","正常 正常":"正常","未见异常 未见异常":"正常"})
	all_data["0216"]=all_data["0216"].fillna("异常")
	all_data["0217"]=all_data["0217"].map({np.nan:None,"正常":"正常","正常 正常":"正常","未见异常 未见异常":"正常"})
	all_data["0217"]=all_data["0217"].fillna("异常")
	all_data["0405"]=all_data["0405"].map({np.nan:None,"未闻及":"正常","正常":"正常","无":"正常","未见异常 未见异常":"正常"})
	all_data["0405"]=all_data["0405"].fillna("异常")
	all_data["0406"]=all_data["0406"].map({np.nan:None,"未触及 未触及":"正常","正常":"正常","未及":"正常",
		                                   "未见异常 未见异常":"正常"})
	all_data["0406"]=all_data["0406"].fillna("异常")
	all_data["0407"]=all_data["0407"].map({np.nan:None,"未触及 未触及":"正常","未及":"正常","正常":"正常",
		                                   "未见异常 未见异常":"正常","未触及":"正常","不大":"正常",})
	all_data["0407"]=all_data["0407"].fillna("异常")
	all_data["0420"]=all_data["0420"].map({np.nan:None,"未闻及异常":"正常","正常 正常":"正常","正常":"正常",
		                                   "未见异常 未见异常":"正常","有力":"正常",})
	all_data["0420"]=all_data["0420"].fillna("异常")




	all_data["0409"]=all_data["0409"]+all_data["0434"]  ## 两次病史检查的数据合并处理，避免重复统计
	temp=[]
	all_data["0409"]=all_data["0409"].astype("str")
	for i in range(len(all_data)):
		if "高血压" in all_data["0409"][i]:
		    temp.append(1)
		elif "血压偏高" in all_data["0409"][i]:
		    temp.append(1)
		elif all_data["0409"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["高血压史"]=temp

	temp=[]
	for i in range(len(all_data)):
		if "高血脂" in all_data["0409"][i]:
		    temp.append(1)
		elif "血脂偏高" in all_data["0409"][i]:
		    temp.append(1)
		elif all_data["0409"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["高血脂史"]=temp
		                    
	temp=[]
	for i in range(len(all_data)):
		if "糖尿病" in all_data["0409"][i]:
		    temp.append(1)
		elif "血糖偏高" in all_data["0409"][i]:
		    temp.append(1)
		elif all_data["0409"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["糖尿病史"]=temp

	temp=[]
	for i in range(len(all_data)):
		if "冠心病" in all_data["0409"][i]:
		    temp.append(1)
		elif all_data["0409"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["冠心病史"]=temp
		                    
	temp=[]
	for i in range(len(all_data)):
		if "肝" in all_data["0409"][i]:
		    temp.append(1)
		elif all_data["0409"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["肝病史"]=temp

		                    
	temp=[]
	all_data["0439"]=all_data["0439"].astype("str")
	for i in range(len(all_data)):
		if "冠心病" in all_data["0439"][i]:
		    temp.append(1)
		elif all_data["0439"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["父母冠心病"]=temp
		                    
	temp=[]
	for i in range(len(all_data)):
		if "高血压" in all_data["0439"][i]:
		    temp.append(1)
		elif all_data["0439"][i]=="nan":
		    temp.append(np.nan)    
		
		else:
		    temp.append(0)
	all_data["父母高血压"]=temp
		                    
	temp=[]
	for i in range(len(all_data)):
		if "糖尿病" in all_data["0439"][i]:
		    temp.append(1)
		elif all_data["0439"][i]=="nan":
		    temp.append(np.nan)
		else:
		    temp.append(0)
	all_data["父母糖尿病"]=temp

	columns=["4001"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "重度减弱" in all_data[col][i]:
		        temp.append(3)
		    elif "中度减弱" in all_data[col][i]:
		        temp.append(2)
		    elif "减弱" in all_data[col][i]:
		        temp.append(1)
		    elif "轻度硬化" in all_data[col][i]:
		        temp.append(1)
		    elif "硬化" in all_data[col][i]:
		        temp.append(2)
		    elif "钙化" in all_data[col][i]:
		        temp.append(3)
		    elif all_data[col][i]=="nan":
		        temp.append(np.nan)
		    else:
		        temp.append(0)
		all_data[col]=temp

	columns=["0436"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "无过敏" in all_data[col][i]:
		        temp.append("正常")
		    elif "过敏史不详" in all_data[col][i]:
		        temp.append("正常")
		    elif "过敏" in all_data[col][i]:
		        temp.append("过敏")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp
		
	columns=["1402"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "增快" in all_data[col][i]:
		        temp.append("增快")
		    elif "减慢" in all_data[col][i]:
		        temp.append("减慢")
		    elif "弹性降低" in all_data[col][i]:
		        temp.append("弹性降低")
		    elif "顺应性降低" in all_data[col][i]:
		        temp.append("顺应性降低")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp

		
	columns=["A705"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "脂肪肝" in all_data[col][i]:
		        temp.append("脂肪肝")
		    elif "脂肪含量超过正常值" in all_data[col][i]:
		        temp.append("脂肪肝")
		    elif "硬度值偏高" in all_data[col][i]:
		        temp.append("肝硬化")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp
		
		
	columns=["0987"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "术后" in all_data[col][i]:
		        temp.append("术后")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp
		
	columns=["0984"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "增生" in all_data[col][i]:
		        temp.append("增生")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp
		
	columns=["1308","1316","1330"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "动脉硬化" in all_data[col][i]:
		        temp.append("动脉硬化")
		    elif "黄斑" in all_data[col][i]:
		        temp.append("病变")
		    elif "弧形斑" in all_data[col][i]:
		        temp.append("病变")
		    elif "色素斑" in all_data[col][i]:
		        temp.append("病变")
		    elif "病变" in all_data[col][i]:
		        temp.append("病变")
		    elif "豹纹状眼底" in all_data[col][i]:
		        temp.append("病变")
		    elif "结膜炎" in all_data[col][i]:
		        temp.append("结膜炎")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp




	columns=["0113","0114","0115","0117","0118","0120","0121","0122","0123","0124"]
	for col in columns:
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "高回声" in all_data[col][i]:
		        temp.append("高回声")
		    elif "强回声" in all_data[col][i]:
		        temp.append("高回声")
		    elif "低回声" in all_data[col][i]:
		        temp.append("低回声")
		    elif "弱回声" in all_data[col][i]:
		        temp.append("低回声")
		    elif "无回声" in all_data[col][i]:
		        temp.append("无回声")
		    elif "弥漫性" in all_data[col][i]:
		        temp.append("弥漫性")
		    elif "欠清晰" in all_data[col][i]:
		        temp.append("欠清晰")
		    elif all_data[col][i]=="nan":
		        temp.append(None)
		    else:
		        temp.append("正常")
		all_data[col]=temp



	temp=[]
	all_data["0421"]=all_data["0421"].astype("str")
	for i in range(len(all_data)):
		if "早搏" in all_data["0421"][i]:
		    temp.append("早搏")
		elif "房颤" in all_data["0421"][i]:
		    temp.append("房颤")
		elif "过速" in all_data["0421"][i]:
		    temp.append("过速")
		elif "过缓" in all_data["0421"][i]:
		    temp.append("过缓")
		elif "不齐" in all_data["0421"][i]:
		    temp.append("不齐")
		elif all_data["0421"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0421"]=temp

	temp=[]
	all_data["3601"]=all_data["3601"].astype("str")
	for i in range(len(all_data)):
		if "严重骨质疏松" in all_data["3601"][i]:
		    temp.append("严重骨质疏松")
		elif "疏松" in all_data["3601"][i]:
		    temp.append("疏松")
		elif "减少" in all_data["3601"][i]:
		    temp.append("减少")
		elif "降低" in all_data["3601"][i]:
		    temp.append("降低")
		elif all_data["3601"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["3601"]=temp



	temp=[]
	all_data["0426"]=all_data["0426"].astype("str")
	for i in range(len(all_data)):
		if "收缩期杂音" in all_data["0426"][i]:
		    temp.append("收缩期杂音")
		elif "舒张期杂音" in all_data["0426"][i]:
		    temp.append("舒张期杂音")
		elif all_data["0426"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0426"]=temp

	temp=[]
	all_data["0435"]=all_data["0435"].astype("str")
	for i in range(len(all_data)):
		if "腹部有压痛" in all_data["0435"][i]:
		    temp.append("腹部有压痛")
		elif all_data["0435"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0435"]=temp

	temp=[]
	all_data["0702"]=all_data["0702"].astype("str")
	for i in range(len(all_data)):
		if "缺齿" in all_data["0702"][i]:
		    temp.append("缺齿")
		elif all_data["0702"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0702"]=temp

	temp=[]
	all_data["0730"]=all_data["0730"].astype("str")
	for i in range(len(all_data)):
		if "义齿" in all_data["0730"][i]:
		    temp.append("义齿")
		elif "有" in all_data["0730"][i]:
		    temp.append("义齿")
		elif all_data["0730"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0730"]=temp

	temp=[]
	all_data["1328"]=all_data["1328"].astype("str")
	for i in range(len(all_data)):
		if "色弱" in all_data["1328"][i]:
		    temp.append("色弱")
		elif "色盲" in all_data["1328"][i]:
		    temp.append("色盲")
		elif all_data["1328"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["1328"]=temp

	temp=[]
	all_data["0210"]=all_data["0210"].astype("str")
	for i in range(len(all_data)):
		if "鼻炎" in all_data["0210"][i]:
		    temp.append("鼻炎")
		elif "鼻窦炎" in all_data["0210"][i]:
		    temp.append("鼻窦炎")
		elif "息肉" in all_data["0210"][i]:
		    temp.append("息肉")
		elif "大" in all_data["0210"][i]:
		    temp.append("大")
		elif all_data["0210"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0210"]=temp

	temp=[]
	all_data["0423"]=all_data["0423"].astype("str")
	for i in range(len(all_data)):
		if "粗" in all_data["0423"][i]:
		    temp.append("粗")
		elif "弱" in all_data["0423"][i]:
		    temp.append("弱")
		elif "消失" in all_data["0423"][i]:
		    temp.append("消失")
		elif all_data["0423"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0423"]=temp

	temp=[]
	all_data["0911"]=all_data["0911"].astype("str")
	for i in range(len(all_data)):
		if "淋巴结肿大" in all_data["0911"][i]:
		    temp.append("淋巴结肿大")
		elif "淋巴结大" in all_data["0911"][i]:
		    temp.append("淋巴结肿大")
		elif all_data["0911"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0911"]=temp

	temp=[]
	all_data["0912"]=all_data["0912"].astype("str")
	for i in range(len(all_data)):
		if "不肿大" in all_data["0912"][i]:
		    temp.append("正常")
		elif "无肿大" in all_data["0912"][i]:
		    temp.append("正常")
		elif "结节" in all_data["0912"][i]:
		    temp.append("结节")
		elif "肿大" in all_data["0912"][i]:
		    temp.append("肿大")
		elif "欠光滑" in all_data["0912"][i]:
		    temp.append("欠光滑")
		elif all_data["0912"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0912"]=temp

	temp=[]
	all_data["0929"]=all_data["0929"].astype("str")
	for i in range(len(all_data)):
		if "结节" in all_data["0929"][i]:
		    if "增生" in all_data["0929"][i]:
		        temp.append("结节，增生")
		    else:
		        temp.append("结节")
		elif "增生" in all_data["0929"][i]:
		    temp.append("增生")
		elif all_data["0929"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0929"]=temp

	temp=[]
	all_data["0973"]=all_data["0973"].astype("str")
	for i in range(len(all_data)):
		if "已手术" in all_data["0973"][i]:
		    temp.append("已手术")
		elif "疝" in all_data["0973"][i]:
		    temp.append("疝")
		elif all_data["0973"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0973"]=temp

	temp=[]
	all_data["0974"]=all_data["0974"].astype("str")
	for i in range(len(all_data)):
		if "皮炎" in all_data["0974"][i]:
		    temp.append("皮炎")
		elif "癣" in all_data["0974"][i]:
		    temp.append("癣")
		elif "疹" in all_data["0974"][i]:
		    temp.append("疹")
		elif "银屑病" in all_data["0974"][i]:
		    temp.append("银屑病")
		elif "白癜风" in all_data["0974"][i]:
		    temp.append("白癜风")
		elif all_data["0974"][i]=="nan":
		    temp.append(None)
		else:
		    temp.append("正常")
	all_data["0974"]=temp


	columns=["100010","3190","3191","3192","3195","3196","3197","3207","3430","2228","2229","2230",
		    "2233","2231","360","3301","3189","3194","3485","3486","2282","30002"]
	for col in tqdm(columns):
		temp=[]
		all_data[col]=all_data[col].astype("str")
		for i in range(len(all_data)):
		    if "++++" in all_data[col][i]:
		        temp.append(4)
		    elif "+++" in all_data[col][i]:
		        temp.append(3)
		    elif "++" in all_data[col][i]:
		        temp.append(2)
		    elif "+-" in all_data[col][i]:
		        temp.append(0.5)
		    elif "+" in all_data[col][i]:
		        temp.append(1)
		    elif "阳性" in all_data[col][i]:
		        temp.append(1)
		    elif all_data[col][i]=="nan":
		        temp.append(np.nan)
		    else:
		        temp.append(0)
		all_data[col]=temp
		all_data[col]=all_data[col].fillna(all_data[col].mean())
		


	columns=["2302"]
	for col in columns:
		all_data[col]=all_data[col].replace({"正常":"健康"})
		temp=[]
		for i in range(len(all_data)):
		    
		    pattern = re.compile(r'[\u4e00-\u9fa5]+')
		    try:
		        temp.append(pattern.findall(all_data[col][i])[0])
		    except:
		        temp.append(np.nan)
		all_data[col]=temp
		all_data[col]=all_data[col].fillna("未查")
		all_data[col]=all_data[col].replace({"肥健康":"健康","正常疲劳反应":"健康"})




	def take_func(x,string,items):
		try:
		    result=x.split(string)[1:]
		    result = string.join(result)
		    for item in items:
		        result=result.split(item)[0]
		    return result
		except:
		    return(np.nan)


	# In[14]:


	feature_0101=all_data["0101"].astype(str)

	items=["甲状腺","瓣","子宫","乳腺","乳房","室腔"]
	all_data["0101颈总动脉"]=feature_0101.map(lambda x: take_func(x,"颈总动脉",items))

	items=["颈总动脉","瓣","子宫","乳腺","乳房","甲状腺"]
	all_data["0101心"]=feature_0101.map(lambda x: take_func(x,"室腔",items))

	items=["颈总动脉","瓣","子宫","乳腺","乳房","甲状腺"]
	all_data["0101瓣"]=feature_0101.map(lambda x: take_func(x,"瓣",items))

	from copy import copy
	feature_0102=all_data["0102"].astype(str)

	###所有的检查项目
	all_items=["甲状腺","心","肝","胆","脾","胰","颈总动脉","乳腺","前列腺","膀胱","子宫","卵巢","肾","胃","乳房","附件","颈动脉"]

	####我们需要的提取的feature
	features=["心","颈总动脉","颈动脉"]
	for feature in features:
		items=copy(all_items)
		items.remove(feature)
		all_data["0102"+feature]=feature_0102.map(lambda x: take_func(x,feature,items))

	all_data["心"]=all_data["0102心"]+all_data["0101心"]+all_data["0101瓣"]
	all_data["动脉"]=all_data["0101颈总动脉"]+all_data["0102颈总动脉"]+all_data["0102颈动脉"]

	all_data.drop(["0101心","0101颈总动脉","0102心","0102颈总动脉","0101瓣","0102颈动脉",],axis=1,inplace=True)


	# In[15]:


	temp=[]
	all_data["动脉"]=all_data["动脉"].astype(str)
	for i in range(len(all_data)):
		if "不增厚" in all_data["动脉"][i]:
		    temp.append("不增厚")
		elif "增厚" in all_data["动脉"][i]:
		    temp.append("增厚")
		elif "斑块" in all_data["动脉"][i]:
		    temp.append("斑块")
		elif "硬化" in all_data["动脉"][i]:
		    temp.append("硬化")
		else:
		    temp.append("正常")
	all_data["动脉"]=temp


	# In[16]:


	temp=[]
	all_data["心"]=all_data["心"].astype(str)
	for i in range(len(all_data)):
		if "功能减低" in all_data["心"][i]:
		    temp.append("功能减低")
		elif "E峰＜A峰" in all_data["心"][i]:
		    temp.append("e峰小于A峰")
		elif "E峰<A峰" in all_data["心"][i]:
		    temp.append("e峰小于A峰")
		else:
		    temp.append("正常")
	all_data["心室"]=temp

	temp=[]
	for i in range(len(all_data)):
		if "无反流" in all_data["心"][i]:
		    temp.append("正常")
		elif "未见明显反流" in all_data["心"][i]:
		    temp.append("正常")
		elif "反流" in all_data["心"][i]:
		    temp.append("反流")
		else:
		    temp.append("正常")
	all_data["动脉瓣"]=temp
	all_data.drop("心",axis=1,inplace=True)
	
	
	
	
		## 删除结构过于单一和无用features
	all_data.drop(["0101","0102","1102","0116","0119","0201","0202","0731","0732","300131","3731","3813","0409","0434","0439",
		           "0203","0206","0207","0208","0209","0222","0403","0413","0429","0715","0726","0728",
		           "0501","0503","0509","0516","0537","0539","0541","0703","0705","0706","0707","0709",
		           "0901","0947","0949","0954","0972","0975","0977","0978","0979","0980","0985","1001",
		           "1103","1301","1302","1303","1304","1305","1313","1314","1315","3399","A201","A202",
		           "0212","0430","0432","0433","0976","0422","0427","1329","2501","979027","0986","1104",
		           "A601","0225","0414","0415","0428","0440","0546","0981","0982","0983","0213",
		          "A301","A302","3101","0218","1335","3725","3738","1337","1002","0224","0441","0220",
		          "439032"],axis=1,inplace=True)
	import lightgbm as lgb
	reg_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=20,   
		                          learning_rate=0.05, n_estimators=720,
		                          max_bin = 55, bagging_fraction = 0.8,
		                          bagging_freq = 5, feature_fraction = 0.2319,
		                          feature_fraction_seed=9, bagging_seed=9,n_jobs=4,
		                          min_data_in_leaf =16, min_sum_hessian_in_leaf = 11)
		                          
	columns=["193","10002","0425","319","2372","314","100007","2174","1115","2333","317","10003",
		     "100006","183","100005","269011","2420","1345"] #feature importance top20
	
	print("filling nan value....")	     
	all_data_temp=pd.get_dummies(all_data)
	for col in tqdm(columns,desc="fillna"):
		####用非缺失值做训练集合，缺失值做预测集
		
		X_train_temp=all_data_temp[all_data_temp[col].notnull()].drop(col,axis=1)
		Y_train_temp=all_data_temp[col][all_data_temp[col].notnull()]
		assert np.sum(X_train_temp.index!=Y_train_temp.index)==0
		X_test_temp=all_data_temp[all_data_temp[col].isnull()].drop(col,axis=1)

		if all_data.corr()[col].sort_values(ascending=False)[1]>0.3 and len(X_train_temp)>4000:    
		    
		    reg_lgb.fit(X_train_temp,Y_train_temp)
		    y_pred=reg_lgb.predict(X_test_temp)               

		    X_train_temp[col]=Y_train_temp
		    X_test_temp[col]=y_pred                                           
		    df_temp=pd.concat([X_train_temp,X_test_temp])                     ###合并缺失值与非缺失值
		    
		    predicted_feature=pd.DataFrame(df_temp[col],index=df_temp.index)

		    all_data=pd.concat([all_data.drop(col,axis=1),predicted_feature],axis=1,join="inner")
		

	# ## box cox sknewed data

	# In[18]:


	numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
	# Check the skew of all numerical features
	skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
	
	skewness = pd.DataFrame({'Skew' :skewed_feats})

	skewness = skewness[abs(skewness.Skew) > 0.75]
	

	from scipy.special import boxcox1p
	skewed_features = skewness.index
	lam = 0.15
	for feat in skewed_features:
		all_data[feat] = boxcox1p(all_data[feat], lam)

	
	all_data.to_csv("../data/all_data_all_done.csv")
