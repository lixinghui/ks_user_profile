

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,norm
from tqdm import tqdm  
import re

def FE_numeric_data():

	Y=pd.read_csv('../data/Y_train.csv',low_memory=False)
	Y=Y.set_index("vid")
	Y_pred=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',low_memory=False,encoding="gbk")
	Y_pred=Y_pred.set_index("vid")
	
	m_test=Y_pred.shape[0]
	all_data=pd.read_csv("../data/all_data_pivot.csv",low_memory=False)

	assert np.sum(all_data["vid"]!=(list(Y.index)+list(Y_pred.index)))==0

	# 以下用group后用mean填充

	columns=["0424","10004","1117","1321","1322","190","191","192","2403","2404","2405","316","320",
		     "1814","1815","1840","1850","2372","31","32","33","34","38","39","37","312","313","315",
		     "2406","1127","155","269003","269004","269005","269006","269008","269009","269010",
		    "269012","269013","269014","269015","269016","269017","269018","269019","269020","269021",
		    "269022","269023","269024","269025","1845"]

	for col in tqdm(columns):
		all_data[col]=all_data[col].astype(str)
		temp=[]
		for i in range(len(all_data)):
		    pattern = re.compile(r'\d+\.{0,1}\d+')
		    try:
		        temp.append(pattern.findall(all_data[col][i])[0])
		    except:
		        temp.append(np.nan)
		all_data[col]=temp
		all_data[col]=all_data[col].astype("float32")
		all_data[col]=all_data.groupby(["gender","age"])[col].transform(lambda x: x.fillna(x.mean()))


	#以下feature直接用mean填充（group会放大异常值）

	columns1=["0424","10004","1117","1321","1322",'459161', '809035', '459156', '319100', '0111', '809016',
		     "2410","2165","2421","2413","10013","2168","1842","2412","300028","300048","300113","709001",
		     "100008","300044","0104","300067","300125","0107","300009","300014","809003",
		     '459155', '0105', '459158', '0106', '300006', '311', '3184', '35', '300129', '0109', '310', 
		     "1814","1815","183","1840","1850","31","32","33","34","38","39","37","312", '36',
		     "313","315","316","320","190","191","192","2403","2404","2405",'300069', '459159', '0108', '1124',
		     "2406","1127","155","269003","269004","269005","269006","269008","269009","269010", '459154',
		    "269012","269013","269014","269015","269016","269017","269018","269019","269020","269021",
		    "269022","269023","269024","269025","100012","100013","100014","10009","1106","1107","1112","1325",
		    "1326","139","143","1474","2386","2409","269007","300001","300008","300011","300012","300013",
		    "300021","300092","669001","669002","669004","669005","669006","669009","669021","809001",
		    "809004","809008","809009","809010","809013","809017","809021","809023","809025","809026","979001","979002",
		    "979003","979004","979005","979006","979007","979008","979009","004997","1110","1319","1320","1844","1873",
		    "189","20002","279006","300007","300068","300070","300074","300076","669003","669007","669008",
		    "809013","809018","809019","809022","809027","1125","1331","1845","979011","979012",
		    "2390","2407","2986","300035","30006","300078","321","809002","809007","809020","809024","809029",
		    "809031","809032","809033","809034","979010","979025","979026","979027","A701","A703",
		    ]+[str(i) for i in range(979013,979024,1)]+[str(i) for i in range(809037,809062,1)]
	for col in tqdm(set(columns1)-set(columns)):
		all_data[col]=all_data[col].astype(str)
		temp=[]
		for i in range(len(all_data)):
		    pattern = re.compile(r'\d+\.{0,1}\d+')
		    try:
		        temp.append(pattern.findall(all_data[col][i])[0])
		    except:
		        temp.append(np.nan)
		all_data[col]=temp
		all_data[col]=all_data[col].astype("float32")
		all_data[col]=all_data[col].fillna(all_data[col].mean())


	# 以下数据阴性用0填充，缺失值用mean填充


	columns=["300005","3429","3193","3730","2177","2376","300017","300018","300019","979024","269026",
		     "669024","2371","300036","1363"]
	for col in tqdm(columns):
		all_data[col]=all_data[col].astype(str)
		all_data[col].fillna("None",inplace=True)
		temp=[]
		for i,j in enumerate(all_data[col].values):
		    
		    if "+" in j or "阳性" in j:
		        if col=="3730" or col=="300019" or col=="2371":
		            temp.append(5)
		        elif col=="669004":
		            temp.append(1)
		        else:
		            temp.append(40)
		    elif "-" in j or "阴性" in j:
		        temp.append(0)
		    else:
		        pattern = re.compile(r'\d+\.{0,1}\d+')
		        try:
		            temp.append(pattern.findall(j)[0])
		        except:
		            temp.append(np.nan)
		all_data[col]=temp
		all_data[col]=all_data[col].astype("float32")
		all_data[col]=all_data[col].fillna(all_data[col].mean())


	# 针对feature_importance高但缺失值非常多的数据，我们着重处理（用相关性最高的数据group后填充缺失值）   


	columns=["193","10002","0425","319","2372","314","100007","2174","1115","2333","317","10003",
		     "100006","183","100005","269011","2420","1345"]
	for col in tqdm(columns):
		all_data[col]=all_data[col].astype(str)
		temp=[]
		for i in range(len(all_data)):
		    pattern = re.compile(r'\d+\.{0,1}\d+')
		    try:
		        temp.append(pattern.findall(all_data[col][i])[0])
		    except:
		        temp.append(np.nan)
		all_data[col]=temp
		all_data[col]=all_data[col].astype("float32")


	all_data["193"]=all_data.groupby(all_data["192"]>all_data["192"].mean()
		                            )["193"].transform(lambda x: x.fillna(x.mean()))

	all_data["10002"]=all_data.groupby(all_data["192"]>all_data["192"].mean()
		                            )["10002"].transform(lambda x: x.fillna(x.mean()))

	all_data["0425"]=all_data.groupby(all_data["0424"]>all_data["0424"].mean()
		                            )["0425"].transform(lambda x: x.fillna(x.mean()))

	all_data["319"]=all_data.groupby(all_data["312"]>all_data["312"].mean()
		                            )["319"].transform(lambda x: x.fillna(x.mean()))


	all_data["2372"]=all_data["2372"].fillna(all_data["2372"].mean())  ###没有相关性较好的数据，直接用平均值填充

	all_data["314"]=all_data.groupby(all_data["37"]>all_data["37"].mean()
		                            )["314"].transform(lambda x: x.fillna(x.mean()))
	all_data["100007"]=all_data["100007"].fillna(all_data["100007"].mean())

	all_data["2174"]=all_data.groupby(all_data["1845"]>all_data["1845"].mean()
		                            )["2174"].transform(lambda x: x.fillna(x.mean()))

	all_data["1115"]=all_data.groupby(all_data["1117"]>all_data["1117"].median()
		                            )["1115"].transform(lambda x: x.fillna(x.mean()))

	all_data["2333"]=all_data["2333"].fillna(all_data["2333"].mean())  ###没有相关性较好的数据，直接用平均值填充

	all_data["317"]=all_data.groupby(all_data["316"]>all_data["316"].mean()
		                            )["317"].transform(lambda x: x.fillna(x.mean()))

	all_data["10003"]=all_data.groupby(all_data["183"]>all_data["183"].mean()
		                            )["10003"].transform(lambda x: x.fillna(x.mean()))

	all_data["100006"]=all_data["100006"].fillna(all_data["100006"].mean())  ###没有相关性较好的数据，直接用平均值填充

	all_data["100005"]=all_data.groupby(all_data["320"]>all_data["320"].mean()
		                            )["100005"].transform(lambda x: x.fillna(x.mean()))

	all_data["183"]=all_data.groupby(all_data["10003"]>all_data["10003"].mean()
		                            )["183"].transform(lambda x: x.fillna(x.mean()))

	all_data["269011"]=all_data.groupby(all_data["38"]>all_data["38"].mean()
		                            )["269011"].transform(lambda x: x.fillna(x.mean()))

	all_data["2420"]=all_data.groupby(all_data["0424"]>all_data["0424"].mean()
		                            )["2420"].transform(lambda x: x.fillna(x.mean()))

	all_data["1345"]=all_data.groupby(all_data["100012"]>all_data["100012"].mean()
		                            )["1345"].transform(lambda x: x.fillna(x.mean()))



	all_data=all_data.set_index("vid")
	X_train=all_data[:-m_test]
	X_test=all_data[-m_test:]


	df=pd.concat([X_train,Y],axis=1)



	# ##### 根据scatter图表剔除异常值，其实也可以用方差，均值方法，剔除异常值

	df=df[df["0104"]<100]
	df=df[df["100008"]<10]
	df=df[df["709001"]<1]
	df=df[df["2412"]<100]
	df=df[df["1842"]<100]
	df=df[df["10013"]<400]
	df=df[df["1363"]<250]
	df=df[df["2410"]<60]
	df=df[df["A701"]<100]
	df=df[df["809031"]<100]
	df=df[df["300035"]<80]
	df=df[df["2986"]<100]
	df=df[df["2407"]<4000]
	df=df[df["1331"]<4]
	df=df[df["669008"]<100]
	df=df[df["300076"]<100]
	df=df[df["300074"]<20]
	df=df[df["300070"]<15]
	df=df[df["300068"]<50]
	df=df[df["1873"]<40]
	df=df[df["1844"]<4]
	df=df[df["1110"]<50]
	df=df[df["979014"]<2.0]
	df=df[df["979002"]<1.5]
	df=df[df["809025"]<15]
	df=df[df["809009"]<15]
	df=df[df["669021"]<100]
	df=df[df["669006"]<60]
	df=df[df["669004"]<40]
	df=df[df["669002"]<15]
	df=df[df["669001"]<40]
	df=df[df["300017"]<20]
	df=df[df["300012"]<20]
	df=df[df["300008"]<4]
	df=df[df["300001"]<60]
	df=df[df["2376"]<800]
	df=df[df["2177"]<200]
	df=df[df["1474"]<300]
	df=df[df["139"]<6]
	df=df[df["1112"]<20]
	df=df[df["10009"]<60]
	df=df[df["100014"]<60]
	df=df[df["100013"]<12]
	df=df[df["269023"]<1.5]
	df=df[df["269022"]<4]
	df=df[df["269014"]<15]
	df=df[df["269005"]<3]
	df=df[df["155"]<40]
	df=df[df["1345"]<400]
	df=df[df["1127"]<3000]
	df=df[df["34"]<2]
	df=df[df["33"]<10]
	df=df[df["317"]>220]
	df=df[df["312"]<20]
	df=df[df["2372"]<20]
	df=df[df["2333"]<10]
	df=df[df["10003"]<60]
	df=df[df["10004"]<500]
	df=df[df["1115"]<300]
	df=df[df["10002"]<40]
	df=df[df["1117"]<400]
	df=df[df["1814"]<400]
	df=df[df["1815"]<200]
	df=df[(df["183"]>50)&(df["183"]<120)]
	df=df[df["1850"]<40]
	df=df[df["190"]<250]
	df=df[df["192"]<100]
	df=df[df["193"]<30]
	df=df[df["2174"]>30]
	df=df[(df["2403"]>20)&(df["2403"]<10000)]
	df=df[df["2405"]>10]
	df=df[df["300005"]<100]
	df=df[df["3429"]<100]
	df=df[df["3730"]<10]


	X_train=df.iloc[:,:-5]
	Y_train=df.iloc[:,-5:]

	all_data=pd.concat([X_train,X_test],axis=0)
	num_train=X_train.shape[0]
	num_test=X_test.shape[0]
	
	print(num_train,num_test,Y_train.shape)
	Y_train.to_csv("../data/Y_train_outlier_done.csv")

	all_data.to_csv("../data/all_data_outlier_done.csv")


