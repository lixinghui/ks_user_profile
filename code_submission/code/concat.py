import pandas as pd
import numpy as np

def Concatenate():
 
	df1=pd.read_csv("../data/1_stacked_xgb.csv").set_index("vid")
	df2=pd.read_csv("../data/2_stacked_xgb.csv").set_index("vid")
	df3=pd.read_csv("../data/3_stacked_xgb.csv").set_index("vid")
	df4=pd.read_csv("../data/4_stacked_xgb.csv").set_index("vid")
	df5=pd.read_csv("../data/5_stacked_xgb.csv").set_index("vid")

	df=pd.concat([df1["收缩压"],df2["舒张压"],df3["血清甘油三酯"],df4["血清高密度脂蛋白"],df5["血清低密度脂蛋白"]],axis=1)
	df.to_csv("../submit/submit_update.csv",header=False)

