from preprocessing import Preprocessing
from mannul_features import Extract_mannul_features
from FE_numeric import FE_numeric_data
from FE_text import FE_text_data
from predict1 import Predict_first
from predict2 import Predict_second
from predict3 import Predict_third
from predict4 import Predict_fourth
from predict5 import Predict_fifth
from concat import Concatenate

predict=False           ###是否进行预测，输出预测结果，预测结果输出在submit文件夹
model_name=input("please input the cv model:")        ###选择用于交叉验证的模型，不进行交叉验证设置None
start_nod=int(input("""where to start?
0 for preprocess
1 for mannul feature 
2 for FE numeric
3 for FE text
4 for cv_score
5 for predict
"""))

def main():

	print("#####################################################")
	print("Preprocessing...")
	if start_nod <=0:
		Preprocessing()  ###运行预处理函数                           ###预处理原始数据
	print("Preprocessing Done!!!")
	
	
	print("#####################################################")	
	print("Extract_mannul_features...")
	if start_nod <=1:
		Extract_mannul_features()                                    ###抽取手工特征
	print("Extraction Done!!!")
	
	
	print("#####################################################")
	print("Dealing with numeric features...")
	if start_nod <=2:
		FE_numeric_data()                                            ###数值型数据特征工程    
	print("Dealing with text features...")
	if start_nod <=3:
		FE_text_data()												  ###文本型数据特征工程
	print("Feature engineering Done!!!")
	print("#####################################################")
	print("Training && Predicting....")                           ####训练并预测数据
	if start_nod >=4:
				
		Predict_first(model_name=model_name,predict=predict)
		print("Systolic Prediction Done!")
	
		Predict_second(model_name=model_name,predict=predict)
		print("Diastolic Prediction Done!")
	
		Predict_third(model_name=model_name,predict=predict)
		print("Glycerin Prediction Done!")
	
		Predict_fourth(model_name=model_name,predict=predict)
		print("HDL Prediction Done!")
	
		Predict_fifth(model_name=model_name,predict=predict)
		print("LDL Prediction Done!")
	
		if predict:
			Concatenate()          ####将单个预测结果整理输出为最终结果
		print("Prediction Done!!!")
	

	
if __name__=="__main__":
	main()
	
	
