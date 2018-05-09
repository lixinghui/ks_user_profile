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


def main():

	print("#####################################################")
	print("Preprocessing...")
	#Preprocessing()  ###运行预处理函数
	print("Preprocessing Done!!!")
	
	
	print("#####################################################")	
	print("Extract_mannul_features...")
	#Extract_mannul_features()
	print("Extraction Done!!!")
	
	
	print("#####################################################")
	print("Dealing with numeric features...")
	#FE_numeric_data()
	print("Dealing with text features...")
	#FE_text_data()
	print("Feature engineering Done!!!")
	print("#####################################################")
	print("Training && Predicting....")
	Predict_first()
	print("Systolic Prediction Done!")
	
	Predict_second()
	print("Diastolic Prediction Done!")
	
	Predict_third()
	print("Glycerin Prediction Done!")
	
	Predict_fourth()
	print("HDL Prediction Done!")
	
	Predict_fifth()
	print("LDL Prediction Done!")
	
	#Concatenate()
	print("Prediction Done!!!")

	
	
if __name__=="__main__":
	main()
	
	
