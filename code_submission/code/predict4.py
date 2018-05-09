import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew,norm
from tqdm import tqdm  
import re

def Predict_fourth():

	all_data=pd.read_csv("../data/all_data_all_done.csv",low_memory=False)
	all_data=all_data.set_index("vid")
	Y_pred=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',low_memory=False,encoding="gbk")
	Y_pred=Y_pred.set_index("vid")
	m_test=Y_pred.shape[0]


	all_data.drop(["num_items","0702","0929"],axis=1,inplace=True)  

	# ### getdummy之后生成最终数据


	all_data=pd.get_dummies(all_data,drop_first=True)

	X_train=all_data.iloc[:-m_test,:]
	X_test=all_data.iloc[-m_test:,:]


	# ### 由于目标函数是log1p的平方差，所以我们对y进行log1p转换




	Y_train=pd.read_csv("../data/Y_train_outlier_done.csv")
	Y_train.set_index("vid",inplace=True)
	Y_train=np.log(Y_train+1)



	assert np.sum(X_train.index!=Y_train.index)==0
	assert np.sum(X_test.index!=Y_pred.index)==0


	# ### 建立模型 交叉验证性能

	# In[19]:


	from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC,LinearRegression
	from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
	from sklearn.neural_network import MLPRegressor
	from sklearn.kernel_ridge import KernelRidge
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import RobustScaler
	from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
	from sklearn.model_selection import KFold, cross_val_score, train_test_split
	from sklearn.metrics import mean_squared_error as mse
	import xgboost as xgb
	import lightgbm as lgb


	# In[20]:


	n_folds=5
	def rmse_cv(model,i):
		mse= -cross_val_score(model, X_train.values, Y_train.values[:,i], 
		                               scoring="neg_mean_squared_error", cv = n_folds)
		return(mse)




	reg_lasso=make_pipeline(RobustScaler(), Lasso(alpha =0.00015, random_state=1,max_iter=10000))




	reg_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0015, l1_ratio=.1, max_iter=10000,random_state=3))



	reg_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=20,   
		                          learning_rate=0.05, n_estimators=720,
		                          max_bin = 55, bagging_fraction = 0.8,
		                          bagging_freq = 5, feature_fraction = 0.2319,
		                          feature_fraction_seed=9, bagging_seed=9,n_jobs=4,
		                          min_data_in_leaf =16, min_sum_hessian_in_leaf = 11)



	reg_GDBT = GradientBoostingRegressor(n_estimators=1174, learning_rate=0.015,
		                               max_depth=9, max_features='sqrt',
		                               min_samples_leaf=46, min_samples_split=8,
		                               loss='huber', random_state =10) 


	reg_xgb = xgb.XGBRegressor(colsample_bytree=0.7184, 
		                       gamma=0.1253,n_estimators=740,n_jobs=4,
		                         learning_rate=0.02, max_depth=8,
		                         min_child_weight=16.154, reg_alpha=0.2695,
		                         subsample=0.8171, silent=1,reg_lambda=0.1855,
		                         )
	 

	for i in range(3,4):
		scores=rmse_cv(reg_xgb,i)
		print("xgb scores {:.5f}(with std: {:.5f})".format(scores.mean(),scores.std()))



	reg_et=ExtraTreesRegressor(n_estimators=354,max_features=0.3,          
		                       max_depth=68,n_jobs=-1,min_samples_split=2,
		                         min_samples_leaf=6,random_state=42)



	class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
		def __init__(self, base_models, meta_model, n_folds=5):
		    self.base_models = base_models
		    self.meta_model = meta_model
		    self.n_folds = n_folds
		# We again fit the data on clones of the original models
		def fit(self, X, y):
		    self.base_models_ = [list() for x in self.base_models]
		    self.meta_model_ = clone(self.meta_model)
		    kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
		    # Train cloned base models then create out-of-fold predictions
		    # that are needed to train the cloned meta-model
		    out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
		    
		    for i, model in enumerate(self.base_models):
		        for train_index, holdout_index in kfold.split(X, y):
		            instance = clone(model)
		            
		            self.base_models_[i].append(instance)

		            instance.fit(X[train_index], y[train_index])
		            y_pred = instance.predict(X[holdout_index])
		            
		            out_of_fold_predictions[holdout_index, i] = y_pred
		            
		    # Now train the cloned  meta-model using the out-of-fold predictions as new feature
		    self.meta_model_.fit(out_of_fold_predictions, y)
		    return self
	   
		#Do the predictions of all base models on the test data and use the averaged predictions as 
		#meta-features for the final prediction which is done by the meta-model
		def predict(self, X):
		    meta_features = np.column_stack([
		        np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_ ])
		    return self.meta_model_.predict(meta_features)


	# In[32]:


	reg_lasso_stack=make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1,max_iter=10000))

	stacked_averaged_models = StackingAveragedModels(base_models = (reg_lgb,reg_GDBT,reg_lasso,reg_ENet,reg_et,reg_xgb),
		                                             meta_model = reg_lasso_stack )





#	stacked_averaged_models.fit(X_train.values,Y_train.values[:,3])
#	y4_stacked=stacked_averaged_models.predict(X_test.values)


#	df_sub=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',
#		                   engine='python',encoding="gbk")
#	df_sub["血清高密度脂蛋白"]=np.exp(y4_stacked)-1
#
#	df_sub.to_csv('../data/4_stacked_xgb.csv',index=False)

