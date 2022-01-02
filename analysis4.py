import numpy as np
import pandas as pd
import os.path
from pathlib import Path
from pytorch import create_final_model
hyper =  pd.read_csv('hyperparam.txt',delimiter='\s+')

def read_k_fold_data(f):
	data = {}
	for i in range(5):
		data[i] = pd.read_csv(f.replace('XXXX',str(i)),delimiter=' ',header=None)
	return data

def estimate_validation_results(f):
	data = read_k_fold_data(f)
	avg_train_mse, avg_val_mse, avg_test_mse, avg_train_pc, avg_val_pc, avg_test_pc = [], [], [], [], [], []
	for i in range(5):
		avg_train_mse.append(data[i].iloc[0,0])	
		avg_val_mse.append(data[i].iloc[0,1])
		avg_test_mse.append(data[i].iloc[0,2])
		avg_train_pc.append(data[i].iloc[1].mean())
		avg_val_pc.append(data[i].iloc[2].mean())
		avg_test_pc.append(data[i].iloc[3].mean())
	return avg_train_mse, avg_val_mse, avg_test_mse, avg_train_pc, avg_val_pc, avg_test_pc


def return_model(f,criteria):
	avg_train_mse, avg_val_mse, avg_test_mse, avg_train_pc, avg_val_pc, avg_test_pc = estimate_validation_results(f)
	if criteria == "avg_val_mse":
		if np.mean(avg_val_mse) == 0:
			return 1000
		else:
			return np.mean(avg_val_mse)


def return_model_stat(f):
	avg_train_mse, avg_val_mse, avg_test_mse, avg_train_pc, avg_val_pc, avg_test_pc = estimate_validation_results(f)
	print('avg_train_mse',np.mean(avg_train_mse))
	print('avg_val_mse',  np.mean(avg_val_mse))
	print('avg_test_mse', np.mean(avg_test_mse))
	print('avg_train_pc', np.mean(avg_train_pc))
	print('avg_val_pc',   np.mean(avg_val_pc))
	print('avg_test_pc',  np.mean(avg_test_pc))
	print('---------XXXXXXXXX---------XXXX--------XXXXXXXXXX--------')


def train_final_model(hyper_arg,which):
	hyper_val =  hyper.iloc[hyper_arg]
	create_final_model(hyper_arg,hyper_val,which,pca=False)
	return None




### total numer of hyperprmset space ---- 131220
def run_final(which):
	count=0
	start=1000
	for i in range(131220):
		if  Path('text_out/Angles_hyper_'+str(i)+'_k_fold_4.txt').is_file():
			f='text_out/'+which+'_hyper_'+str(i)+'_k_fold_XXXX.txt'
			tmp = return_model(f,"avg_val_mse")
			if tmp < start:
				print("model_index = ",i,start,which)
				start = tmp
				return_model_stat(f)
				count=count+1
				print("XXXXXXXX-----",count,"-------XXXXXXXX")
				if count>10:
					train_final_model(i,which)
			None
		else:
			None
	#		print("python main.py",i, i+1,"&")


#run_final("JRF")
#run_final('Muscle')
#run_final('JM')
run_final('Angles')

