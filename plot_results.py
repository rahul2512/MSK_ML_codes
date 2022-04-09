import numpy as np
import pandas as pd, keras
import os.path
from pathlib import Path
from pytorch import run_final_model, run_cross_valid, plot_saved_model
from pytorch_utilities import hyper_param

hyper =  pd.read_csv('hyperparam.txt',delimiter='\s+')
#hyper =  pd.read_csv('hyperparam_linear.txt',delimiter='\s+')

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


def return_model(f,criteria):     #### obkective of this function is to obtain average cross validation error
	avg_train_mse, avg_val_mse, avg_test_mse, avg_train_pc, avg_val_pc, avg_test_pc = estimate_validation_results(f)
	if criteria == "avg_val_mse":
		if np.mean(avg_val_mse) == 0:    #### this if condition is only because as sometime NANs values or some error create problem
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


def train_final_model(subject_condition,hyper_arg,which,scale_out,pca,model_class):     ########## this function is useful to run a final individual run
	hyper_val =  hyper.iloc[hyper_arg]
	model = run_final_model(subject_condition,which,hyper_arg,hyper_val,pca,scale_out,model_class)
	return model


### total numer of hyperprmset space ---- 131220
def run_final(subject_condition,which,scale_out,model_class):    #### this model find some best model as per CV error and then run train_final_model
        count=0
        start=1000  ### initiate with large mse value
        for i in range(43741):
                if  Path('text_out/'+which+'_hyper_'+str(i)+'_k_fold_'+subject_condition+'_4.txt').is_file():
                        f='text_out/'+which+'_hyper_'+str(i)+'_k_fold_'+subject_condition+'_XXXX.txt'
                        tmp = return_model(f,"avg_val_mse")
                        if tmp < start:
                                print("model_index = ",i,start,which)
                                if count < 6:       ######### after arriviing a certain accuracy save all models
                                        start = tmp
                                return_model_stat(f)
                                count=count+1
                                print("XXXXXXXX-----",count,"-------XXXXXXXX")
                                if count>4:
                                        train_final_model(subject_condition,i,which,scale_out,model_class)
                        None
                else:
                        None




##############################################################################
##############################################################################
##############################################################################
#########   RUN CODES BELOW
##############################################################################
##############################################################################
##############################################################################



#############
######  STEP-ONE

# hyper_param()   ###### create hyperparameters file ... only need to do it once




#############
######  STEP-Two
# order of belowe func (which,hyper_arg,hyper.iloc[hyper_arg],k_fold,pca,scale_out)
## below is done by generating parallel script not the way described below

# size_of_hyper_space = 100
# for index in range(size_of_hyper_space):
#     run_cross_valid('subject_exposed','JRF',index,hyper.iloc[index],5,False,True)
index=4719
# run_cross_valid('subject_naive','JRF',index,hyper.iloc[index],4,False,True)   ####### for subject-naive case only 4 fold CV possible

 # train_final_model('subject_naive',index,'Angles',True,pca=False)
#train_final_model('subject_naive',index,'JRF',True,pca=False)
#train_final_model('subject_naive',index,'JM',True,pca=False)
# train_final_model('subject_naive',index,'Muscle',True,pca=False)
# train_final_model('subject_exposed',4719,'MuscleAct',True,pca=False)



def stat_basic_models(model_class):

    for cat in ['MuscleAct','Angles','JM','JRF','Muscle']:
        for sub in ['subject_naive','subject_exposed']:
            for ind in range(1):
                train_final_model(sub,ind,cat,True,False,model_class)
                input()
#stat_basic_models('Linear')



###########################  
# plot final saved model 
###########################  
#plot_saved_model(which, hyper_arg1, hyper_val1, hyper_arg2, hyper_val2, pca,scale_out,model_class):

def plot_figures_in_article():
    plot_saved_model('JM',5237,hyper.iloc[5237],3072,hyper.iloc[3072],False,True,'NN')
    plot_saved_model('JRF', 11742,hyper.iloc[11742],1626,hyper.iloc[1626],False,True,'NN')
    plot_saved_model('Muscle',4719,hyper.iloc[4719],8477,hyper.iloc[8477],False,True,'NN')
    plot_saved_model('Angles',2706,hyper.iloc[2706],545,hyper.iloc[545],False,True,'NN')
    plot_saved_model('MuscleAct',4397,hyper.iloc[4397],9438,hyper.iloc[9438],False,True,'NN')

plot_figures_in_article()


