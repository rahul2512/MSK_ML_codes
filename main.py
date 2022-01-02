import pandas as pd
from pytorch import final_run, run_final_model
import numpy as np
import sys

hyper =  pd.read_csv('hyperparam.txt',delimiter='\s+')

def cluster_run(start_ind, end_ind):
    for hyper_arg in np.arange(start_ind, end_ind):
        hyper_val =  hyper.iloc[hyper_arg]
#        final_run(hyper_arg,hyper_val,pca=False,scale_out=False)
        run_final_model("JRF",hyper_arg,hyper_val,pca=False,scale_out=False)

#start_ind = int(sys.argv[1])
#end_ind = int(sys.argv[2])
start_ind,end_ind=3,3+1
print("start_ind,end_ind = ",start_ind,end_ind)
cluster_run(start_ind, end_ind)
