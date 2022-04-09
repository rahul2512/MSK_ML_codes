from pytorch_utilities import *
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
import torch as tr
import numpy as np
import statsmodels.api as sm
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from fractions import Fraction
import sys, copy
import scipy
from scipy import signal
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec



######## Note only two randomly selected trials are chosen for test data
#STRS = {"JRF,12", "Muscle,4", "JM,10", "Angles,10"};

### Below code is not using in the final version
def reduce_dim_pca(X_train,X_test):
    pca = PCA(n_components=200)
    train_shape = X_train.shape[0]
    test_shape = X_test.shape[0]
    X = pd.concat([X_train,X_test])
    X = StandardScaler().fit_transform(X)
    principalComponents = pca.fit_transform(X)
#    print(np.shape(principalComponents))
#    print(pca.explained_variance_ratio_)
#    print("Variance explained by ", np.sum(pca.explained_variance_ratio_))
    X_train = principalComponents[0:train_shape]
    X_test  =  principalComponents[train_shape:train_shape+test_shape]
    X_train_pca,X_test_pca = pd.DataFrame(X_train), pd.DataFrame(X_test) 
    return X_train_pca, X_test_pca


def scale_output(Y_train,Y_test):  #scale the output features
    tmp1 = copy.deepcopy(Y_train)
    std = tmp1.std()
    ttt = Y_train.append(Y_test)
    Y_train,Y_test = Y_train/std, Y_test/std
    return Y_train,Y_test, std

def smooth_input(X_train,X_test):  ### not using this in the final version
    #filtered using a fourth- order, zero-lag, low-pass Butterworth filter with a cut-off frequency of 6 Hz
    # idea is to smoothen the input for each of the trial

    tmp1 = (X_train[484] == 0)  ##### index of 0 i.e. idea is to identify the index of new trial
    tmp1 = np.where(tmp1)[0]   ##### we have 13 trails in training data 
    tmp2 = (X_train[484] == 1)  ##### index of 0 i.e. idea is to identify the index of new trial
    tmp2 = np.where(tmp2)[0]   ##### we have 13 trails in training data 
    sos = signal.butter(4,0.2,'lowpass',output='sos')
    huh = copy.deepcopy(X_train)
    huh2 = copy.deepcopy(X_train)
    print(tmp1)
    print(tmp2)
    for l in range(len(tmp1)):
         huh.iloc[tmp1[l]:tmp2[l]+1] = signal.sosfilt(sos, X_train.iloc[tmp1[l]:tmp2[l]+1])
        # X_test  = signal.sosfilt(sos, X_test)
    huh2.iloc[::] = signal.sosfilt(sos, X_train.iloc[::])
    for k in range(2,484):
        plt.plot(np.arange(5007),X_train[k],color='k',ls='--')
        plt.plot(np.arange(5007),huh[k],color='g',ls=':')
        plt.plot(np.arange(5007)+62,huh2[k],color='r',ls='--')
        plt.show()
        plt.close()
        input()
    input()
    return X_train,X_test


def read_total_data(subject_condition,which,pca,scale_out):
    if which=="JRF":
        out_cols = np.arange(0,12)
    if which=="Muscle":
        out_cols = np.arange(12,33)
    if which=="JM":
        out_cols = np.arange(33,43)
    if which=="Angles":
        out_cols = np.arange(43,53)
    if which=="MuscleAct":
        out_cols = np.arange(53,74)

    path = "./data/"

    if subject_condition == 'subject_exposed':
        X_train = pd.read_csv(path+"Full_inputsTRAIN.csv",header=None,usecols = np.arange(2,485))
        Y_train = pd.read_csv(path+"Full_outputsTRAIN.csv",header=None,usecols = out_cols)
    
        X_test  = pd.read_csv(path+"Full_inputsTEST.csv",header=None,usecols = np.arange(2,485))
        Y_test  = pd.read_csv(path+"Full_outputsTEST.csv",header=None,usecols = out_cols)

    elif subject_condition == 'subject_naive':
        X_train = pd.read_csv(path+"Full_inputsTRAIN_excludesubject=1.csv",header=None,usecols = np.arange(2,485))
        Y_train = pd.read_csv(path+"Full_outputsTRAIN_excludesubject=1.csv",header=None,usecols = out_cols)
    
        X_test  = pd.read_csv(path+"Full_inputsTEST_excludesubject=1.csv",header=None,usecols = np.arange(2,485))
        Y_test  = pd.read_csv(path+"Full_outputsTEST_excludesubject=1.csv",header=None,usecols = out_cols)

    else:
        print("wrong subject category used -----allowed categories are subject_exposed and subject_naive")
        sys.exit()
    if which in ["Muscle" ,"MuscleAct"]:
#######  ListModify[list_] := {list[[1 ;; 5]] // Max, list[[6 ;; 7]] // Max,    list[[14 ;; 19]] // Max, list[[20 ;; 21]] // Max};
        print(which)
        tmp = copy.deepcopy(Y_train.iloc[:,[0,1,2,3]])
        tmp.iloc[:,0] = Y_train.iloc[:,[0,1,2,3,4]].max(axis=1)
        tmp.iloc[:,1] = Y_train.iloc[:,[5,6]].max(axis=1)
        tmp.iloc[:,2] = Y_train.iloc[:,[13,14,15,16,17,18]].max(axis=1)
        tmp.iloc[:,3] = Y_train.iloc[:,[19,20]].max(axis=1)
        Y_train = tmp

        tmp = copy.deepcopy(Y_test.iloc[:,[0,1,2,3]])
        tmp.iloc[:,0] = Y_test.iloc[:,[0,1,2,3,4]].max(axis=1)
        tmp.iloc[:,1] = Y_test.iloc[:,[5,6]].max(axis=1)
        tmp.iloc[:,2] = Y_test.iloc[:,[13,14,15,16,17,18]].max(axis=1)
        tmp.iloc[:,3] = Y_test.iloc[:,[19,20]].max(axis=1)
        Y_test = tmp
    # X_train,X_test = smooth_input(X_train,X_test)  ### note that need to smoothen it for each trial
    if scale_out ==True:
        Y_train,Y_test,sca = scale_output(Y_train,Y_test)

    if pca == True:
        X_train_pca,X_test_pca = reduce_dim_pca(X_train,X_test)
        return X_train, Y_train, X_test, Y_test, X_train_pca,X_test_pca
    else:
        return X_train, Y_train, X_test, Y_test,sca


def split_validation_data(subject_condition,which,pca,scale_out,k_fold_index):
    if pca ==True:
        X_train, Y_train, X_test, Y_test,X_train_pca,X_test_pca = read_total_data(subject_condition,which,pca,scale_out)
    else: 
        X_train, Y_train, X_test, Y_test = read_total_data(subject_condition,which,pca,scale_out)

    if subject_condition == 'subject_exposed':
    
        tmp = (X_train[484] == 0)  ##### index of 0 i.e. idea is to identify the index of new trial
        tmp = np.where(tmp)[0]   ##### we have 13 trails in training data 
        tmp = np.append(tmp,Y_train.shape[0])
        val_trials = np.random.choice(np.arange(13),2,replace=False)  ### out of 13 trails choose two trails numbers at random
        val_index1 = [tmp[val_trials[0]],tmp[val_trials[0]+1]]
        val_index2 = [tmp[val_trials[1]],tmp[val_trials[1]+1]]
        val_entries = np.concatenate([np.arange(val_index1[0],val_index1[1]),np.arange(val_index2[0],val_index2[1])])

    elif subject_condition == 'subject_naive':

        tmp = (X_train[484] == 0)  ##### index of 0 i.e. idea is to identify the index of new trial
        tmp = np.where(tmp)[0]   ##### we have 12 trails in training data coming from 4 subjects
        tmp = np.append(tmp,Y_train.shape[0])
        trial_list = [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
        val_trials = trial_list[k_fold_index]                     ### Choosing one subject at a time
        print(val_trials)
        val_index1 = [tmp[val_trials[0]],tmp[val_trials[0]+1]]
        val_index2 = [tmp[val_trials[1]],tmp[val_trials[1]+1]]
        val_index3 = [tmp[val_trials[2]],tmp[val_trials[2]+1]]
        val_entries = np.concatenate([np.arange(val_index1[0],val_index1[1]),np.arange(val_index2[0],val_index2[1]),np.arange(val_index3[0],val_index3[1])])

    if pca == True:
        X_val, Y_val     = copy.deepcopy(X_train_pca.iloc[val_entries]), copy.deepcopy(Y_train.iloc[val_entries])
        X_Train, Y_Train = copy.deepcopy(X_train_pca.drop(val_entries)), copy.deepcopy(Y_train.drop(val_entries))
        return X_Train, Y_Train, X_val, Y_val, X_test_pca, Y_test
    else:
        X_val, Y_val     = copy.deepcopy(X_train.iloc[val_entries]), copy.deepcopy(Y_train.iloc[val_entries])
        X_Train, Y_Train = copy.deepcopy(X_train.drop(val_entries)), copy.deepcopy(Y_train.drop(val_entries))
    
        return X_Train, Y_Train, X_val, Y_val, X_test, Y_test

    
def combined_plot(model1,model2,X_Test1,Y_Test1,X_Test2,Y_Test2,label,scale_out,model_class,sc1,sc2):
    ## Need to think about computing each trial separately and how that affects the output
    NRMSE_list,  PC_list  = [],[]
    NRMSE2_list, PC2_list = [],[]
    YP1, YP2 = model1.predict(X_Test1), model2.predict(X_Test2)
    YT1, YT2 = np.array(Y_Test1),np.array(Y_Test2) 
    a,b = np.shape(YT1)
    a2,b2 = np.shape(YT2)
    sc1,sc2 = sc1.to_numpy(),sc2.to_numpy()
    if 'MuscleAct' in label:
        sc1,sc2 = sc1*100,sc2*100
    YP1, YT1 = YP1*sc1, YT1*sc1
    YP2, YT2 = YP2*sc2, YT2*sc2
    #### the below loop is to set the time in terms of percentage of task
    count,aa = -1,[]
    df = X_Test1[484]
    zero_entries = np.where(df==0)
    zero_entries = np.concatenate([zero_entries[0],np.array([a])])   #### adding last element
    df2 = X_Test2[484]
    zero_entries2 = np.where(df2==0)
    zero_entries2 = np.concatenate([zero_entries2[0],np.array([a2])])   #### adding last element

    for u in df:
        if u == 0:
            count = count + 1
        aa.append(u+count)
    count=0
    if 'JRF' in label:
        fig = plt.figure(figsize=(8,10.5))
        gs1 = gridspec.GridSpec(700, 560)
        gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.06)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
        ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
        ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
        ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
        ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
        ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
        ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
        ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
        ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
        ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])
        ax50 = plt.subplot(gs1[600+d2:700  ,   0+d1:100 ])
        ax51 = plt.subplot(gs1[600+d2:700  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
        ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
        ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
        ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
        ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
        ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
        ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
        ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
        ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])
        ax52 = plt.subplot(gs1[600+d2:700  , 310+d1:410 ])
        ax53 = plt.subplot(gs1[600+d2:700  , 460+d1:560 ])

        ax_list  = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41, ax50, ax51]
        ax_list2 = [ax02, ax03, ax12, ax13, ax22, ax23, ax32, ax33, ax42, ax43, ax52, ax53]

        ss,b_xlabel = 8,9
        ylabel = [ 'Trunk \n Mediolateral', 'Trunk \n Proximodistal', 'Trunk \n Anteroposterior', 'Shoulder \n Mediolateral',
                  'Shoulder \n Proximodistal', 'Shoulder \n Anteroposterior', 'Elbow \n Mediolateral', 'Elbow \n Proximodistal',
                  'Elbow \n Anteroposterior', 'Wrist \n Mediolateral', 'Wrist \n Proximodistal', 'Wrist \n Anteroposterior']
        plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']

    elif 'Muscle' in label:
        fig = plt.figure(figsize=(8,4))
        gs1 = gridspec.GridSpec(215, 560)
        gs1.update(left=0.07, right=0.98,top=0.84, bottom=0.15)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[0:100 -d2    , 0+d1:100  ])
        ax01 = plt.subplot(gs1[0:100 -d2   , 150+d1:250 ])
        ax10 = plt.subplot(gs1[115+d2:215  , 0+d1:100 ])
        ax11 = plt.subplot(gs1[115+d2:215  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[0:100 -d2   , 310+d1:410 ])
        ax03 = plt.subplot(gs1[0:100 -d2   , 460+d1:560 ])
        ax12 = plt.subplot(gs1[115+d2:215  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[115+d2:215  , 460+d1:560 ])

        ax_list = [ax00,ax01,ax10 ,ax11]
        ax_list2= [ ax02,ax03, ax12 ,ax13 ]
        ss,b_xlabel = 8,1
        ylabel = ['Pectoralis major \n (Clavicle)','Biceps Brachii','Deltoid (Medial)','Brachioradialis']
        plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']


    elif 'JM' in label:
        fig = plt.figure(figsize=(8,8.25))
        gs1 = gridspec.GridSpec(580, 560)
        gs1.update(left=0.075, right=0.98,top=0.945, bottom=0.08)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
        ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
        ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
        ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
        ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
        ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
        ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
        ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
        ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
        ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
        ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
        ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
        ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
        ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
        ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
        ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
        ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
        ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])

        ax_list  = [ax00, ax10, ax01, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
        ax_list2 = [ax02, ax12, ax03, ax13, ax22, ax23, ax32, ax33, ax42, ax43]

        ss,b_xlabel = 8,7


        ylabel = [ 'Trunk Flexion / \n Extension', 'Trunk Internal / \n External Rotation', 'Trunk Right / \n Left Bending',
                  'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation',
                  'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']
        plot_list = ['(a)','(c)','(b)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)']

    elif 'Angles' in label:
        new_order = [7,8,9,0,1,2,3,4,5,6] 
        YP1 = YP1[:,new_order]
        YP2 = YP2[:,new_order]
        YT1 = YT1[:,new_order]
        YT2 = YT2[:,new_order]
        fig = plt.figure(figsize=(8,8.25))
        gs1 = gridspec.GridSpec(580, 560)
        gs1.update(left=0.065, right=0.98,top=0.945, bottom=0.07)
        d1, d2 =10, 10
        ax00 = plt.subplot(gs1[  0+d2:100  ,   0+d1:100 ])
        ax01 = plt.subplot(gs1[  0+d2:100  , 150+d1:250 ])
        ax10 = plt.subplot(gs1[120+d2:220  ,   0+d1:100 ])
        ax11 = plt.subplot(gs1[120+d2:220  , 150+d1:250 ])
        ax20 = plt.subplot(gs1[240+d2:340  ,   0+d1:100 ])
        ax21 = plt.subplot(gs1[240+d2:340  , 150+d1:250 ])
        ax30 = plt.subplot(gs1[360+d2:460  ,   0+d1:100 ])
        ax31 = plt.subplot(gs1[360+d2:460  , 150+d1:250 ])
        ax40 = plt.subplot(gs1[480+d2:580  ,   0+d1:100 ])
        ax41 = plt.subplot(gs1[480+d2:580  , 150+d1:250 ])

        ax02 = plt.subplot(gs1[  0+d2:100  , 310+d1:410 ])
        ax03 = plt.subplot(gs1[  0+d2:100  , 460+d1:560 ])
        ax12 = plt.subplot(gs1[120+d2:220  , 310+d1:410 ])
        ax13 = plt.subplot(gs1[120+d2:220  , 460+d1:560 ])
        ax22 = plt.subplot(gs1[240+d2:340  , 310+d1:410 ])
        ax23 = plt.subplot(gs1[240+d2:340  , 460+d1:560 ])
        ax32 = plt.subplot(gs1[360+d2:460  , 310+d1:410 ])
        ax33 = plt.subplot(gs1[360+d2:460  , 460+d1:560 ])
        ax42 = plt.subplot(gs1[480+d2:580  , 310+d1:410 ])
        ax43 = plt.subplot(gs1[480+d2:580  , 460+d1:560 ])

        ax_list  = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41]
        ax_list2 = [ax02, ax03, ax12, ax13, ax22, ax23, ax32, ax33, ax42, ax43]

        ss,b_xlabel = 8,7

        plot_list = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']

        ylabel = ['Trunk Forward / \n Backward Bending', 'Trunk Right / \n Left Bending', 'Trunk Internal / \n External Rotation', 
                  'Shoulder Flexion / \n Extension', 'Shoulder Abduction / \n Adduction', 'Shoulder Internal / \n External Rotation', 
                  'Elbow Flexion / \n Extension', 'Elbow Pronation / \n Supination', 'Wrist Flexion / \n Extension', 'Wrist Radial / \n Ulnar Deviation']

     ######following loop computes stats
    for i in range(b):
        count = 2   ## computing for all trials
        for c in range(count+1):    ########## this loop is required to separate trials
            if c < 2:
                ttmmpp = np.arange(zero_entries[c],zero_entries[c+1])
                PC =  scipy.stats.pearsonr(YP1[ttmmpp,i],YT1[ttmmpp,i])[0]
                NRMSE =  mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)/sc1[i]
                NRMSE_list.append(NRMSE)
                PC_list.append(PC)

            ttmmpp2 = np.arange(zero_entries2[c],zero_entries2[c+1])
            PC2  = scipy.stats.pearsonr(YP2[ttmmpp2,i],YT2[ttmmpp2,i])[0]
            NRMSE2  =  mean_squared_error(YP2[ttmmpp2,i], YT2[ttmmpp2,i],squared=False)/sc2[i]
            NRMSE2_list.append(NRMSE2)
            PC2_list.append(PC2)
    NRMSE_list = np.around(NRMSE_list,2)
    NRMSE2_list = np.around(NRMSE2_list,2)
    PC_list  = np.around(PC_list,2)
    PC2_list = np.around(PC2_list,2)
    print("Printing statisics for ----", label, "mean, std, max, min, iqr")
    print(np.around(np.mean(PC_list),2),' & ' ,  np.around(np.std(PC_list),2),' & ' , np.around(np.max(PC_list),2),' & ' , np.around(np.min(PC_list),2),' & ' ,  np.around(scipy.stats.iqr(PC_list),2),' & ' , 'PC Exposed')    
    print(np.around(np.mean(NRMSE_list),2),' & ' , np.around(np.std(NRMSE_list),2),' & ' , np.around(np.max(NRMSE_list),2),' & ' , np.around(np.min(NRMSE_list),2),' & ' ,  np.around(scipy.stats.iqr(NRMSE_list),2), 'NRMSE2 exposed')    

    print(np.around(np.mean(PC2_list),2),' & ' , np.around(np.std(PC2_list),2),' & ' , np.around(np.max(PC2_list),2),' & ' , np.around(np.min(PC2_list),2),' & ' ,  np.around(scipy.stats.iqr(PC2_list),2),' & ' , 'PC2 Naive')    
    print(np.around(np.mean(NRMSE2_list),2),' & ' , np.around(np.std(NRMSE2_list),2),' & ' , np.around(np.max(NRMSE2_list),2),' & ' , np.around(np.min(NRMSE2_list),2),' & ' ,  np.around(scipy.stats.iqr(NRMSE2_list),2), 'NRMSE2 Naive')    
    
    sparse_plot=5
    for i in range(b):
        push_plot = 0
        count = 0   ## plotting first trial
        for c in range(count+1):    ########## this loop is required to separate trials
            ttmmpp = np.arange(zero_entries[c],zero_entries[c+1])
            ttmmpp2 = np.arange(zero_entries2[c],zero_entries2[c+1])
            
            if ax_list[i] == ax00:
                label1, label2 = 'NN prediction', 'MSK model output'
            else:
                label1, label2 = '_no_legend_', '_no_legend_'
                
            ax_list[i].plot([aa[q] + push_plot for q in ttmmpp][::sparse_plot] ,YP1[ttmmpp,i][::sparse_plot],color='red', lw=0.8,label=label1)   ### np.arange(a)
            ax_list[i].plot([aa[q] + push_plot for q in ttmmpp][::sparse_plot] ,YT1[ttmmpp,i][::sparse_plot],color='blue',lw=0.8,label=label2)

            ax_list2[i].plot([aa[q] + push_plot for q in ttmmpp2][::sparse_plot] ,YP2[ttmmpp2,i][::sparse_plot],color='red', lw=0.8,label ='_no_legend_')#,label=label1)   ### np.arange(a)
            ax_list2[i].plot([aa[q] + push_plot for q in ttmmpp2][::sparse_plot] ,YT2[ttmmpp2,i][::sparse_plot],color='blue',lw=0.8,label ='_no_legend_')#,label=label2)
            Title =  scipy.stats.pearsonr(YP1[ttmmpp,i],YT1[ttmmpp,i])[0]
            NRMSE  = mean_squared_error(YP1[ttmmpp,i], YT1[ttmmpp,i],squared=False)

            Title2  = scipy.stats.pearsonr(YP2[ttmmpp2,i],YT2[ttmmpp2,i])[0]
            NRMSE2  = mean_squared_error(YP2[ttmmpp2,i], YT2[ttmmpp2,i],squared=False)
        
            push_plot = push_plot + 0.1

        NRMSE,NRMSE2 = NRMSE/sc1[i],NRMSE2/sc2[i]

        push2 = 0.05
        ax_list[i].set_xlim(0,count+1)
        ax_list2[i].set_xlim(0,count+1)
        ind = ['Trial '+str(i+1) for i in range(count+1)]
        Title = str(np.around(Title/(count+1),2))
        NRMSE = str(np.around(NRMSE/(count+1),2))
        Title2 = str(np.around(Title2/(count+1),2))
        NRMSE2 = str(np.around(NRMSE2/(count+1),2))
        if len(Title) == 3:
            Title = Title+'0'
        if len(Title2) == 3:
            Title2 = Title2+'0'
        if len(NRMSE) == 3:
            NRMSE = NRMSE+'0'
        if len(NRMSE2) == 3:
            NRMSE2 = NRMSE2+'0'

        Title = plot_list[i] + "  r = " + Title + ", NRMSE = " + NRMSE
        Title2 = plot_list[i] + "  r = " + Title2 + ", NRMSE = " + NRMSE2
        ax_list[i].text(-0.25, 1.1, Title, transform=ax_list[i].transAxes, size=ss)#,fontweight='bold')
        ax_list2[i].text(-0.25, 1.1, Title2, transform=ax_list2[i].transAxes, size=ss)#,fontweight='bold')
        minor_ticks = [] 
        percent = ['0%','25%','50%','75%','100%']
        push3 = 0
        for sn in range(count+1):
            for sn1 in np.arange(sn,sn+1.25,0.25):
                minor_ticks.append(sn1+push3)
            push3=push3+0.1

        ax_list[i].set_xticks(minor_ticks ,minor=True)
        ax_list[i].set_xticks(np.array(minor_ticks[2::5])+0.0005,minor=False)

        ax_list[i].set_ylabel(ylabel[i],fontsize=ss)
        # ax_list[i].yaxis.set_label_coords(-0.28,0.5)

        ax_list2[i].set_xticks(minor_ticks ,minor=True)
        ax_list2[i].set_xticks(np.array(minor_ticks[2::5])+0.0005,minor=False)
        ax_list2[i].set_ylabel(ylabel[i],fontsize=ss)
        # ax_list2[i].yaxis.set_label_coords(-0.28,0.5)

        for axx1,axx2 in zip(ax_list[0:len(ax_list)-2], ax_list2[0:len(ax_list2)-2]):
            axx1.set_xticklabels([],fontsize=ss,minor=False)
            axx2.set_xticklabels([],fontsize=ss,minor=False)

        for axx1,axx2 in zip(ax_list[-2:], ax_list2[-2:]):
            axx1.set_xticklabels([],fontsize=ss,minor=False)
            axx1.set_xticklabels(percent*(count+1),fontsize=ss,minor=True,rotation=45)
            axx1.set_xlabel("% of task completion",fontsize=ss)

            axx2.set_xticklabels([],fontsize=ss,minor=False)
            axx2.set_xticklabels(percent*(count+1),fontsize=ss,minor=True,rotation=45)
            axx2.set_xlabel("% of task completion",fontsize=ss)

        ax_list[i].tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
        ax_list[i].tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
        ax_list[i].tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')

        ax_list2[i].tick_params(axis='x', labelsize=ss,   pad=14,length=3,width=0.5,direction= 'inout',which='major')
        ax_list2[i].tick_params(axis='x', labelsize=ss-1, pad=2, length=3,width=0.5,direction= 'inout',which='minor')
        ax_list2[i].tick_params(axis='y', labelsize=ss,   pad=3, length=3,width=0.5,direction= 'inout')

    if 'JM' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(3, 1.54))
        ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        
    elif 'Angles' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(3, 1.54))
        ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')

    elif 'Muscle' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(2.95, 1.55))
        ax00.text(0.94, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        
    elif 'JRF' in label:
        ax00.legend(fontsize=ss-1,loc='upper center',fancybox=True,ncol=1, frameon=True,framealpha=1, bbox_to_anchor=(2.9, 1.55))
        ax00.text(0.9, 1.35, "(I) Subject-exposed", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
        ax00.text(4.35, 1.35, "(II) Subject-naive", transform=ax00.transAxes, size=ss+0.5,fontweight='bold')
    
    if scale_out == True:
        label = label + '_scaled_out'
    fig.savefig('./plots_out/Both_sub'+'_'+model_class+'_'+label+'_combine'+'.pdf',dpi=600)
    plt.close()

def create_PC_data(model,X1,Y2):    ###obselete now
    Y1 = model.predict(X1)
    Y1,Y2 = np.array(Y1),np.array(Y2) 
    Y1,Y2 = np.nan_to_num(Y1, nan=0),np.nan_to_num(Y2, nan=0) 
    a,b = np.shape(Y1)
    PC = np.zeros(b)
    for i in range(b):
        PC[i] = np.around(scipy.stats.pearsonr(Y1[:,i],Y2[:,i])[0],3)
    return PC

def save_outputs(subject_condition,model,hyper_val, X_Train, Y_Train, X_val, Y_val, X_test, Y_test,label):
    train_error = create_PC_data(model,X_Train, Y_Train)
    val_error = create_PC_data(model,X_val, Y_val)
    test_error = create_PC_data(model,X_test, Y_test)
    mse = np.zeros(np.shape(train_error)[0])
    mse[0] = model.evaluate(X_Train, Y_Train,verbose=0)[0]
    mse[1] = model.evaluate(X_val, Y_val,verbose=0)[0]
    mse[2] = model.evaluate(X_test, Y_test,verbose=0)[0]
    out = np.vstack([mse,train_error, val_error, test_error])
    out = np.nan_to_num(out, nan=0, posinf=2222)
    np.savetxt('./text_out/'+subject_condition+'_'+label+'.txt',out,fmt='%1.6f')
    if 'comp' in label:    ### only saves the model in the final run not in cross validation
        model.save('./model_out/'+subject_condition+'_'+label+'.h5')
    return None

def run_NN(X_Train, Y_Train, X_val, Y_val,hyper_val,which,model_class):
    inp_dim = X_Train.shape[1]
    out_dim = Y_Train.shape[1]
    opt, kinit, batch_size, epoch, act, num_nodes, H_layer, metric, loss, lr, p , regularizer_val =   hyper_val
    if opt == 'Adam':
        optim = keras.optimizers.Adam
    elif opt == 'RMSprop':
        optim = keras.optimizers.RMSprop
    elif opt == 'SGD':
        optim = keras.optimizers.SGD
    #inp_dim, out_dim, nbr_Hlayer, Neu_layer, activation, p_drop, lr, optim,loss,metric,kinit
    loss = keras.losses.mean_squared_error
    if which == 'Muscle':
        final_act = None #'relu'
    else:
        final_act = None
    if model_class == 'NN':
        model = initiate_NN_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'Linear':
        model = initiate_Linear_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    elif model_class == 'LR':
        model = initiate_LR_model(inp_dim, out_dim, H_layer, num_nodes, act, p, lr, optim, loss, [metric], kinit,final_act,regularizer_val)
    history = model.fit(X_Train, Y_Train, validation_data = (X_val,Y_val),epochs=epoch, batch_size=batch_size, verbose=2,shuffle=True)
    return model

def run_cross_valid(subject_condition,which,hyper_arg,hyper_val,k_fold,pca,scale_out, model_class):
    for k in range(k_fold):
        X_Train, Y_Train, X_val, Y_val, X_test, Y_test = split_validation_data(subject_condition,which,pca,scale_out,k)     ###### k index is required for subject_naive case
        model = run_NN(X_Train, Y_Train, X_val, Y_val, hyper_val, which, model_class)
        save_outputs(subject_condition,model, hyper_val, X_Train, Y_Train, X_val, Y_val, X_test, Y_test,label=which+'_hyper_'+str(hyper_arg)+'_k_fold_'+str(k))
    return None

def run_final_model(subject_condition,which,hyper_arg,hyper_val,pca,scale_out, model_class):
	X_Train, Y_Train, X_Test, Y_Test = read_total_data(subject_condition,which,pca,scale_out)
	model = run_NN(X_Train, Y_Train, X_Test, Y_Test, hyper_val, which, model_class)
#	save_outputs(subject_condition,model, hyper_val, X_Train, Y_Train,X_Test, Y_Test, X_Test, Y_Test,label='comp'+which+'_hyper_'+str(hyper_arg)+'_')
	return model

def create_final_model(hyper_arg,hyper_val,which,pca,scale_out, model_class):
	model = run_final_model(which,hyper_arg,hyper_val,pca,scale_out, model_class)
	return model


def load_model(subject_condition,hyper_arg,which):
    path = './model_out/'+subject_condition+'_comp'+which+'_hyper_'+str(hyper_arg)+'_.h5'  
    model = keras.models.load_model(path)
    return model


def plot_saved_model(which, hyper_arg1,hyper_val1, hyper_arg2,hyper_val2, pca,scale_out,model_class):
	_, Y_Train1, X_Test1, Y_Test1,sc1 = read_total_data('subject_exposed',which,pca,scale_out)
	_, Y_Train2, X_Test2, Y_Test2,sc2 = read_total_data('subject_naive',which,pca,scale_out)
	model1 = load_model('subject_exposed',hyper_arg1,which)
	model2 = load_model('subject_naive' ,hyper_arg2,which)
	combined_plot(model1,model2,X_Test1,Y_Test1,X_Test2,Y_Test2,"Test_"+which+"_"+str(hyper_arg1)+"_"+str(hyper_arg2),scale_out,model_class,sc1,sc2)
