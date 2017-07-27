import math, time, thread, sys
import pp
from numpy.random import RandomState
import pandas
import numpy as np
import xgboost
import sklearn
import sklearn.ensemble
import gc
import warnings
import cPickle as pickle


# parallel cerenkov framework

def method_generate():
    pass

def locus_group(dataset, cutoff_bp):
    ''' distribute each SNP into a group and give each SNP a group ID
        * input: feature matrix
        * output: feature matrix with group id
    '''
    feat = dataset # //TODO check if assigning value by "=" will have reference
    feat["coord"] = dataset.index.get_level_values("coord")
    feat["group_id"] = ""
    chromSet = [str(i) for i in range(1,23)]+["X"]

    for chrom in chromSet:
        chrom_name = "chromchr" + chrom
        SNP_chrom = feat.loc[feat[chrom_name]==1]
        SNP_chrom = SNP_chrom.sort_values(["coord"], ascending=True) # //TODO need to add a "ChromCoord" column into feature matrix, since the coordinate is normalized.
        SNP_chrom["group_id"] = SNP_chrom["coord"] - SNP_chrom["coord"].shift() # calculate the difference of adjacent ChromCoord
        
        SNP_chrom.ix[0,"group_id"] = 1.0 # fill in the missing first difference of ChromCoord
        SNP_chrom["group_id"] = SNP_chrom["group_id"] > cutoff_bp # True if distance > cutoff_bp; else False
        SNP_chrom["group_id"] = SNP_chrom["group_id"].astype(int) # transform to integer
        SNP_chrom["group_id"] = SNP_chrom["group_id"].cumsum(axis=0) # cumsum the "group_id" column
        SNP_chrom["group_id"] = SNP_chrom["group_id"].astype(str)
        SNP_chrom["group_id"] = chrom + "_" + SNP_chrom["group_id"] # add chrom prefix to group id
        
        feat.loc[SNP_chrom.index, "group_id"] = SNP_chrom["group_id"] # assign values back to feature matrix

    del feat["coord"]
    return feat





def locus_sampling(dataset, n_rep, n_fold, cutoff_bp=50000, slope_allowed=0.5, seed=1337):

    '''
        * input: label (Pandas DataFrame with rsnp id as index)
        * output: assigned groups
    '''
    feat = locus_group(dataset, cutoff_bp)
    label = dataset["label"] # //TODO check if label necessary
    n_case = len(dataset)
    n_pos = np.sum(dataset["label"])
    max_fold_case_num = math.ceil((1 + slope_allowed) * n_case / n_fold)
    max_fold_pos_num = math.ceil((1 + slope_allowed) * max_fold_case_num * n_pos / n_case)
    fold_case_num = [0 for i in range(n_fold)] # initialize a fold case number list
    fold_pos_num = [0 for i in range(n_fold)] # initialize a fold positive case number list
    fold = pandas.DataFrame(data=0, index=dataset.index, columns=np.arange(1, n_rep+1))

    # assign groups to folds
    for group in feat.groupby("group_id"):
        # check if there is at least 1 rSNP in each group
        if group[1]["label"].nonzero() is None:
            print "[ERROR INFO] There is no positive cases in group ", group[0]
            sys.exit()

    # assign each group
    group_case = [group for group in feat.groupby("group_id")]
    group_case.sort(key=lambda x: len(x[1]), reverse=True) # sort the group case according to number of elements each group
    
    for i_rep in range(1, n_rep+1):

        rs = RandomState(seed+i_rep)

        for group in group_case:

            group_count = len(group[1])
            group_pos_count = np.sum(group[1]["label"])

            ind_allowed = [i for i in range(n_fold) if fold_case_num[i] + group_count <= max_fold_case_num and fold_pos_num[i] + group_pos_count <= max_fold_pos_num ]
            
            # sample from allowed indexes
            if len(ind_allowed) > 1:
                probs = np.array([(1.0 - (fold_case_num[i]*1.0 / max_fold_case_num) * (1.0 - (fold_pos_num[i]*1.0 / max_fold_pos_num))) for i in ind_allowed])
                norm_probs = probs / probs.sum() # np.random.choice need probabilities summed up to 1.0
                ind_selected = rs.choice(ind_allowed, size=1, p=norm_probs)[0]
            else:
                ind_selected = ind_allowed[0]

            fold.loc[group[1].index.values, i_rep] = ind_selected + 1
            fold_case_num[ind_selected] += group_count
            fold_pos_num[ind_selected] += group_pos_count

        # check if all SNP are assigned
        if 0 in fold.ix[:, i_rep]:
            print "[ERROR INFO] Some SNP is not assigned to any fold!"
            sys.exit()

        fold_case_num = [0] * n_fold
        fold_pos_num = [0] * n_fold

    return fold





def snp_sampling(dataset, n_rep, n_fold, seed=1337):
    '''
        * input: dataset (Pandas DataFrame with rsnp id as index)
        * output: assigned groups
    '''
    # //TODO think about whether we need to balance positive-negative case numbers in each fold, or totally random?
    np.random.seed(seed)
    n_label = len(dataset)
    fold = pandas.DataFrame(data=0, index=dataset.index, columns=np.arange(1, n_rep+1))

    for i_rep in range(1, n_rep+1):

        fold0 = [(x%n_fold+1) for x in range(n_label)] # initial fold: 1,2,3,4...,n_label
        np.random.shuffle(fold0)
        fold.ix[:,i_rep] = fold0

    return fold





def fold_generate(dataset, n_rep, n_fold, fold_assign_method, **kwargs):
    
    # assign each SNP to a fold in each replication
    if fold_assign_method == "SNP":
        fold = snp_sampling(dataset, n_rep, n_fold, kwargs["seed"])
    else:
        fold = locus_sampling(dataset, n_rep, n_fold, kwargs["cutoff_bp"], kwargs["slope_allowed"], kwargs["seed"])
  
    # assign SNPs to each train-test process
    fold_table = pandas.DataFrame(columns=["fold_no", "train_index", "test_index"], index=np.arange(1, n_rep * n_fold + 1))

    for i_rep in range(1, n_rep+1):
        for i_fold in range(1, n_fold+1):
            test_index = fold.loc[fold[i_rep]==i_fold].index.values
            train_index = fold.loc[fold[i_rep]!=i_fold].index.values
            fold_table.ix[(i_rep-1)*n_fold+i_fold, "fold_no"] = (i_rep, i_fold)
            fold_table.ix[(i_rep-1)*n_fold+i_fold, "train_index"] = train_index
            fold_table.ix[(i_rep-1)*n_fold+i_fold, "test_index"] = test_index
    
    # //TODO pay attention that: a dataframe generated by pandas==20.3 cannot be pickle loaded by pandas==18.0!!!
    pickle.dump(fold_table, open("fold_table.p", "wb"))
    print "dump fold_table to fold_table.p!"
    
    return fold_table





def hp_generate():
    # generate the hyperparameters for each classifier
    pass
    # return hp_table





def task_pool():
    pass





def cerenkov_ml():
    pass





def cerenkov_analysis():
    pass





# machine learning methods

def cerenkov17(dataset, hyperparameters, fold, fold_assign_method):
    pass





def gwava_rf(dataset, hyperparameters, fold, fold_assign_method):
    pass





def gwava_xgb(dataset, hyperparameters, fold, fold_assign_method):
    pass