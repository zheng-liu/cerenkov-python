# In python 2.7, the default `/` operator is integer division if inputs are integers.
# If you want float division, use this special import
from __future__ import division
import math, time, thread, sys
from numpy.random import RandomState
import pandas
import numpy as np
import xgboost
import sklearn
import sklearn.ensemble
import gc
import warnings
import cPickle as pickle
import itertools
import multiprocessing


# parallel cerenkov framework
def method_generate():
    
    method_table = [cerenkov17]
    pickle.dump(method_table, open("method_table.p", "wb"))
    print "[INFO] dump to method_table.p!"
    return method_table


def locus_group(dataset, cutoff_bp):
    """ distribute each SNP into a group and give each SNP a group ID
        * input: feature matrix
        * output: feature matrix with group id
    """
    feat = dataset  # TODO check if assigning value by "=" will have reference
    feat["group_id"] = ""
    chromSet = [str(i) for i in range(1, 23)]+["X"]

    for chrom in chromSet:
        chrom_name = "chromchr" + chrom
        SNP_chrom = feat.loc[feat[chrom_name] == 1]
        SNP_chrom = SNP_chrom.sort_values(["coord"], ascending=True)
        # calculate the difference of adjacent ChromCoord
        SNP_chrom["group_id"] = SNP_chrom["coord"] - SNP_chrom["coord"].shift()
        
        SNP_chrom.ix[0, "group_id"] = 1.0  # fill in the missing first difference of ChromCoord
        SNP_chrom["group_id"] = SNP_chrom["group_id"] > cutoff_bp  # True if distance > cutoff_bp; else False
        SNP_chrom["group_id"] = SNP_chrom["group_id"].astype(int)  # transform to integer
        SNP_chrom["group_id"] = SNP_chrom["group_id"].cumsum(axis=0)  # cumsum the "group_id" column
        SNP_chrom["group_id"] = SNP_chrom["group_id"].astype(str)
        SNP_chrom["group_id"] = chrom + "_" + SNP_chrom["group_id"]  # add chrom prefix to group id
        
        feat.loc[SNP_chrom.index, "group_id"] = SNP_chrom["group_id"]  # assign values back to feature matrix

    return feat


def locus_sampling(dataset, n_rep, n_fold, cutoff_bp=50000, slope_allowed=0.5, seed=1337):
    # Use 1-D index here for accelaration
    feat = dataset.reset_index(level=1)  # restore "coord" column
    feat = locus_group(feat, cutoff_bp)

    n_case = feat.shape[0]
    n_pos = np.sum(feat["label"])

    max_fold_case_num = math.ceil((1 + slope_allowed) * n_case / n_fold)
    max_fold_pos_num = math.ceil((1 + slope_allowed) * max_fold_case_num * n_pos / n_case)

    # `fold` here uses 1-D index from `feat`, not 2-D index from `dataset`
    fold = pandas.DataFrame(data=0, index=feat.index, columns=np.arange(n_rep))

    groups = feat.groupby("group_id")

    for group in groups:
        # check if there is at least 1 rSNP in each group
        if group[1]["label"].nonzero() is None:
            print "[ERROR INFO] There is no positive cases in group ", group[0]
            sys.exit()

    group_case = list(groups)
    # sort the group case according to number of elements each group (in place)
    group_case.sort(key=lambda x: x[1].shape[0], reverse=True)

    for i_rep in range(n_rep):

        fold_case_num = [0] * n_fold  # initialize a fold case number list
        fold_pos_num = [0] * n_fold  # initialize a fold positive case number list

        rs = RandomState(seed + i_rep)

        for group in group_case:

            group_count = group[1].shape[0]
            group_pos_count = np.sum(group[1]["label"])

            ind_allowed = [i for i in range(n_fold) if fold_case_num[i] + group_count <= max_fold_case_num
                           and fold_pos_num[i] + group_pos_count <= max_fold_pos_num]

            # sample from allowed indexes
            if len(ind_allowed) > 1:
                probs = np.array([(1 - (fold_case_num[i] / max_fold_case_num) *
                                   (1 - (fold_pos_num[i] / max_fold_pos_num))) for i in ind_allowed])
                norm_probs = probs / probs.sum()  # np.random.choice need probabilities summed up to 1.0
                ind_selected = rs.choice(ind_allowed, size=1, p=norm_probs)[0]
            else:
                ind_selected = ind_allowed[0]

            fold.loc[group[1].index.values.tolist(), i_rep] = ind_selected + 1

            fold_case_num[ind_selected] += group_count
            fold_pos_num[ind_selected] += group_pos_count

        # check if all SNP are assigned
        if 0 in fold.loc[:, i_rep]:
            print "[ERROR INFO] Some SNP is not assigned to any fold!"
            sys.exit()
    
    # reconstruct 2nd index from "coord" column for `fold`, as same as `dataset`
    fold = fold.merge(feat.loc[:, "coord"].to_frame(), how="left", left_index=True, right_index=True)
    fold.set_index("coord", append=True, inplace=True)
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
    print "[INFO] dump fold_table to fold_table.p!"
    
    return fold_table





def hp_generate():
    # generate the hyperparameters for each classifier
    learning_rate = [0.1]
    n_estimators = [30]
    gamma = [10]
    subsample = [1]
    colsample_bytree = [0.5, 0.55]
    base_score = [0.1082121]
    scale_pos_weight = [1]
    max_depth = [5,6]
    seed=[1337]
    nthread = [1]

    hp_comb = itertools.product(
    	              learning_rate, 
                      n_estimators,
                      gamma,
                      subsample,
                      colsample_bytree,
                      base_score,
                      scale_pos_weight,
                      max_depth,
                      seed,
                      nthread)
    #n_hp = len(list(hp_comb))
    hp_key = (
        "learning_rate",
        "n_estimators",
        "gamma",
        "subsample",
        "colsample_bytree",
        "base_score",
        "scale_pos_weight",
        "max_depth",
        "seed",
        "nthread"
    	)
    hp_col = [dict(zip(hp_key, hp)) for hp in hp_comb]
    n_hp = len(hp_col)
    cerenkov17_hp = pandas.DataFrame(columns=["hp_no", "hp"], index=np.arange(1, n_hp + 1))
    cerenkov17_hp["hp_no"] = np.arange(1, n_hp + 1)
    cerenkov17_hp["hp"] = hp_col

    hp_table = {
        "cerenkov17": cerenkov17_hp
    }
    
    pickle.dump(hp_table, open("hp_table.p", "wb"))
    print "[INFO] dump to hp_table.p!"
    return hp_table
    




def data_generate(method, data):
    data_table = dict(zip(method, data))
    pickle.dump(data_table, open("data_table.p", "wb"))
    print "[INFO] dump to data_table.p!"
    return data_table





def task_pool(method_table, hp_table, data_table, fold_table):
    n_hp = sum([len(hp_table[method.__name__]) for method in method_table])
    n_fold = len(fold_table)
    n_task = n_hp * n_fold
    task_ind = 1
    task_table = pandas.DataFrame(columns=["method", "hp", "data", "fold"], index=np.arange(1, n_task + 1))
    
    for method in method_table:
        method_str = method.__name__
        for hp_ind, hp in hp_table[method_str].iterrows():
            for fold_ind, fold in fold_table.iterrows():
                task_table.ix[task_ind, "method"] = method
                task_table.ix[task_ind, "hp"] = hp_ind
                task_table.ix[task_ind, "data"] = method_str
                task_table.ix[task_ind, "fold"] = fold_ind
                task_ind += 1

    pickle.dump(task_table, open("task_table.p", "wb"))
    print "[INFO] dump to task_table.p!"
    return task_table





def cerenkov_ml(task_table, method_table, fold_table, hp_table, data_table, fold_assign_method, ncpus, feature_reduced):
    
    if fold_assign_method == "SNP":
        task_table["auroc"] = -1.0
        task_table["aupvr"] = -1.0
    else:
        task_table["avgrank"] = -1.0
    
    if ncpus == -1:
        n_cores = multiprocessing.cpu_count()
    else:
        n_cores = ncpus

    p = multiprocessing.Pool(processes=n_cores)
    result_pool = []

    for task_no, task in task_table.iterrows():
        
        method = task["method"]
        method_name = method.__name__
        hp_ind = task["hp"]
        hp = hp_table[method_name].ix[hp_ind, "hp"] # time: 0.0004
        data_ind = task["data"] # time: 3.0e-5
        data = data_table[data_ind] # time: 9.0e-7
        fold_ind = task["fold"] # time: 2.0e-5
        fold = fold_table.loc[[fold_ind]] # time: 6.0e-4
        args = (data, hp, fold, fold_assign_method, task_no) # time: 5.0e-6
        result_pool.append(p.apply_async(method, args=args))
        print "[INFO] a job submitted!", "task_no", task_no
    
    p.close()
    p.join()

    if fold_assign_method == "SNP":

        for result in result_pool:
            result = result.get()
            result_task_no = result["task_no"]
            task_table.ix[result_task_no, "auroc"] = result["auroc"]
            task_table.ix[result_task_no, "aupvr"] = result["aupvr"]
    else:
        for result in result_pool:
            result = result.get()
            result_task_no = result["task_no"]
            task_table.ix[result_task_no, "avgrank"] = result["avgrank"]
    
    pickle.dump(task_table, open("task_table.p", "wb"))
    print "[INFO] dump to task_table.p!"
    return task_table


def cerenkov_analysis(fold_assign_method):
    # //TODO check if all the files are there
    method_table = pickle.load(open("method_table.p", "rb"))
    data_table = pickle.load(open("data_table.p", "rb"))
    fold_table = pickle.load(open("fold_table.p", "rb"))
    hp_table = pickle.load(open("hp_table.p", "rb"))
    task_table = pickle.load(open("task_table.p", "rb"))

    if fold_assign_method == "SNP":
        pass

# machine learning methods

def cerenkov17(dataset, hyperparameters, fold, fold_assign_method, task_no):
    
    if fold_assign_method == "LOCUS":
        feat = dataset.drop(["label", "group_id"], axis=1)
    else:
        feat = dataset.drop(["label"], axis=1)

    label = dataset["label"]

    train_index = fold["train_index"].values[0].tolist()
    test_index = fold["test_index"].values[0].tolist()

    X_train = feat.loc[train_index]
    y_train = label.loc[train_index]
    X_test = feat.loc[test_index]
    y_test = label.loc[test_index]

    clf_cerenkov17 = xgboost.XGBClassifier(**hyperparameters)
    clf_cerenkov17.fit(X_train, y_train)

    y_test_probs = clf_cerenkov17.predict_proba(X_test)[:, clf_cerenkov17.classes_ == 1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs
    if fold_assign_method == "LOCUS":
        rank_test = pandas.DataFrame(data=y_test_probs, columns=["probs"])
        rank_test["group_id"] = dataset.loc[X_test.index.values, "group_id"].values
        rank_test["label"] = y_test
        rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False)
        avgrank = rank_test.loc[rank_test["label"]==1]["rank"].mean()
        result = {"avgrank": avgrank, "task_no": task_no}
    else:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)

        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
        
        result = {"auroc": auroc, "aupvr": aupvr, "task_no": task_no}
    print "[INFO] cerenkov17 done! task no: ", task_no
    return result






def gwava_rf(dataset, hyperparameters, fold, fold_assign_method):
    pass





def gwava_xgb(dataset, hyperparameters, fold, fold_assign_method):
    pass




