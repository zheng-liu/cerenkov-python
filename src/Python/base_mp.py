# In python 2.7, the default `/` operator is integer division if inputs are integers.
# If you want float division, use this special import
from __future__ import division
import math, time, thread, sys
from numpy.random import RandomState
import pandas as pd
import numpy as np
import scipy
import xgboost
import sklearn
import sklearn.ensemble
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
import cPickle as pickle
import itertools
import multiprocessing




def binary_encode(dfm, cat_col_name, cat_col_sep, new_col_prefix=None, integrate=False):
    """
    Binary-encode categorical column `cat_col_name` separated by `cat_col_sep`
    in data frame `dfm` to multiple binary columns with the same prefix `new_col_prefix`

    :param dfm: the data frame
    :param cat_col_name: the name of the categorical column whose values would be encoded
    :param cat_col_sep: the separator of the categorical values
    :param new_col_prefix: the prefix of the new columns after binary encoding
    :param integrate:
        if True, integrate the encoded part into the original data frame and return the integrated one;
        otherwise return only the encoded part
    :return:
    """

    """
    E.g. MySQL returns `GROUP_CONCAT(tf.name)` in `tfName` column in a comma-separated string, such as

        tfName
        ARID3A,ATF1,ATF2
        ARID3A

    Calling `binary_encode(dfm, 'tfName', ',', 'tf_')` would transform the column as:

        tfName               tf_ARID3A   tf_ATF1  tf_ATF2
        ARID3A,ATF1,ATF2  =>     1           1        1
        ARID3A                   1           0        0
    """
    encoded = dfm.loc[:, cat_col_name].str.get_dummies(sep=cat_col_sep)

    if new_col_prefix is not None:
        # Add a prefix to all column names
        encoded = encoded.add_prefix(new_col_prefix)

    if integrate:
        dfm = pd.concat([dfm, encoded], axis=1).drop(cat_col_name, axis=1)
        return dfm
    else:
        return encoded

def one_hot_encode(dfm, cat_col_name, new_col_prefix=None, integrate=False):
    """
    One-hot-encode categorical column `cat_col_name` in data frame `dfm`
    to multiple binary columns with the same prefix `new_col_prefix`
    """

    """
    LabelEncoder transforms the categorical values to integers. E.g

        minorAllele      minorAllele
            A                0
            C        =>      1
            G                2
            T                3

    Note that data structure is changed from `pandas.Series` to `numpy.ndarray`
    """
    le = LabelEncoder()
    le_data = pd.Series(le.fit_transform(dfm.loc[:, cat_col_name]))  # le_data.shape == (nrow,)

    # reshaping is required by OneHotEncoder below
    # `reshape` is not well documented by Numpy developers. Refer to https://stackoverflow.com/a/42510505 instead
    le_data = le_data.values.reshape(-1, 1)  # le_data.shape == (nrow, 1)

    """
    OneHotEncoder encodes categorical integer features using one-hot scheme. E.g

        minorAllele      0  1  2  3
            0            1  0  0  0
            1        =>  0  1  0  0
            2            0  0  1  0
            3            0  0  0  1
    """
    ohe = OneHotEncoder(dtype=np.int, sparse=False)
    ohe_data = ohe.fit_transform(le_data)

    """
    Encapsulate the encoded data into a DataFrame. E.g.

        0  1  2  3      A  C  G  T
        1  0  0  0      1  0  0  0
        0  1  0  0  =>  0  1  0  0
        0  0  1  0      0  0  1  0
        0  0  0  1      0  0  0  1

    `le.classes_` stores the original categorical values; it would be ['A', 'C', 'G', 'T'] in the example.
    """
    ohe_data = pd.DataFrame(data=ohe_data, columns=le.classes_)

    if new_col_prefix is not None:
        # Add a prefix to all column names
        ohe_data = ohe_data.add_prefix(new_col_prefix)

    if integrate:
        dfm = pd.concat([dfm, ohe_data], axis=1).drop(cat_col_name, axis=1)
        return dfm
    else:
        return ohe_data





# parallel cerenkov framework
def method_generate():
    
    method_table = [cerenkov17, gwava_xgb, gwava_rf]
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
    # del feat["coord"]
    return feat





def locus_sampling(dataset, n_rep, n_fold, cutoff_bp=50000, slope_allowed=0.5, seed=1337):
    # Use 1-D index here for accelaration
    feat = dataset.reset_index(level="coord")  # restore "coord" column
    feat = locus_group(feat, cutoff_bp)

    n_case = feat.shape[0]
    n_pos = np.sum(feat["label"])

    max_fold_case_num = math.ceil((1 + slope_allowed) * n_case / n_fold)
    max_fold_pos_num = math.ceil((1 + slope_allowed) * max_fold_case_num * n_pos / n_case)

    # `fold` here uses 1-D index from `feat`, not 2-D index from `dataset`
    fold = pd.DataFrame(data=0, index=feat.index, columns=np.arange(1, n_rep+1))

    groups = feat.groupby("group_id")

    for group in groups:
        # check if there is at least 1 rSNP in each group
        if group[1]["label"].nonzero() is None:
            print "[ERROR INFO] There is no positive cases in group ", group[0]
            sys.exit()

    group_case = list(groups)
    # sort the group case according to number of elements each group (in place)
    group_case.sort(key=lambda x: x[1].shape[0], reverse=True)

    for i_rep in range(1, n_rep+1):

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
    fold = pd.DataFrame(data=0, index=dataset.index, columns=np.arange(1, n_rep+1))

    for i_rep in range(1, n_rep+1):

        fold0 = [(x%n_fold+1) for x in range(n_label)] # initial fold: 1,2,3,4...,n_label
        np.random.shuffle(fold0)
        fold.ix[:,i_rep] = fold0

    return fold





def fold_generate(dataset, n_rep, n_fold, fold_assign_method, **kwargs):
    
    # assign each SNP to a fold in each replication
    if fold_assign_method == "SNP":
        fold = snp_sampling(dataset, n_rep, n_fold, kwargs["seed"])
    elif fold_assign_method == "LOCUS":
        fold = locus_sampling(dataset, n_rep, n_fold, kwargs["cutoff_bp"], kwargs["slope_allowed"], kwargs["seed"])
    else:
        print "Invalid fold assign method!"
        exit(0)
  
    # assign SNPs to each train-test process
    fold_table = pd.DataFrame(columns=["fold_index", "i_fold", "i_rep", "train_index", "test_index"], index=np.arange(1, n_rep * n_fold + 1))

    for i_rep in range(1, n_rep + 1):
        for i_fold in range(1, n_fold + 1):
            test_index = fold.loc[fold[i_rep] == i_fold].index.values
            train_index = fold.loc[fold[i_rep] != i_fold].index.values
            fold_table.ix[(i_rep - 1) * n_fold + i_fold, "i_fold"] = i_fold
            fold_table.ix[(i_rep - 1) * n_fold + i_fold, "i_rep"] = i_rep
            fold_table.ix[(i_rep - 1) * n_fold + i_fold, "fold_index"] = (i_rep - 1) * n_fold + i_fold
            fold_table.ix[(i_rep - 1) * n_fold + i_fold, "train_index"] = train_index
            fold_table.ix[(i_rep - 1) * n_fold + i_fold, "test_index"] = test_index
    
    # //TODO pay attention that: a dataframe generated by pandas==20.3 cannot be pickle loaded by pandas==18.0!!!
    pickle.dump(fold_table, open("fold_table.p", "wb"))
    print "[INFO] dump fold_table to fold_table.p!"

    return fold_table





def hp_generate(pos_neg_ratio):

    # xgboost-GBDT: 3888

    # maximize -- AUPVR
    # eta = 0.1
    # gamma = 10
    # n_estimators = 30
    # max_depth=6
    # subsample = 1.0
    # colsample_bytree = 0.85
    # scale_pos_weight = 1

    # minimize -- AVGRANK
    # eta = 0.1
    # gamma = 100
    # n_estimators = 30
    # max_depth = 6
    # subsample = 0.85
    # scale_pos_weight = 8
    

    # generate the hyperparameters for each classifier

    # # cerenkov17 hp tuning set
    # cerenkov17_learning_rate = [0.01, 0.05, 0.1]
    # cerenkov17_n_estimators = [10, 20, 30]
    # cerenkov17_gamma = [0, 1, 10, 100]
    # cerenkov17_subsample = [0.5, 0.75, 0.85, 1.0]
    # cerenkov17_colsample_bytree = [0.5, 0.75, 0.85]
    # cerenkov17_scale_pos_weight = [0.125, 1.0, 8.0]
    # cerenkov17_max_depth = [6, 8, 10]
    # cerenkov17_seed=[1337]
    # cerenkov17_nthread = [1]



    # # cerenkov17 hp for OSU17 dataset
    # cerenkov17_learning_rate = [0.1]
    # cerenkov17_n_estimators = [30]
    # cerenkov17_gamma = [10]
    # cerenkov17_subsample = [1.0]
    # cerenkov17_colsample_bytree = [0.85]
    # cerenkov17_scale_pos_weight = [1.0]
    # cerenkov17_max_depth = [6]
    # cerenkov17_seed=[1337]
    # cerenkov17_nthread = [1]

    # cerenkov17_hp_comb = itertools.product(
    #                   cerenkov17_learning_rate, 
    #                   cerenkov17_n_estimators,
    #                   cerenkov17_gamma,
    #                   cerenkov17_subsample,
    #                   cerenkov17_colsample_bytree,
    #                   cerenkov17_scale_pos_weight,
    #                   cerenkov17_max_depth,
    #                   cerenkov17_seed,
    #                   cerenkov17_nthread)

    # cerenkov17_hp_key = (
    #     "learning_rate",
    #     "n_estimators",
    #     "gamma",
    #     "subsample",
    #     "colsample_bytree",
    #     "scale_pos_weight",
    #     "max_depth",
    #     "seed",
    #     "nthread"
    #     )

    # cerenkov17_hp_col = [dict(zip(cerenkov17_hp_key, cerenkov17_hp)) for cerenkov17_hp in cerenkov17_hp_comb]
    # cerenkov17_n_hp = len(cerenkov17_hp_col)
    # cerenkov17_hp = pd.DataFrame(columns=["hp_no", "hp"], index=np.arange(1, cerenkov17_n_hp + 1))
    # cerenkov17_hp["hp_no"] = np.arange(1, cerenkov17_n_hp + 1)
    # cerenkov17_hp["hp"] = cerenkov17_hp_col


    # cerenkov17 hp for OSU18 dataset
    cerenkov17_learning_rate = [0.1]
    cerenkov17_n_estimators = [40]
    cerenkov17_gamma = [10]
    cerenkov17_reg_lambda = [1]
    cerenkov17_subsample = [1.0]
    cerenkov17_base_score = [0.0647084410101579]
    cerenkov17_scale_pos_weight = [1.0]
    cerenkov17_max_depth = [7]
    cerenkov17_seed=[1337]
    cerenkov17_nthread = [1]

    cerenkov17_hp_comb = itertools.product(
                      cerenkov17_learning_rate, 
                      cerenkov17_n_estimators,
                      cerenkov17_gamma,
                      cerenkov17_reg_lambda,
                      cerenkov17_subsample,
                      cerenkov17_base_score,
                      cerenkov17_scale_pos_weight,
                      cerenkov17_max_depth,
                      cerenkov17_seed,
                      cerenkov17_nthread)

    cerenkov17_hp_key = (
        "learning_rate",
        "n_estimators",
        "gamma",
        "reg_lambda",
        "subsample",
        "base_score",
        "scale_pos_weight",
        "max_depth",
        "seed",
        "nthread"
        )

    cerenkov17_hp_col = [dict(zip(cerenkov17_hp_key, cerenkov17_hp)) for cerenkov17_hp in cerenkov17_hp_comb]
    cerenkov17_n_hp = len(cerenkov17_hp_col)
    cerenkov17_hp = pd.DataFrame(columns=["hp_no", "hp"], index=np.arange(1, cerenkov17_n_hp + 1))
    cerenkov17_hp["hp_no"] = np.arange(1, cerenkov17_n_hp + 1)
    cerenkov17_hp["hp"] = cerenkov17_hp_col


    # gwava_rf
    gwava_rf_n_estimators = [100]
    gwava_rf_max_features = [14] # //TODO check if max_features is the mtry in R Ranger Random Forest
    gwava_rf_bootstrap = [True]
    gwava_rf_n_jobs = [1]
    gwava_rf_random_state = [1337]

    gwava_rf_hp_comb = itertools.product(
                       gwava_rf_n_estimators,
                       gwava_rf_max_features,
                       gwava_rf_bootstrap,
                       gwava_rf_n_jobs,
                       gwava_rf_random_state)

    gwava_rf_hp_key = (
        "n_estimators",
        "max_features",
        "bootstrap",
        "n_jobs",
        "random_state"
        )
    
    gwava_rf_hp_col = [dict(zip(gwava_rf_hp_key, gwava_rf_hp)) for gwava_rf_hp in gwava_rf_hp_comb]
    gwava_rf_n_hp = len(gwava_rf_hp_col)
    gwava_rf_hp = pd.DataFrame(columns=["hp_no", "hp"], index=np.arange(1, gwava_rf_n_hp + 1))
    gwava_rf_hp["hp_no"] = np.arange(1, gwava_rf_n_hp + 1)
    gwava_rf_hp["hp"] = gwava_rf_hp_col



    # gwava_xgb hp for OSU18 dataset
    gwava_xgb_learning_rate = [0.15]
    gwava_xgb_n_estimators = [40]
    gwava_xgb_gamma = [5]
    gwava_xgb_reg_lambda = [10]
    gwava_xgb_subsample = [1.0]
    gwava_xgb_base_score = [0.0647084410101579]
    gwava_xgb_scale_pos_weight = [1.0]
    gwava_xgb_max_depth = [7]
    gwava_xgb_seed=[1337]
    gwava_xgb_nthread = [1]

    gwava_xgb_hp_comb = itertools.product(
                      gwava_xgb_learning_rate, 
                      gwava_xgb_n_estimators,
                      gwava_xgb_gamma,
                      gwava_xgb_reg_lambda,
                      gwava_xgb_subsample,
                      gwava_xgb_base_score,
                      gwava_xgb_scale_pos_weight,
                      gwava_xgb_max_depth,
                      gwava_xgb_seed,
                      gwava_xgb_nthread)

    gwava_xgb_hp_key = (
        "learning_rate",
        "n_estimators",
        "gamma",
        "reg_lambda",
        "subsample",
        "base_score",
        "scale_pos_weight",
        "max_depth",
        "seed",
        "nthread"
        )

    gwava_xgb_hp_col = [dict(zip(gwava_xgb_hp_key, gwava_xgb_hp)) for gwava_xgb_hp in gwava_xgb_hp_comb]
    gwava_xgb_n_hp = len(gwava_xgb_hp_col)
    gwava_xgb_hp = pd.DataFrame(columns=["hp_no", "hp"], index=np.arange(1, gwava_xgb_n_hp + 1))
    gwava_xgb_hp["hp_no"] = np.arange(1, gwava_xgb_n_hp + 1)
    gwava_xgb_hp["hp"] = gwava_xgb_hp_col



    # generate hp_table
    hp_table = {
        "cerenkov17": cerenkov17_hp,
        "gwava_xgb": gwava_xgb_hp,
        "gwava_rf": gwava_rf_hp
    }
    
    pickle.dump(hp_table, open("hp_table.p", "wb"))
    print "[INFO] dump to hp_table.p!"
    return hp_table
    




def pre_data_generate(data):
    
    data["coord"] = data.index.get_level_values("coord").values.tolist()
    return data

def post_data_generate(data):
    del data["coord"]
    return data


def data_generate(method, data, fold_assign_method, **kwargs):
    if fold_assign_method == "SNP":
        data_table = dict(zip(method, data))
    elif fold_assign_method == "LOCUS":
        cutoff_bp = kwargs["cutoff_bp"]
        data = [post_data_generate(locus_group(pre_data_generate(d), cutoff_bp)) for d in data]
        data_table = dict(zip(method, data))
    else:
        print "Invalid fold assign method!"
        exit(0)

    pickle.dump(data_table, open("data_table.p", "wb"))
    print "[INFO] dump to data_table.p!"
    return data_table





def task_pool(method_table, hp_table, data_table, fold_table):
    n_hp = sum([len(hp_table[method.__name__]) for method in method_table])
    n_fold = len(fold_table)
    n_task = n_hp * n_fold
    task_ind = 1
    task_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold"], index=np.arange(1, n_task + 1))
    
    for method in method_table:
        method_str = method.__name__
        print method_str
        for hp_ind, hp in hp_table[method_str].iterrows():
            for fold_ind, fold in fold_table.iterrows():
                task_table.ix[task_ind, "method"] = method
                task_table.ix[task_ind, "hp"] = hp_ind
                task_table.ix[task_ind, "data"] = method_str
                task_table.ix[task_ind, "fold_index"] = fold["fold_index"]
                task_table.ix[task_ind, "i_rep"] = fold["i_rep"]
                task_table.ix[task_ind, "i_fold"] = fold["i_fold"]
                task_ind += 1

    return task_table





def cerenkov_ml(task_table, method_table, fold_table, hp_table, data_table, fold_assign_method, ncpus, feature_reduced):
    
    if fold_assign_method == "SNP":
        task_table["auroc"] = -1.0
        task_table["aupvr"] = -1.0
    elif fold_assign_method == "LOCUS":
        task_table["auroc"] = -1.0
        task_table["aupvr"] = -1.0
        task_table["avgrank"] = -1.0
    else:
        print "Invalid fold assign method!"
        exit(0)
    

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
        fold_index = task["fold_index"] # time: 2.0e-5
        fold = fold_table.loc[[fold_index]] # time: 6.0e-4
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
            task_table.ix[result_task_no, "auroc"] = result["auroc"]
            task_table.ix[result_task_no, "aupvr"] = result["aupvr"]
            task_table.ix[result_task_no, "avgrank"] = result["avgrank"]

    # convert function to string of function name
    for task_no, task in task_table.iterrows():
        task_table.ix[task_no, "method"] = task["method"].__name__

    pickle.dump(task_table, open("task_table.p", "wb"))
    print "[INFO] dump to task_table.p!"
    return task_table





# def cerenkov_analysis(fold_assign_method):

#     sns.set(style="whitegrid", color_codes=True)

#     # //TODO check if all the files are there
#     method_table = pickle.load(open("method_table.p", "rb"))
#     data_table = pickle.load(open("data_table.p", "rb"))
#     fold_table = pickle.load(open("fold_table.p", "rb"))
#     hp_table = pickle.load(open("hp_table.p", "rb"))
#     task_table = pickle.load(open("task_table.p", "rb"))
    
#     analysis = {"hp_optimal":{}}

#     if fold_assign_method == "SNP":

#     	# analyze results
#         plot_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold", "auroc", "aupvr"])

#         for i, gb_i in task_table.groupby(["method"]):
#             method = i
#             hp_optimal = 0
#             aupvr_optimal = -1.0

#             for j, gb_j in gb_i.groupby(["hp"]):
#                 if gb_j["aupvr"].mean() > aupvr_optimal:
#                     hp_optimal = j
#                     aupvr_optimal = gb_j["aupvr"].mean()
            
#             analysis["hp_optimal"][i] = hp_table[i].ix[j, "hp"]
#             plot_table = plot_table.append(task_table.loc[(task_table["method"] == method) & (task_table["hp"] == hp_optimal)], ignore_index=True)
        
#         analysis["plot_table"] = plot_table
#         pickle.dump(analysis, open("analysis.p", "wb"))
#         print "[INFO] dump to analysis.p!"
        
#         # plot results
#         plot_auroc = sns.boxplot(x="method", y="auroc", data=plot_table).get_figure()
#         plot_auroc.savefig("auroc.pdf")
#         print "[INFO] auroc figure saved to auroc.pdf!"
        
#         plot_auroc.clf()

#         plot_aupvr = sns.boxplot(x="method", y="aupvr", data=plot_table).get_figure()
#         plot_aupvr.savefig("aupvr.pdf")
#         print "[INFO] aupvr figure saved to aupvr.pdf!"

#     elif fold_assign_method == "LOCUS":

#         plot_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold", "avgrank"])

#         for i, gb_i in task_table.groupby(["method"]):
#             method = i
#             hp_optimal = 0
#             avgrank_optimal = float("inf")

#             for j, gb_j in gb_i.groupby(["hp"]):
#                 if gb_j["avgrank"].mean() < avgrank_optimal:
#                     hp_optimal = j
#                     avgrank_optimal = gb_j["avgrank"].mean()
            
#             analysis["hp_optimal"][i] = hp_table[i].ix[j, "hp"]
#             plot_table = plot_table.append(task_table.loc[(task_table["method"] == method) & (task_table["hp"] == hp_optimal)], ignore_index=True)

#         analysis["plot_table"] = plot_table
#         pickle.dump(analysis, open("analysis.p", "wb"))
#         print "[INFO] dump to analysis.p!"

#         # plot results
#         plot_avgrank = sns.boxplot(x="method", y="avgrank", data=plot_table).get_figure()
#         plot_avgrank.savefig("avgrank.pdf")
#         print "[INFO] avgrank figure saved to avgrank.pdf!"

#     else:
#         print "Invalid fold assign method!"
#         exit(0)





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

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)

        rank_test = pd.DataFrame(data=y_test_probs, columns=["probs"])
        rank_test["group_id"] = dataset.loc[X_test.index.values, "group_id"].values
        rank_test["label"] = y_test.values
        rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False, method="average")
        avgrank = rank_test.loc[rank_test["label"]==1]["rank"].mean()

        result = {"auroc": auroc, "aupvr": aupvr, "avgrank": avgrank, "task_no": task_no}

    else:

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
        
        result = {"auroc": auroc, "aupvr": aupvr, "task_no": task_no}

    print "[INFO] cerenkov17 done! task no: ", task_no
    return result






def gwava_rf(dataset, hyperparameters, fold, fold_assign_method, task_no):
    
    # NaN value check
    if pd.isnull(dataset).values.any() == True:
        print "[ERROR] NaN value detected in dataset!"
        exit(0)


    if fold_assign_method == "LOCUS":
        feat = dataset.drop(["label", "group_id"], axis=1)
    elif fold_assign_method == "SNP":
        feat = dataset.drop(["label"], axis=1)
    else:
        print "Invalid fold assign method!"
        exit(0)

    label = dataset["label"]

    train_index = fold["train_index"].values[0].tolist()
    test_index = fold["test_index"].values[0].tolist()

    X_train = feat.loc[train_index]
    y_train = label.loc[train_index]
    X_test = feat.loc[test_index]
    y_test = label.loc[test_index]

    clf_gwava_rf = sklearn.ensemble.RandomForestClassifier(**hyperparameters)
    clf_gwava_rf.fit(X_train, y_train)

    y_test_probs = clf_gwava_rf.predict_proba(X_test)[:, clf_gwava_rf.classes_ == 1]

    if fold_assign_method == "LOCUS":

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)

        rank_test = pd.DataFrame(data=y_test_probs, columns=["probs"])
        rank_test["group_id"] = dataset.loc[X_test.index.values, "group_id"].values
        rank_test["label"] = y_test.values
        rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False, method="average")
        avgrank = rank_test.loc[rank_test["label"] == 1]["rank"].mean()

        result = {"auroc": auroc, "aupvr": aupvr, "avgrank": avgrank, "task_no": task_no}

    else:
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)

        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)

        result = {"auroc": auroc, "aupvr": aupvr, "task_no": task_no}

    print "[INFO] gwava_rf done! task no: ", task_no

    return result




def gwava_xgb(dataset, hyperparameters, fold, fold_assign_method, task_no):
    
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

    clf_gwava_xgb = xgboost.XGBClassifier(**hyperparameters)
    clf_gwava_xgb.fit(X_train, y_train)

    y_test_probs = clf_gwava_xgb.predict_proba(X_test)[:, clf_gwava_xgb.classes_ == 1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs
    
    if fold_assign_method == "LOCUS":

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
        
        rank_test = pd.DataFrame(data=y_test_probs, columns=["probs"])
        rank_test["group_id"] = dataset.loc[X_test.index.values, "group_id"].values
        rank_test["label"] = y_test.values
        rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False, method="average")
        avgrank = rank_test.loc[rank_test["label"]==1]["rank"].mean()
        
        result = {"auroc": auroc, "aupvr": aupvr, "avgrank": avgrank, "task_no": task_no}
    
    else:

        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_test_probs)
        auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)

        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
        aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
        
        result = {"auroc": auroc, "aupvr": aupvr, "task_no": task_no}

    print "[INFO] gwava_xgb done! task no: ", task_no
    return result