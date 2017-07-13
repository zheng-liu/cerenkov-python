# Define all the methods in cerenkov_ml_base.py
# Call method list using "getattr"

## workplan_list: a list of "workplans". Each workplan is a list with elements:
##     workplan.classifier_feature_matrix_name: the integer ID of the feature matrix to use for this workplan
##     workplan.classifier_hyperparameter_list: the list of hyperparameters to use for this classifier
##     workplan.classifier_function_name: the function to be used for training the classifier
## classifier_functions_list: list of functions, each with signature: function(classifier_feature_matrix,
##                                                                             classifier_hyperparameter_list,
##                                                                             label_vector,
##                                                                             inds_cases_test) , returns list with four elements
##                                                                             (train_auroc, train_aupvr, test_auroc, test_aupvr)
## classifier_feature_matrices_list: one feature matrix for each type of feature matrix data strucutre to use **NO CASE LABELS**
## case_label_vec: numeric vector containing the feature labels (0 or 1 only)

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


# class cerenkov_result():
    
#     ''' Define the cerenkov operations

#         * initial
#         * 
#         * 
#     '''
#     def __init__(self):
#         self.result = []

#     def append(new_result):
#         # append ml feedback into result list
#         self.result.append(new_result)





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
        # SNP_chrom["group_id"][0] = 1.0 # fill in the missing first difference of ChromCoord
        SNP_chrom.ix[0,"group_id"] = 1.0 # fill in the missing first difference of ChromCoord
        SNP_chrom["group_id"] = SNP_chrom["group_id"] > cutoff_bp # True if distance > cutoff_bp; else False
        SNP_chrom["group_id"] = SNP_chrom["group_id"].astype(int) # transform to integer
        SNP_chrom["group_id"] = SNP_chrom["group_id"].cumsum(axis=0) # cumsum the "group_id" column
        SNP_chrom["group_id"] = SNP_chrom["group_id"].astype(str)
        SNP_chrom["group_id"] = chrom + "_" + SNP_chrom["group_id"] # add chrom prefix to group id
        # feat["group_id"][SNP_chrom.index] = SNP_chrom["group_id"] # assign values back to feature matrix
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
    fold_list = []

    # assign groups to folds
    for group in feat.groupby("group_id"):
        # check if there is at least 1 rSNP in each group
        if group[1]["label"].nonzero() is None:
            print "[ERROR INFO] There is no positive cases in group ", group[0]
            sys.exit()

    # assign each group
    group_case = [group for group in feat.groupby("group_id")]
    group_case.sort(key=lambda x: len(x[1]), reverse=True) # sort the group case according to number of elements each group

    for i_rep in range(n_rep):
        
        rs = RandomState(seed+i_rep)
        fold = pandas.DataFrame(data=[[0,"0-0"] for i in range(n_case)], columns=["fold_id", "group_id"])
        fold.index = dataset.index

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

            fold.ix[group[1].index, "fold_id"] = ind_selected + 1
            fold_case_num[ind_selected] += group_count
            fold_pos_num[ind_selected] += group_pos_count
        
        fold["group_id"] = feat["group_id"]

        # check if all SNP are assigned
        if 0 in fold["fold_id"]:
            print "[ERROR INFO] Some SNP is not assigned to any fold!"
            sys.exit()

        fold_list.append(fold)
        
        fold_case_num = [0] * n_fold
        fold_pos_num = [0] * n_fold

    del dataset["group_id"]
    return fold_list





def snp_sampling(dataset, n_rep, n_fold):
    '''
        * input: dataset (Pandas DataFrame with rsnp id as index)
        * output: assigned groups
    '''
    # //TODO think about whether we need to balance positive-negative case numbers in each fold, or totally random?
    fold_list = []
    n_label = len(dataset)
    
    fold0 = [(x%n_fold+1) for x in range(n_label)] # initial fold: 1,2,3,4...,n_label

    for _ in range(n_rep):
        np.random.shuffle(fold0)
        fold = pandas.DataFrame(data=fold0, columns=["fold_id"])
        fold.index = dataset.index
        fold_list.append(fold)
    
    return fold_list





def cerenkov17(dataset, hyperparameters, folds, case_fold_assign_method):
    
    ''' cerenkov17 takes in the (feature, label, hyperparameters, fold) 4-element tuple, and output the performance.
        
        * 
    '''

    feat = dataset.drop("label", axis=1)
    label = dataset["label"]
    K = len(set(folds["fold_id"])) # K-folds for each cv fold
    result_list = []

    for fold_id in range(1, K+1):
        test_index = folds[folds["fold_id"] == fold_id].index
        train_index = folds[folds["fold_id"] != fold_id].index

        X_test = feat.loc[test_index, :]
        y_test = label.loc[test_index].tolist() # //TODO we should guarantee that the y_test should have index as SNP IDs

        X_train = feat.loc[train_index, :]
        y_train = label.loc[train_index].tolist()

        clf_cerenkov17 = xgboost.XGBClassifier(**hyperparameters)
        clf_cerenkov17.fit(X_train, y_train)

        y_test_probs = clf_cerenkov17.predict_proba(X_test)[:, clf_cerenkov17.classes_==1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs

        if case_fold_assign_method == "LOCUS":
            rank_test = pandas.DataFrame(data=y_test_probs, columns=["probs"])
            rank_test["group_id"] = folds[folds["fold_id"] == fold_id]["group_id"].values
            rank_test["label"] = y_test
            rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False)
            avgrank = rank_test.loc[rank_test["label"]==1]["rank"].mean()
            result = {"avgrank": avgrank}
            result_list.append(result)
        else:
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,y_test_probs)
            auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)

            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
            aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
            
            result = {"fpr":fpr, "tpr":tpr, "auroc":auroc, "precision":precision, "recall":recall, "aupvr":aupvr}
            result_list.append(result)

    return result_list





def gwava_rf(dataset, hyperparameters, folds, case_fold_assign_method):
    ''' gwava_rf takes in the (feature, label, hyperparameters, fold) 4-element tuple, and output the performance.
        
        * 
    '''
    feat = dataset.drop("label", axis=1)
    label = dataset["label"]
    K = len(set(folds["fold_id"])) # K-folds for each cv fold
    result_list = []

    for fold_id in range(1, K+1):
        test_index = folds[folds["fold_id"] == fold_id].index
        train_index = folds[folds["fold_id"] != fold_id].index

        X_test = feat.loc[test_index, :]
        y_test = label.loc[test_index].tolist() # //TODO we should guarantee that the y_test should have index as SNP IDs

        X_train = feat.loc[train_index, :]
        y_train = label.loc[train_index].tolist()
        

        clf_gwava_rf = sklearn.ensemble.RandomForestClassifier(**hyperparameters)
        clf_gwava_rf.fit(X_train, y_train)

        y_test_probs = clf_gwava_rf.predict_proba(X_test)[:, clf_gwava_rf.classes_==1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs

        if case_fold_assign_method == "LOCUS":
            rank_test = pandas.DataFrame(data=y_test_probs, columns=["probs"])
            rank_test["group_id"] = folds[folds["fold_id"] == fold_id]["group_id"].values
            rank_test["label"] = y_test
            rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False)
            avgrank = rank_test.loc[rank_test["label"]==1]["rank"].mean()
            result = {"avgrank": avgrank}
            result_list.append(result)
        else:
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,y_test_probs)
            auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)

            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
            aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
            
            result = {"fpr":fpr, "tpr":tpr, "auroc":auroc, "precision":precision, "recall":recall, "aupvr":aupvr}
            result_list.append(result)

    return result_list




def gwava_xgb(dataset, hyperparameters, folds, case_fold_assign_method):
    ''' gwava_xgb takes in the (feature, label, hyperparameters, fold) 4-element tuple, and output the performance.
        
        * 
    '''

    feat = dataset.drop("label", axis=1)
    label = dataset["label"]
    K = len(set(folds["fold_id"])) # K-folds for each cv fold
    result_list = []

    for fold_id in range(1, K+1):
        test_index = folds[folds["fold_id"] == fold_id].index
        train_index = folds[folds["fold_id"] != fold_id].index

        X_test = feat.loc[test_index, :]
        y_test = label.loc[test_index].tolist() # //TODO we should guarantee that the y_test should have index as SNP IDs

        X_train = feat.loc[train_index, :]
        y_train = label.loc[train_index].tolist()

        clf_gwava_xgb = xgboost.XGBClassifier(**hyperparameters)
        clf_gwava_xgb.fit(X_train, y_train)

        y_test_probs = clf_gwava_xgb.predict_proba(X_test)[:, clf_gwava_xgb.classes_==1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs

        if case_fold_assign_method == "LOCUS":
            rank_test = pandas.DataFrame(data=y_test_probs, columns=["probs"])
            rank_test["group_id"] = folds[folds["fold_id"] == fold_id]["group_id"].values
            rank_test["label"] = y_test
            rank_test["rank"] = rank_test.groupby("group_id")["probs"].rank(ascending=False)
            avgrank = rank_test.loc[rank_test["label"]==1]["rank"].mean()
            result = {"avgrank": avgrank}
            result_list.append(result)
        else:
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,y_test_probs)
            auroc = sklearn.metrics.roc_auc_score(y_test, y_test_probs)

            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_test_probs)
            aupvr = sklearn.metrics.average_precision_score(y_test, y_test_probs)
            
            result = {"fpr":fpr, "tpr":tpr, "auroc":auroc, "precision":precision, "recall":recall, "aupvr":aupvr}
            result_list.append(result)

    return result_list





def cerenkov_ml(method_list, dataset_list, hyperparameter_list, \
                number_cv_replications, number_folds, case_assignment_method="SNP", \
                feature_reduced=False, ncpus=-1):
    # //TODO write all the checks
    # //TODO add feature dimension reduction procedure

    ''' check input

        * check if each method has a feature matrix
        * check if ncpus > 0
    '''

    ''' initializations:
        
        * build (method, feature_matrix, case_label) tuples
        * take number_cv_replications, num_folds
        * take case_assignment_method, select cv assignment approach
    '''
    
    # label = label_vec
    feature_list = [dataset_list[i].drop("label", axis=1) for i in range(len(dataset_list))]
    label = dataset_list[0]["label"]
    num_rep = number_cv_replications
    num_folds = number_folds
    num_method = len(method_list)
    num_feature = len(feature_list)

    # select sampling method
    if case_assignment_method == "LOCUS": # cross validation in locus sampling way
        sampling = locus_sampling
    elif case_assignment_method == "SNP": # cross validation in SNP centered sampling way
        sampling = snp_sampling
    else:
        print "[ERROR INFO] Currently only \"LOCUS\" and \"SNP\" assignments allowed!"
    
    # //TODO think about whether the feature_list[0] must be our method in which the location information are used
    fold_list = sampling(dataset_list[0], num_rep, num_folds)
    
    # //TODO write an if else to control the "feature_reduced" logic

    ''' machine learning in parallelization

        * start parallel python
        * assign folds using the corresponding sampling
        * plugin all hyperparameters
        * train, test models
        * performance results
    '''


    jobs = []
    result = []
    # init parallel python server
    ppservers = ()

    if ncpus == -1:
        job_server = pp.Server(ppservers=ppservers)
        print "Starting with ", job_server.get_ncpus(), "CPUs"
    else:
        job_server = pp.Server(ncpus, ppservers=ppservers)
        print "Starting with ", job_server.get_ncpus(), "CPUs"

    # submit jobs to server
    for method, dataset, hyperparameters, fold in zip(method_list, dataset_list, hyperparameter_list, fold_list):
        args = (dataset, hyperparameters, fold, "LOCUS")
        # job_server.submit(method, args, modules=("time","pandas","numpy","xgboost"), callback=cr.append)
        jobs.append(job_server.submit(method, args, modules=("time","pandas","numpy","xgboost", "sklearn.ensemble")))
        print "a job submitted"

    # # wait for jobs in all groups to finish
    # job_server.wait()

    for f in jobs:
        result.extend(f())

    # display result
    job_server.print_stats()

    return result