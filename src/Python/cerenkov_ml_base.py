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

import pandas as pd
import numpy as np
import xgboost
import gc
import warnings


class cerenkov_result():
    
    ''' Define the cerenkov operations

        * initial
        * 
        * 
    '''
    def __init__(self):
        self.result = []

    def append(new_result):
        # append ml feedback into result list
        self.result.append(new_result)

def locus_sampling():
    '''
        * input: label
        * output: assigned groups
    '''
    pass

def snp_sampling():
    '''
        * input: label
        * output: assigned groups
    '''
    pass

def cerenkov17(feature, label, hyperparameters, folds, case_fold_assign_method):
    
    ''' cerenkov17 takes in the (feature, label, hyperparameters, fold) 4-element tuple, and output the performance.
        
        * 
    '''
    K = len(set(folds)) # K-folds for each cv fold

    for fold_id in range(1, K+1):
        test_index = folds[folds[fold_id] == fold_id].index
        train_index = folds[folds[fold_id] != fold_id] .index

        X_test = feature.loc[test_index, :]
        y_test = label.loc[test_index, :] # //TODO we should guarantee that the y_test should have index as SNP IDs

        X_train = feature.loc[train_index, :]
        y_train = label.loc[train_index, :]

        clf_cerenkov17 = xgb.XGBClassifier(**hyperparameters)
        clf_cerenkov17.fit(X_train, y_train)

        y_test_pred = clf_cerenkov17.predict_proba(X_test)[:, clf_cerenkov17.classes_==1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs
        
        if case_fold_assign_method == "LOCUS":
            avgrank = get_avgrank(y_test, y_test_pred, locus_table) 
        else:
            aupvr = get_aupvr(y_test, y_test_pred)



def cerenkov17_test(feature, label, hyperparameters, folds):

    ''' test the distributed machine learning of cerenkov
    '''
    start_time = time.time()
    K = len(set(folds["fold_id"])) # K-folds for each cv fold

    for fold_id in range(1, K+1):
        test_index = folds[folds["fold_id"] == fold_id].index
        train_index = folds[folds["fold_id"] != fold_id].index

        X_test = feature.loc[test_index, :]
        y_test = label.loc[test_index].tolist() # //TODO we should guarantee that the y_test should have index as SNP IDs

        X_train = feature.loc[train_index, :]
        y_train = label.loc[train_index].tolist()

        clf_cerenkov17 = xgboost.XGBClassifier(**hyperparameters)
        clf_cerenkov17.fit(X_train, y_train)

        y_test_pred = clf_cerenkov17.predict_proba(X_test)[:, clf_cerenkov17.classes_==1] # //TODO we should guarantee that the y_test_pred should have index as SNP IDs

    end_time = time.time()
    task_time = end_time - start_time
    
    return y_test_pred



def cerenkov_ml(method_list, feature_list, label_vec, hyperparameter_list, \
                number_cv_replications, number_folds, case_fold_assign_method="SNP", \
                feature_reduced=False, ncpus=-1):
    # //TODO write all the checks
    ''' check input

        * check if each method has a feature matrix
        * check if ncpus > 0
    '''

    

    ''' initializations:
        
        * build (method, feature_matrix, case_label) tuples
        * take number_cv_replications, num_folds
        * take case_fold_assign_method, select cv assignment approach
    '''

    label = label_vec
    num_rep = number_cv_replications
    num_folds = number_folds
    num_method = len(method_list)
    num_feature = len(feature_list)

    # select sampling method
    if case_fold_assign_method == "LOCUS": # cross validation in locus sampling way
        sampling = locus_sampling
    else: # cross validation in SNP centered sampling way
        sampling = snp_sampling
    
    # //TODO think about whether the feature_list[0] must be our method in which the location information are used
    fold_list = [sampling(feature_list[0], num_rep, num_folds) for i in range(num_method)] # assign folds for each method for num_rep*num_folds folds in total   
    
    # //TODO write an if else to control the "feature_reduced" logic

    ''' machine learning in parallelization

        * start parallel python
        * assign folds using the corresponding sampling
        * plugin all hyperparameters
        * train, test models
        * performance results
    '''


    # init parallel python server
    ppservers = ()

    if ncpus == -1:
        job_server = pp.Server(ppservers=ppservers)
        print "Starting with ", job_server.get_ncpus(), "CPUs"
    else:
        job_server = pp.Server(ncpus, ppservers=ppservers)
        print "Starting with ", job_server.get_ncpus(), "CPUs"


    # init result
    result = ml_result()


    # submit jobs to server
    for method, feature in zip(method_list, feature_list):
        for hyperparameters in hyperparameter_list:
            for fold in fold_list:
                args = (feature, hyperparameters, label, fold)
                job_server.submit(method, args)
    
    # wait for jobs in all groups to finish
    job_server.wait()
    
    # display result
    job_server.print_stats()

def cerenkov_ml_test(method_list, feature_list, label_vec, hyperparameter_list, fold_list, ncpus=-1):
    # //TODO write all the checks
    ''' check input

        * check if each method has a feature matrix
        * check if ncpus > 0
    '''
    


    ''' initializations:
        
        * build (method, feature_matrix, case_label) tuples
        * take number_cv_replications, num_folds
        * take case_fold_assign_method, select cv assignment approach
    '''

    label = label_vec

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
    for method, feature, hyperparameters, fold in zip(method_list, feature_list, hyperparameter_list, fold_list):
        args = (feature, label, hyperparameters, fold)
        # job_server.submit(method, args, modules=("time","pandas","numpy","xgboost"), callback=cr.append)
        jobs.append(job_server.submit(method, args, modules=("time","pandas","numpy","xgboost")))
        print "a job submitted"

    # # wait for jobs in all groups to finish
    # job_server.wait()

    for f in jobs:
        result.append(f())

    # display result
    job_server.print_stats()

    return result