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

def locus_group(dataset, cutoff_bp):
    pass

def locus_sampling(dataset, n_rep, n_fold, cutoff_bp=50000, slopt_allowed=0.5, seed=1337):
    pass

def snp_sampling(dataset, n_rep, n_fold):
    pass

def fold_generate(dataset, fold_assign_method):
    pass

def hp_generate():
    # generate the hyperparameters for each classifier
    pass

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


    