# Define all the methods in cerenkov_ml_base.py
# Call method list using "getattr"

<<<<<<< HEAD
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


def run_mult_classifs_mult_hyperparams_cv(workplan_list, classifier_functions_list, classifier_feature_matrices_list,
                                          case_label_vec, num_cv_replications, num_folds, feature_reducer_functions_list, assign_case_to_folds):
    ## run multiple classifiers under multiple hyperparameter sets in multiple replication cv folds
    
    ## //TODO write iterations of "func_lapply_first_level" and "func_lapply_second_level"

    ## //TODO sort the classifier hyperparameter sets
    ''' classifier_hyperparameter_type_names_unique <- sort(unique(unlist(lapply(p_workplan_list, function(p_workplan) { p_workplan$classifier_hyperparameter_set_type_name })))) '''
    
    ## we need to know how many cases there are, in order to assign the cases to cross-validation folds
    ## if some feature matrix share different number of entries, there should be alert
    num_case = [len(feature_matrix) for feature_matrix in classifier_feature_matrices_list]
    if len(np.unique(num_case)) > 1: # check if all feature matrices share same number of cases
        print "------------ ALERT: number of cases in each classifier should be equal! ------------\n"
        exit(2)


# check feature matrix contains NaN or not.
# note that XGBoost can tolerate NaN while RF cannot.
def feature_check(feature_data):
    pass


def cerenkov_ml(workplan_list, feature_matrix_list, case_label_vec, number_cv_replications, num_folds, case_fold_assign_method):
    pass
=======
''' get_avgrank: calculate the average rank of regulatory SNP in its cluster
get_avgrank()


'''



''' cerenkov_ml: machine learning main function in CERENKOV project
cerenkov_ml(workplan_list, feature_matrix_list, case_label_vec, number_cv_replications, num_folds, case_fold_assign_method)
    workplan_list: a list of (classifier_name, classifier_function, hyperparameter_set_name, hyperparameter_set) tuple.
    feature_matrix_list: (feature_matrix_name, feature_matrix) tuple.
    case_label_vec: case labels for each entry in feature matrix.
    num_cv_replications: number of cross-validation replications.
    num_folds: number of folds for each of the cross-validation.
    case_fold_assign_method: the method of assigning each case to folds (locus sampling, SNP based sampling, etc).

return: ml_results (auroc, aupvr, avgrank)
'''

def cerenkov_ml(workplan_list, feature_matrix_list, case_label_vec, number_cv_replications, num_folds, case_fold_assign_method):
    pass

>>>>>>> 254e812653a1833e1a561568ed9fc7293eb8fcdf
