# Define all the methods in cerenkov_ml_base.py
# Call method list using "getattr"

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

