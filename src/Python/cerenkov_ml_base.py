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

def run_mult_classifs_mult_hyperparams_cv(workplan_list, 
                                          classifier_functions_list, 
                                          classifier_feature_matrices_list, 
                                          case_label_vec,
                                          num_cv_replications, 
                                          num_folds, 
                                          # func_lapply_first_level, 
                                          # func_lapply_second_level, 
                                          feature_reducer_functions_list, 
                                          assign_case_to_folds
                                          )
{
    # run multiple classifiers under multiple hyperparameter sets in multiple replication cv folds
    # //TODO write iterations of "func_lapply_first_level" and "func_lapply_second_level"
    
    # sort the classifier hyperparameter sets
    
    # 
}