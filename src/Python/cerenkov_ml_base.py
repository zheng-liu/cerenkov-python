# Define all the methods in cerenkov_ml_base.py

def run_mult_classifs_mult_hyperparams_cv(workplan_list, 
                                          classifier_functions_list, 
                                          classifier_feature_matrices_list, 
                                          case_label_vec,
                                          num_cv_replications, 
                                          num_folds, 
                                          func_lapply_first_level, 
                                          func_lapply_second_level, 
                                          feature_reducer_functions_list, 
                                          assign_case_to_folds
                                          )
{
    # run multiple classifiers under multiple hyperparameter sets in multiple replication cv folds
    # //TODO write iterations of "func_lapply_first_level" and "func_lapply_second_level"
    
}