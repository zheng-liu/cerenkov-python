## cerenkov_ml.R -- master script for tuning, training, and testing classifiers using R/parallel;
##                  it can run on a single multicore machine or using a PSOCKcluster of EC2 instances
##
## Author:  Stephen Ramsey
## 
## Packages required by this script:
##   PRROC
##
## Packages conditionally required by this script:
##   xgboost, ranger, dismo, Matrix, aws.ec2, pbapply
##
##   Note:  do not use this program with Ranger version 0.6.6 (stability issues); use Ranger 0.6.0 only
##
##   If you want to use partial least squares classification for feature reduction, you also need:
##   pls, methods

## How to run this script in EC2 using a single m4.16xlarge instance:
## (0) run a "m4.16xlarge" EC2 instance with the "CERENKOV_CLUSTER8" AMI
## (1) set "g_par$flag_create_ec2_instances = FALSE" below and "g_par$notify_by_text_msg = TRUE"
## (2) set "g_par$nthreads_per_process = 1"
## (3) scp "cerenkov_ml.R", "cerenkov_ml_base_functions.R", and the various required ".Rdata" files into the instance
## (4) ssh into the instance
## (5) bash$ nohup Rscript cerenkov_ml.R &
## (6) bash$ tail -f nohup.out

## How to run this script in EC2 using multiple worker nodes:
## (0) make sure you do not have any EC2 instances running (this script will use up your quota of 20 instances)
## (1) launch a m4.2xlarge instance using the "CERENKOV_CLUSTER8" AMI
## (2) assign the instance to security group PSOCKcluster
## (3) assign the instance to subnet CERENKOV_CLUSTER
## (4) optionally enable termination protection
## (5) under "add tags", name the instance "HEAD NODE"
## (6) go to step (3) in the "How to run this script in EC2 using a single m4.16xlarge instance" section above (where "instance" means the head node instance), and continue through step (6)

g_args <- commandArgs(trailingOnly=TRUE)

## ============================ define global parameters =============================

g_par <- list(
    num_folds_cross_validation = 5,     ## for fig. 3:    10
    num_cv_replications = 10 ,          ## for fig. 3:   128
    flag_create_fork_cluster = TRUE,    ## for fig. 3:  TRUE
    override_num_fork_processes = 64,   ## for EC2, set to 64; for my MBP, set to 8
    notify_by_text_msg = TRUE,
    show_progress_bar = FALSE,
    flag_locus_sampling = TRUE,         ## set to false if you want SNP-level sampling
    flag_xgb_importance = FALSE,        ## if you set this to TRUE, make sure you set num_cv_replications=1 and num_folds_cross_validation=1
    random_number_seed = if (is.na(g_args[1])) 1337 else as.integer(g_args[1]),
    nthreads_per_process = 1,
    flag_randomize_classifier_order = FALSE,
    flag_create_ec2_instances = FALSE,  ## don't set this to true if you set "flag_create_fork_cluster" to true
    analysis_label = "gwava_xgboost_tune",
    output_file_base_name = "cerenkov_ml_results",
    parallel_use_load_balancing = TRUE,
    debug_file_parallel=""              ## set to empty string for production run (makes error text go to stdout)
    )

g_par$aws_sns_topic_arn <- if (g_par$notify_by_text_msg) { "arn:aws:sns:us-west-2:315280700912:ramseylab" } else { NULL }

source("cerenkov_aws_functions.R")      ## load functions used for AWS
source("cerenkov_ml_base_functions.R")  ## load functions used for machine-learning

## ============================== load OSU feature data; check for problems with feature data =================================
print("loading OSU data")
load(file="features_cerenkov2_osu18.Rdata")
stopifnot(g_feature_matrix_is_OK(g_feat_cerenkov2_df))
library(Matrix)
g_feat_cerenkov2_matrix_sparse <- sparse.model.matrix(label ~ .-1, data=g_feat_cerenkov2_df)
g_snp_names <- rownames(g_feat_cerenkov2_df)
g_label_vec <- as.integer(as.character(g_feat_cerenkov2_df$label))

## ============================== load feature data; check for problems with feature data =================================

print("loading GWAVA data")

load(file="features_gwava_osu18.Rdata") ## creates an R object called "g_feat_gwava_df"
stopifnot(g_feature_matrix_is_OK(g_feat_gwava_df))
stopifnot(g_snp_names == rownames(g_feat_gwava_df))
stopifnot(g_feat_cerenkov2_df$label == g_feat_gwava_df$label)
g_feat_gwava_matrix_sparse <- sparse.model.matrix(label ~ .-1, data=g_feat_gwava_df)

## build a list of the feature matrices that we will need
g_classifier_feature_matrices_list <- list(
#x   feat_cerenkov2_sparsematrix=g_feat_cerenkov2_matrix_sparse,
   feat_GWAVA_sparsematrix=g_feat_gwava_matrix_sparse
)


## ============================== run invariant setup code =================================
source("cerenkov_ml_run_setup.R")

## ============================== make closures for classifiers  =================================

g_classifier_function_xgboost <- g_make_classifier_function_xgboost(p_nthread=g_par$nthreads_per_process,
                                                                    g_get_perf_results,
                                                                    p_feature_importance_type=NULL,
                                                                    p_make_objective_function=function(...){"binary:logistic"},
                                                                    p_case_group_ids=g_snp_locus_ids)

g_classifier_functions_list <- list(
    XGB=g_classifier_function_xgboost
)

## ============================== assemble final list of feature matrices  =================================

## free up memory
rm(g_feat_cerenkov2_df)
rm(g_feat_cerenkov2_matrix_sparse)
rm(g_feat_gwava_df)
rm(g_feat_gwava_matrix_sparse)


## ============================== make hyperparameter lists  =================================

## ------------------ xgboost hyperparameter lists --------------

g_hyperparameter_grid_list_xgb <- g_make_hyperparameter_grid_list(list(eta=c(0.05, 0.1, 0.15, 0.2),
                                                                       nrounds=c(10, 20, 30, 40),
                                                                       gamma=c(5, 10, 100, 200),
                                                                       lambda=c(0.1, 1.0, 5.0, 10.0),
                                                                       subsample=c(0.75, 0.85, 1),
#                                                                       colsample_bytree=c(0.75, 0.85, 1.0),
                                                                       base_score=g_class_count_frac_positive,
                                                                       scale_pos_weight=c(1.0, 14.45),
                                                                       max_depth=c(5, 6, 7)))


g_classifier_list_xgb_OSU <- lapply(g_hyperparameter_grid_list_xgb,
                                     function(p_hyp) {
                                         list(classifier_feature_matrix_name="feat_cerenkov2_sparsematrix",
                                              classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                              classifier_hyperparameter_set_type_name="XGB",
                                              classifier_set_name="OSU_XGB",
                                              classifier_hyperparameter_list=p_hyp)
                                     })

## g_classifier_list_xgbcustom_OSU <- lapply(g_hyperparameter_grid_list_xgb,
##                                      function(p_hyp) {
##                                          list(classifier_feature_matrix_name="feat_cerenkov2_sparsematrix",
##                                               classifier_function_name="XGB_custom",
##                                               classifier_hyperparameter_set_type_name="XGB",
##                                               classifier_set_name="OSU_XGB_custom",
##                                               classifier_hyperparameter_list=p_hyp)
##                                      })


g_classifier_list_xgb_GWAVA <- lapply(g_hyperparameter_grid_list_xgb,
                                     function(p_hyp) {
                                         list(classifier_feature_matrix_name="feat_GWAVA_sparsematrix",
                                              classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                              classifier_hyperparameter_set_type_name="XGB",
                                              classifier_set_name="GWAVA_XGB",
                                              classifier_hyperparameter_list=p_hyp)
                                     })



## ------------------ ranger hyperparameter lists --------------

## WARNING:  do not set "probability=TRUE" for ranger; memory leak badness will result

### ==================== DO NOT DELETE THIS CODE; KEEP BECAUSE YOU WILL NEED IT LATER ===================
## g_hyperparameter_grid_list_ranger <- g_make_hyperparameter_grid_list(list(mtry=c(15, 20),
##                                                                           num.trees=c(200, 300),
##                                                                           probability=FALSE,  ## do not set "probability=TRUE"
##                                                                           weight_positive_class=1,  #c(g_class_count_ratio_negative_to_positive, 1),
##                                                                           replace=TRUE,
##                                                                           sample.fraction=1))
### ==================== DO NOT DELETE THIS CODE; KEEP BECAUSE YOU WILL NEED IT LATER ===================


## ============================== assemble classifier list  =================================

## TODO: make a "g_check_classifier_list" function that checks for incorrect classifier function
## names, incorrect feature matrix names, etc.

g_classifier_list <- c(
#    g_classifier_list_xgb_OSU,
    g_classifier_list_xgb_GWAVA
)

## ============================== run invariant machine-learning code =================================

source("cerenkov_ml_run_ml.R")
