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
## (3) scp "cerenkov_ml.R", "cerenkov_ml_base.R", and the various required ".Rdata" files into the instance
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

## ============================ define global parameters =============================

g_par <- list(
    num_folds_cross_validation = 10,   ## for fig. 3:    10
    num_cv_replications = 200,         ## for fig. 3:   128
    flag_create_fork_cluster = FALSE,   ## for fig. 3:  TRUE
    override_num_fork_processes = 50,  ## for EC2, set to 64; for my MBP, set to 8
    notify_by_text_msg = TRUE,
    show_progress_bar = TRUE,
    flag_locus_sampling = TRUE,        ## set to false if you want SNP-level sampling
    flag_xgb_importance = FALSE,       ## if you set this to TRUE, make sure you set num_cv_replications=1 and num_folds_cross_validation=1
    random_number_seed = 1337,
    nthreads_per_process = 1,
    flag_randomize_workplan_order = FALSE,
    flag_create_ec2_instances = FALSE  ## don't set this to true if you set "flag_create_fork_cluster" to true
    )

if (g_par$notify_by_text_msg) {
    g_par$aws_sns_topic_arn <- "arn:aws:sns:us-west-2:315280700912:ramseylab"
} else {
    g_par$aws_sns_topic_arn <- NULL
}

## ------only if you are using an EC2 distributed cluster -----
if (g_par$flag_create_ec2_instances) {
    g_ec2_par <- list(
        sleep_sec_cluster_start = 200, ## how many seconds to wait for a worker EC2 instance to start up, before we can ssh in
        sleep_sec_ip_addresses = 10,   ## how many seconds to wait for a secondary IP address on a worker node to become active, after it is associated with the network interface
        cluster_size_instances = 19,   ## this number is currently limited by default limits in EC2; to increase the limit, need
                                       ### to send a request their tech support, at this link:
### https://console.aws.amazon.com/support/home?region=us-west-2#/case/create?issueType=service-limit-increase&limitType=service-code-ec2-instances
        num_ip_addresses_per_instance = 2,
        ami_id = "ami-cae763aa",       ## this is the "CERENKOV_CLUSTER5" AMI
        instance_type = "c4.2xlarge",  ## "m4.2xlarge" ## need two worker processes to fully load a m4.2xlarge instance, it appears
        security_group_settings = "sg-15cdac6d",
        subnet = "subnet-370a5e40",
        username = "ubuntu",
        network_interface_device_name = "ens3",  ## NOTE:  this is probably ubuntu-specific
        subnet_cidr_suffix = 20  ## means it's a /20
        )
    }

## cannot do both forked cluster and EC2 distributed cluster at the same time
stopifnot( !g_par$flag_create_ec2_instances || !g_par$flag_create_fork_cluster)

source("cerenkov_ml_base.R")

## ============================== set random number seed =================================
print(sprintf("setting random number seed to: %d", g_par$random_number_seed))
set.seed(g_par$random_number_seed)

## ============================== load OSU feature data; check for problems with feature data =================================

print("loading OSU data")

load(file="features_OSU.Rdata")
g_snp_names <- rownames(g_feat_osu_df)
stopifnot(g_feature_matrix_is_OK(g_feat_osu_df))
g_label_vec <- as.integer(as.character(g_feat_osu_df$label))

## ============================== set up for global performance measures =================================

if (! require(PRROC, quietly=TRUE)) {
    stop("package PRROC is missing")
}

g_calculate_auroc <- g_make_performance_getter(roc.curve,
                                               g_interval_clipper,
                                               "auc")

g_calculate_aupvr <- g_make_performance_getter(pr.curve,
                                               g_interval_clipper,
                                               "auc.davis.goadrich")

## ============================== set up for locus sampling =================================

if (g_par$flag_locus_sampling) {
    print("loading SNP coordinates data")

    load("snp_coordinates.Rdata")  ## ==> g_snp_coords_df
    g_snp_locus_ids <- g_get_snp_locus_ids(g_snp_coords_df)
    rm(g_snp_coords_df)

    g_assign_cases_to_folds_by_locus <- g_make_assign_cases_to_folds_by_group(g_snp_locus_ids)

    unique_locus_ids <- unique(g_snp_locus_ids)
    
    g_locus_to_snp_ids_map_list <- setNames(lapply(unique_locus_ids, function(p_locus_id) {
        which(g_snp_locus_ids == p_locus_id)
    }), unique_locus_ids)

    g_calculate_avgrank <- g_make_calculate_avgrank_within_groups(g_snp_locus_ids,
                                                                  g_locus_to_snp_ids_map_list,
                                                                  g_rank_by_score_decreasing)

    g_get_perf_results <- g_make_get_perf_results(g_calculate_aupvr,
                                                  g_calculate_auroc,
                                                  g_calculate_avgrank)  
} else {

    g_get_perf_results <- g_make_get_perf_results(g_calculate_aupvr,
                                                  g_calculate_auroc)
}

g_assign_cases_to_folds <- ifelse( g_par$flag_locus_sampling,
                                   c(g_assign_cases_to_folds_by_locus),
                                   c(g_assign_cases_to_folds_by_case) )[[1]]


## ============================== load feature data; check for problems with feature data =================================

library(Matrix)

g_feat_osu_matrix_sparse <- sparse.model.matrix(label ~ .-1, data=g_feat_osu_df)

print("loading GWAVA data")

load(file="features_GWAVA.Rdata") ## creates an R object called "g_feat_gwava_df"
stopifnot(g_feature_matrix_is_OK(g_feat_gwava_df))

stopifnot(g_snp_names == rownames(g_feat_gwava_df))
stopifnot(g_feat_osu_df$label == g_feat_gwava_df$label)

g_feat_gwava_matrix_sparse <- sparse.model.matrix(label ~ .-1, data=g_feat_gwava_df)

print("loading deepsea data")

load(file="features_deepsea_published.Rdata")

stopifnot(g_feature_matrix_is_OK(g_feat_deepsea_published_df))
stopifnot(rownames(g_feat_deepsea_published_df) == g_snp_names)
g_feat_deepsea_published_matrix <- data.matrix(g_feat_deepsea_published_df[, setdiff(names(g_feat_deepsea_published_df), "label")])

print("loading RSVP data")
      
load(file="features_RSVP.Rdata")

stopifnot(g_feature_matrix_is_OK(g_feat_rsvp_df))

stopifnot(rownames(g_feat_rsvp_df) == g_snp_names)
stopifnot(g_feat_rsvp_df$label == g_feat_osu_df$label)

g_feat_rsvp_matrix <- data.matrix(g_feat_rsvp_df[, setdiff(names(g_feat_rsvp_df), "label")])
g_feat_rsvp_imputed_df <- g_impute_missing_data_average_df(g_feat_rsvp_df)
    
rownames(g_feat_rsvp_imputed_df) <- rownames(g_feat_rsvp_df)
stopifnot(g_feat_rsvp_df$label == g_feat_rsvp_imputed_df$label)

print("loading DANQ data")
load(file="features_danq.Rdata")
stopifnot(rownames(g_feat_danq_df) == g_snp_names)
stopifnot(g_feat_danq_df$label == g_label_vec)
stopifnot(g_feature_matrix_is_OK(g_feat_danq_df))

g_feat_danq_matrix <- data.matrix(g_feat_danq_df[, setdiff(names(g_feat_danq_df), "label")])

print("loading deltaSVM data")
load(file="features_deltaSVM.Rdata")
stopifnot(rownames(g_feat_deltaSVM_df) == g_snp_names)
stopifnot(g_feat_deltaSVM_df$label == g_label_vec)
g_feat_deltaSVM_matrix_sparse <- sparse.model.matrix(label ~ .-1, data=g_feat_deltaSVM_df)
stopifnot(g_feature_matrix_is_OK(g_feat_deltaSVM_df))
    
## ===== load scores matrices for passthrough classifiers ======

print("loading CADD scores")
load("scores_cadd.Rdata")
stopifnot(names(g_scores_vec_cadd) == g_snp_names)
stopifnot(! is.na(g_scores_vec_cadd))

print("loading fitCons scores")
load("scores_fitcons.Rdata")
stopifnot(names(g_scores_vec_fitcons) == g_snp_names)
stopifnot(! is.na(g_scores_vec_fitcons))

print("loading eigen scores")
load("scores_eigen.Rdata")
stopifnot(names(g_scores_vec_eigen) == g_snp_names)
stopifnot(! is.na(g_scores_vec_eigen))

print("loading DANN scores")
load("scores_dann.Rdata")
stopifnot(names(g_scores_vec_dann) == g_snp_names)
stopifnot(! is.na(g_scores_vec_dann))

## ===== OSU experimental stuff ======


## ============================== assemble final list of feature matrices  =================================


## build a list of the feature matrices that we will need
g_classifier_feature_matrices_list <- list(
   feat_OSU_sparsematrix=g_feat_osu_matrix_sparse,
   feat_GWAVA_ranger=g_feat_gwava_df[, which(names(g_feat_gwava_df) != "label")],
   feat_GWAVA_sparsematrix=g_feat_gwava_matrix_sparse,
   feat_RSVP_matrix=g_feat_rsvp_matrix,
   feat_RSVP_ranger=g_feat_rsvp_imputed_df[, which(names(g_feat_rsvp_imputed_df) != "label")],   
   feat_deepsea_matrix=g_feat_deepsea_published_matrix,
   feat_DANQ=g_feat_danq_matrix,
   feat_deltaSVM_matrix=g_feat_deltaSVM_matrix_sparse
)

## free up memory
rm(g_feat_osu_df)
rm(g_feat_osu_matrix_sparse)
rm(g_feat_rsvp_matrix)
rm(g_feat_rsvp_df)
rm(g_feat_rsvp_imputed_df)
rm(g_feat_deepsea_published_df)
rm(g_feat_deepsea_published_matrix)
rm(g_feat_gwava_df)
rm(g_feat_gwava_matrix_sparse)
rm(g_feat_danq_df)
rm(g_feat_danq_matrix)
rm(g_feat_deltaSVM_df)
rm(g_feat_deltaSVM_matrix_sparse)

## ============================== make feature reducer function list  =================================

g_feature_reducer_functions_list <- list(PLS=g_feature_reducer_pls)


## ============================== precompute class balance ratio  =================================

g_label_vec_table <- table(g_label_vec)
g_class_count_ratio_negative_to_positive <- setNames(g_label_vec_table["0"]/g_label_vec_table["1"], NULL)
g_class_count_frac_positive <- setNames(g_label_vec_table["1"]/(g_label_vec_table["0"] + g_label_vec_table["1"]), NULL)

print(sprintf("Class label ratio (negative/positive): %f", g_class_count_ratio_negative_to_positive))


## ============================== make closures for classifiers  =================================

g_classifier_function_ranger <- g_make_classifier_function_ranger(p_nthread=g_par$nthreads_per_process,
                                                                   p_get_perf_results=g_get_perf_results)

g_custom_objective_for_training <- g_make_custom_xgboost_objective(g_class_count_ratio_negative_to_positive)

## g_classifier_function_xgboost_custom <- g_make_classifier_function_xgboost(p_nthread=g_par$nthreads_per_process,
##                                                                            g_get_perf_results,
##                                                                            p_feature_importance_type=NULL,
##                                                                            p_objective_function=g_custom_objective_for_training,
##                                                                            p_case_group_ids=g_snp_locus_ids)

g_classifier_function_xgboost <- g_make_classifier_function_xgboost(p_nthread=g_par$nthreads_per_process,
                                                                    g_get_perf_results,
                                                                    p_feature_importance_type=NULL,
                                                                    p_objective_function="binary:logistic",
                                                                    p_case_group_ids=g_snp_locus_ids)

rm(g_snp_locus_ids)

## g_classifier_function_xgboost_importance <- g_make_classifier_function_xgboost(p_nthread=g_par$nthreads_per_process,
##                                                                                p_get_perf_results=g_get_perf_results,
##                                                                                p_feature_importance_type=TRUE)

g_classifier_passthrough_cadd_scores <- g_make_classifier_function_passthrough(g_scores_vec_cadd,
                                                                               p_get_perf_results=g_get_perf_results)

g_classifier_passthrough_fitCons_scores <- g_make_classifier_function_passthrough(g_scores_vec_fitcons,
                                                                                  p_get_perf_results=g_get_perf_results)

g_classifier_passthrough_eigen_scores <- g_make_classifier_function_passthrough(g_scores_vec_eigen,
                                                                                p_get_perf_results=g_get_perf_results)

g_classifier_passthrough_dann_scores <- g_make_classifier_function_passthrough(g_scores_vec_dann,
                                                                               p_get_perf_results=g_get_perf_results)

## free up memory
rm(g_scores_vec_cadd)
rm(g_scores_vec_fitcons)
rm(g_scores_vec_eigen)
rm(g_scores_vec_dann)

g_classifier_functions_list <- list(
    XGB=g_classifier_function_xgboost,
    ##XGB_custom=g_classifier_function_xgboost_custom,
    ranger=g_classifier_function_ranger,
    ## XGB_importance=g_classifier_function_xgboost_importance,
    CADD=g_classifier_passthrough_cadd_scores,
    fitCons=g_classifier_passthrough_fitCons_scores,
    eigen=g_classifier_passthrough_eigen_scores,
    DANN=g_classifier_passthrough_dann_scores
)

## ============================== make hyperparameter lists  =================================

## ------------------ xgboost hyperparameter lists --------------

## g_hyperparameter_grid_list_xgb <- g_make_hyperparameter_grid_list(list(eta=c(0.05, 0.1, 0.15, 0.2),
##                                                                        nrounds=c(10, 20, 30),
##                                                                        gamma=c(0, 0.1, 1, 10),
##                                                                        subsample=c(0.5, 0.75, 0.85),
##                                                                        colsample_bytree=c(0.5, 0.75, 0.85),
##                                                                        base_score=g_class_count_frac_positive,
##                                                                        scale_pos_weight=c(0.125, 1, 8),
##                                                                        max_depth=c(5, 6, 7, 8)))

g_hyperparameter_grid_list_xgb_best_aupvr <- g_make_hyperparameter_grid_list(list(eta=0.1,
                                                                                  nrounds=30,
                                                                                  gamma=10,
                                                                                  subsample=1,
                                                                                  colsample_bytree=0.85,
                                                                                  base_score=g_class_count_frac_positive,
                                                                                  scale_pos_weight=1,
                                                                                  max_depth=6))

g_hyperparameter_grid_list_xgb_best_avgrank <- g_make_hyperparameter_grid_list(list(eta=0.1,
                                                                                  nrounds=30,
                                                                                  gamma=100,
                                                                                  subsample=0.85,
                                                                                  colsample_bytree=0.85,
                                                                                  base_score=g_class_count_frac_positive,
                                                                                  scale_pos_weight=8,
                                                                                  max_depth=6))


g_hyperparameter_grid_list_xgb <- c(g_hyperparameter_grid_list_xgb_best_avgrank,
                                    g_hyperparameter_grid_list_xgb_best_aupvr)

g_hyperparameter_grid_list_xgb_DS <- g_make_hyperparameter_grid_list(list(eta=0.1,
                                                                          nrounds=10,
                                                                          alpha=20,
                                                                          lambda=2000,
                                                                          scale_pos_weight=8))

g_workplan_list_xgb_OSU <- lapply(g_hyperparameter_grid_list_xgb,
                                     function(p_hyp) {
                                         list(classifier_feature_matrix_name="feat_OSU_sparsematrix",
                                              classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                              classifier_hyperparameter_set_type_name="XGB",
                                              workplan_set_name="OSU_XGB",
                                              classifier_hyperparameter_list=p_hyp)
                                     })

## g_workplan_list_xgbcustom_OSU <- lapply(g_hyperparameter_grid_list_xgb,
##                                      function(p_hyp) {
##                                          list(classifier_feature_matrix_name="feat_OSU_sparsematrix",
##                                               classifier_function_name="XGB_custom",
##                                               classifier_hyperparameter_set_type_name="XGB",
##                                               workplan_set_name="OSU_XGB_custom",
##                                               classifier_hyperparameter_list=p_hyp)
##                                      })


g_workplan_list_xgb_deepsea <- lapply(g_hyperparameter_grid_list_xgb_DS,
                                     function(p_hyp) {
                                         list(classifier_feature_matrix_name="feat_deepsea_matrix",
                                              classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                              classifier_hyperparameter_set_type_name="XGBDS",
                                              workplan_set_name="Deepsea_XGB_published",
                                              classifier_hyperparameter_list=p_hyp)
                                     })

g_workplan_list_xgb_danq <- lapply(g_hyperparameter_grid_list_xgb_DS,
                                     function(p_hyp) {
                                         list(classifier_feature_matrix_name="feat_DANQ",
                                              classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                              classifier_hyperparameter_set_type_name="XGBdanq",
                                              workplan_set_name="DANQ_XGB",
                                              classifier_hyperparameter_list=p_hyp)
                                     })

g_workplan_list_xgb_GWAVA <- lapply(g_hyperparameter_grid_list_xgb,
                                     function(p_hyp) {
                                         list(classifier_feature_matrix_name="feat_GWAVA_sparsematrix",
                                              classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                              classifier_hyperparameter_set_type_name="XGB",
                                              workplan_set_name="GWAVA_XGB",
                                              classifier_hyperparameter_list=p_hyp)
                                     })

g_workplan_list_xgb_RSVP <- lapply(g_hyperparameter_grid_list_xgb,
                                   function(p_hyp) {
                                       list(classifier_feature_matrix_name="feat_RSVP_matrix",
                                            classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                            classifier_hyperparameter_set_type_name="XGB",
                                            workplan_set_name="RSVP_XGB",
                                            classifier_hyperparameter_list=p_hyp)
                                   })

g_workplan_list_xgb_deltaSVM <- lapply(g_hyperparameter_grid_list_xgb,
                                   function(p_hyp) {
                                       list(classifier_feature_matrix_name="feat_deltaSVM_matrix",
                                            classifier_function_name=ifelse(g_par$flag_xgb_importance, "XGB_importance", "XGB"),
                                            classifier_hyperparameter_set_type_name="XGB",
                                            workplan_set_name="deltaSVM_XGB",
                                            classifier_hyperparameter_list=p_hyp)
                                   })

g_workplan_list_fitCons <- list(list(classifier_feature_matrix_name=NULL,
                                  classifier_function_name="fitCons",
                                  classifier_hyperparameter_set_type_name="passthrough",
                                  workplan_set_name="fitCons",
                                  classifier_hyperparameter_list=NULL))

g_workplan_list_CADD <- list(list(classifier_feature_matrix_name=NULL,
                                  classifier_function_name="CADD",
                                  classifier_hyperparameter_set_type_name="passthrough",
                                  workplan_set_name="CADD",
                                  classifier_hyperparameter_list=NULL))

g_workplan_list_eigen <- list(list(classifier_feature_matrix_name=NULL,
                                  classifier_function_name="eigen",
                                  classifier_hyperparameter_set_type_name="passthrough",
                                  workplan_set_name="eigen",
                                  classifier_hyperparameter_list=NULL))

g_workplan_list_DANN <- list(list(classifier_feature_matrix_name=NULL,
                                  classifier_function_name="DANN",
                                  classifier_hyperparameter_set_type_name="passthrough",
                                  workplan_set_name="DANN",
                                  classifier_hyperparameter_list=NULL))


## ------------------ ranger hyperparameter lists --------------

## WARNING:  do not set "probability=TRUE" for ranger; memory leak badness will result

### ==================== DO NOT DELETE THIS CODE; KEEP BECAUSE YOU WILL NEED IT LATER ===================
## g_hyperparameter_grid_list_ranger <- g_make_hyperparameter_grid_list(list(mtry=c(15, 20),
##                                                                           num.trees=c(200, 300),
##                                                                           probability=FALSE,
##                                                                           weight_positive_class=1,  #c(g_class_count_ratio_negative_to_positive, 1),
##                                                                           replace=TRUE,
##                                                                           sample.fraction=1))
### ==================== DO NOT DELETE THIS CODE; KEEP BECAUSE YOU WILL NEED IT LATER ===================


g_workplan_list_RSVP_published <- list(list(classifier_feature_matrix_name="feat_RSVP_ranger",
                                            classifier_function_name="ranger",
                                            classifier_hyperparameter_set_type_name="ranger",
                                            workplan_set_name="RSVP_RF_published",
                                            classifier_hyperparameter_list=list(probability=FALSE,
                                                                                mtry=47,
                                                                                num.trees=100,  
                                                                                weight_positive_class=8,
                                                                                replace=TRUE,
                                                                                sample.fraction=1)))


g_workplan_list_gwava_published <- list(list(classifier_feature_matrix_name="feat_GWAVA_ranger",
                                             classifier_function_name="ranger",
                                             classifier_hyperparameter_set_type_name="ranger",
                                             workplan_set_name="GWAVA_RF_published",
                                             classifier_hyperparameter_list=list(probability=FALSE,
                                                                                 mtry=14,
                                                                                 num.trees=100,  
                                                                                 weight_positive_class=1,
                                                                                 replace=TRUE,
                                                                                 sample.fraction=1)))

## ============================== assemble workplan list  =================================

## TODO: make a "g_check_workplan_list" function that checks for incorrect classifier function
## names, incorrect feature matrix names, etc.

g_workplan_list <- c(
    g_workplan_list_xgb_OSU,
#    g_workplan_list_xgbcustom_OSU
    g_workplan_list_xgb_deepsea,
    g_workplan_list_xgb_GWAVA,
    g_workplan_list_xgb_danq,
    g_workplan_list_xgb_RSVP,
    g_workplan_list_xgb_deltaSVM,
    g_workplan_list_gwava_published,
    g_workplan_list_RSVP_published,
    g_workplan_list_CADD,
    g_workplan_list_fitCons,
    g_workplan_list_eigen,
    g_workplan_list_DANN
)

g_workplan_list <- g_workplan_list[order(sapply(g_workplan_list, "[[", "classifier_hyperparameter_set_type_name"),
                                         sapply(g_workplan_list, "[[", "workplan_set_name"))]

names(g_workplan_list) <- 1:length(g_workplan_list)

print(sprintf("Number of workplans to process:  %d", length(g_workplan_list)))

## ============================ create the parallel cluster (fork or socket/ec2) =============================

library(parallel)

library(pbapply)
pboptions(type="txt")  ## force pblapply to display progress bar, even when using Rscript

if (g_par$flag_create_ec2_instances) {
    library(aws.ec2)

    ## here is where we create the EC2 instances that we will use
    g_ec2_instances <- do.call(c, lapply(1:g_ec2_par$cluster_size_instances, function(instance_number) {
        g_run_ec2_instance(g_ec2_par$ami_id,
                           g_ec2_par$instance_type,
                           g_ec2_par$subnet,   
                           g_ec2_par$security_group_settings,
                           g_ec2_par$num_ip_addresses_per_instance)
    }))

    print(sprintf("waiting for %d seconds for cluster to start", g_ec2_par$sleep_sec_cluster_start))
    pbsapply(1:g_ec2_par$sleep_sec_cluster_start, function(x) { Sys.sleep(1) })

    g_ip_addresses <- g_get_and_configure_ip_addresses_for_ec2_instances(g_ec2_instances,
                                                                         g_ec2_par$username,
                                                                         g_ec2_par$subnet_cidr_suffix,
                                                                         g_ec2_par$network_interface_device_name)
        
    print("IP addresses for the cluster: ")
    print(g_ip_addresses)

    ## if we have configured secondary private IP addresses for the instances, it seems sensible to wait a few seconds for the IP addresses to become active
    if (g_ec2_par$num_ip_addresses_per_instance  > 1) {
        print(sprintf("waiting for %d seconds for secondary IP addresses to become active", g_ec2_par$sleep_sec_ip_addresses))
        pbsapply(1:g_ec2_par$sleep_sec_ip_addresses, function(x) { Sys.sleep(1) })
    }

    print(sprintf("creating the PSOCK cluster"))

    g_cluster <- makeCluster(g_ip_addresses,
                             type="SOCK",
                             outfile="/dev/null",
                             rshcmd="ssh -oStrictHostKeyChecking=no",
                             useXDR=TRUE,
                             methods=FALSE)
} else {
    if (g_par$flag_create_fork_cluster) {
        ## our rule of thumb is to assign one process to each logical core, to keep things simple
        g_num_cores_use <- detectCores(logical=TRUE)
        print(sprintf("Number of cores detected: %d", g_num_cores_use))
        if (! is.null(g_par$override_num_fork_processes)) {
            g_num_cores_use <- g_par$override_num_fork_processes
        }
        g_cluster <- makeForkCluster(g_num_cores_use,
                                     outfile="")
    }
    else {
        g_cluster <- NULL
    }

    g_ec2_instances <- NULL
}
    
## randomize order of g_workplan_list, for load-balancing purposes

if (g_par$flag_randomize_workplan_order) {
    g_order_workplan_ids <- sample(length(g_workplan_list))
} else {
    g_order_workplan_ids <- 1:length(g_workplan_list)
}

## ============================ export to cluster =============================

if (! is.null(g_cluster)) {
    print(sprintf("Setting cluster workers to use random number streams with seed %d", g_par$random_number_seed))
    clusterSetRNGStream(g_cluster, g_par$random_number_seed)

    ## use this function for load-balancing parallelism (BUT DO NOT USE FOR DEBUGGING)
    g_func_lapply_cluster_LB <- function(p_X, p_FUNC) {
        parLapplyLB(g_cluster, p_X, p_FUNC)
    }

    ## use this for testing/debugging; exactly reproducible results
    if (g_par$show_progress_bar) {
        g_func_lapply_cluster <- function(p_X, p_FUNC) {
            pblapply(p_X, p_FUNC, cl=g_cluster)
        }
    } else {
        g_func_lapply_cluster <- function(p_X, p_FUNC) {
            parLapply(g_cluster, p_X, p_FUNC)
        }
    }

    clusterExport(cl=g_cluster, varlist=ls())

} else {
    g_func_lapply_cluster <- lapply
}

g_do_cluster_cleanup <- g_make_cluster_cleanup_function(g_cluster, g_ec2_instances)

## set up notifications, via stdout or SMS
g_send_message_notification <- g_make_message_notifier_function(g_par$aws_sns_topic_arn)


## bundle up all the data structures in a simple "runner" function
g_classifier_runner_func <- function() {
    g_run_mult_classifs_mult_hyperparams_cv(g_workplan_list[g_order_workplan_ids],
                                            g_classifier_functions_list,
                                            g_classifier_feature_matrices_list,
                                            g_label_vec,
                                            g_par$num_cv_replications,
                                            g_par$num_folds_cross_validation,
                                            g_func_lapply_cluster,
                                            p_func_lapply_second_level=lapply,
                                            g_feature_reducer_functions_list,
                                            g_assign_cases_to_folds)
}


## wrap error handling around our "runner" function
g_classifier_runner_func_err_handling <- g_make_classifier_runner_func_err_handling(g_classifier_runner_func,
                                                                                    g_send_message_notification,
                                                                                    g_cluster)


## ============================ run the classifier and gather results =============================

        
print(sprintf("Starting ML at time: %s", Sys.time()))

g_ml_results <- g_classifier_runner_func_err_handling()

## save the results to a file
if (! is.null(g_ml_results)) {
    save("g_par",
         "g_workplan_list",
         "g_ml_results",
         "g_order_workplan_ids",
         file="cerenkov_ml.Rdata")
}

g_send_message_notification(sprintf("Finished ML at time: %s", Sys.time()))

g_do_cluster_cleanup()
