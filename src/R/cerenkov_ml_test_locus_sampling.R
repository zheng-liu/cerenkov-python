## cerenkov_ml.R -- master script for tuning, training, and testing classifiers using R/parallel;
##                  it can run on a single multicore machine or using a PSOCKcluster of EC2 instances
##
## Author:  Stephen Ramsey
## 
## Packages required by this script:
##   PRROC
##
## Packages conditionally required by this script:
##   xgboost, ranger, dismo, Matrix, aws.ec2, pbapply, Rcpp
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

## ============================ define global parameters =============================

g_par <- list(
    num_folds_cross_validation = 5,    ## for fig. 3:    10
    num_cv_replications = 20,           ## for fig. 3:   128
    flag_create_fork_cluster = FALSE,   ## for fig. 3:  TRUE
    flag_cluster_load_balancing = TRUE, 
    override_num_fork_processes = 8,   ## for EC2, set to 64; for my MBP, set to 8
    notify_by_text_msg = TRUE,
    show_progress_bar = TRUE,
    flag_locus_sampling = TRUE,        ## set to false if you want SNP-level sampling
    flag_xgb_importance = FALSE,       ## if you set this to TRUE, make sure you set num_cv_replications=1 and num_folds_cross_validation=1
    random_number_seed = 1337,
    nthreads_per_process = 1,
    flag_randomize_classifier_order = TRUE,
    flag_create_ec2_instances = FALSE  ## don't set this to true if you set "flag_create_fork_cluster" to true
    )

if (g_par$notify_by_text_msg) {
    g_par$aws_sns_topic_arn <- "arn:aws:sns:us-west-2:315280700912:ramseylab"
} else {
    g_par$aws_sns_topic_arn <- NULL
}



source("cerenkov_ml_base_functions.R")

## ============================== set random number seed =================================
print(sprintf("setting random number seed to: %d", g_par$random_number_seed))
set.seed(g_par$random_number_seed)

## ============================== load OSU feature data; check for problems with feature data =================================

print("loading OSU data")

load(file="features_cerenkov2_osu18.Rdata")
g_snp_names <- rownames(g_feat_cerenkov2_df)
stopifnot(g_feature_matrix_is_OK(g_feat_cerenkov2_df))
g_label_vec <- as.integer(as.character(g_feat_cerenkov2_df$label))

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

    load("snp_coordinates_osu18.Rdata")  ## ==> g_snp_coords_df
    g_snp_locus_ids <- g_get_snp_locus_ids(g_snp_coords_df)
    rm(g_snp_coords_df)

    g_assign_cases_to_folds_by_locus <- g_make_assign_cases_to_folds_by_group(g_snp_locus_ids)

    g_unique_locus_ids <- unique(g_snp_locus_ids)
    
    g_locus_to_snp_ids_map_list <- setNames(lapply(g_unique_locus_ids, function(p_locus_id) {
        which(g_snp_locus_ids == p_locus_id)
    }), g_unique_locus_ids)

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

print(system.time(replicate(200, g_assign_cases_to_folds_by_locus(5, g_label_vec))))


