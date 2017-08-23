## cannot have both progress bar and load-balancing
stopifnot( !g_par$flag_create_fork_cluster || (!g_par$show_progress_bar || !g_par$parallel_use_load_balancing ) )

## cannot do both forked cluster and EC2 distributed cluster at the same time
stopifnot( !g_par$flag_create_ec2_instances || !g_par$flag_create_fork_cluster)

## ------only if you are using an EC2 distributed cluster -----
if (g_par$flag_create_ec2_instances) {
    g_ec2_par <- g_configure_ec2_instances()
}

## ============================== set random number seed =================================
print(sprintf("setting random number seed to: %d", g_par$random_number_seed))
set.seed(g_par$random_number_seed)

## ============================== set up for locus sampling =================================

if (g_par$flag_locus_sampling) {
    print("loading SNP coordinates data")

    load("snp_coordinates_osu18.Rdata")  ## ==> g_snp_coords_df
    g_snp_locus_ids <- g_get_snp_locus_ids(g_snp_coords_df)
    rm(g_snp_coords_df)

    g_assign_cases_to_folds_by_locus <- g_make_assign_cases_to_folds_by_group(g_snp_locus_ids)

    g_locus_to_snp_ids_map_list <- g_make_group_to_case_ids_map_list(g_snp_locus_ids)
}

g_calculate_avgrank <- if (g_par$flag_locus_sampling) {
    g_make_calculate_avgrank_within_groups(g_snp_locus_ids,
                                           g_locus_to_snp_ids_map_list,
                                           g_rank_by_score_decreasing)
} else {
    NULL
}

## ============================== set up for global performance measures =================================

if (! require(PRROC, quietly=TRUE)) { stop("package PRROC is missing") }

g_calculate_auroc <- g_make_performance_getter(roc.curve,
                                               g_interval_clipper,
                                               "auc")

g_calculate_aupvr <- g_make_performance_getter(pr.curve,
                                               g_interval_clipper,
                                               "auc.davis.goadrich")

g_get_perf_results <- g_make_get_perf_results(g_calculate_aupvr,
                                              g_calculate_auroc,
                                              g_calculate_avgrank)  

g_assign_cases_to_folds <- ifelse( g_par$flag_locus_sampling,
                                   c(g_assign_cases_to_folds_by_locus),
                                   c(g_assign_cases_to_folds_by_case) )[[1]]

## ============================== make feature reducer function list  =================================
g_feature_reducer_functions_list <- list(PLS=g_feature_reducer_pls)

## ============================== precompute class balance ratio  =================================

g_label_vec_table <- table(g_label_vec)
g_class_count_ratio_negative_to_positive <- setNames(g_label_vec_table["0"]/g_label_vec_table["1"], NULL)
g_class_count_frac_positive <- setNames(g_label_vec_table["1"]/(g_label_vec_table["0"] + g_label_vec_table["1"]), NULL)
print(sprintf("Class label ratio (negative/positive): %f", g_class_count_ratio_negative_to_positive))

