## cerenkov_ml_base.R:  here is where I put all the functions that do not reference any global variables

## --------------- Cerenkov-specific functions start here ---------

## p_classifier_list:  a list of "classifiers".  Each classifier is a list with elements:
##    classifier$classifier_feature_matrix_name: the integer ID of the feature matrix to use for this classifier
##    classifier$classifier_hyperparameter_list:  the list of hyperparameters to use for this classifier
##    classifier$classifier_function_name:  the function to be used for training the classifier
## p_classifier_functions_list:  list of functions, each with signature: function(p_classifier_feature_matrix,
##                                                                                p_classifier_hyperparameter_list,
##                                                                                p_label_vector,
##                                                                                p_inds_cases_test), returns list with four elements
##                                                                                (train_auroc, train_aupvr, test_auroc, test_aupvr)
## p_classifier_feature_matrices_list:  one feature matrix for each type of feature matrix data structure to use **NO CASE LABELS**
## p_case_label_vec:  numeric vector containing the feature labels (0 or 1 only)
## p_func_lapply:  function(p_X, p_FUNC) returning a list

g_run_mult_classifs_mult_hyperparams_cv <- function(p_classifier_list,
                                                      p_classifier_functions_list,
                                                      p_classifier_feature_matrices_list,
                                                      p_case_label_vec,
                                                      p_num_cv_replications=1,
                                                      p_num_folds=10,
                                                      p_func_lapply=lapply,
                                                      p_feature_reducer_functions_list=NULL,
                                                      p_assign_cases_to_folds) {
    
    ## get list of unique classifier hyperparameter set type names
    classifier_hyperparameter_set_type_names_unique <- sort(unique(unlist(lapply(p_classifier_list,
                                                                             function(p_classifier) {
                                                                                 p_classifier$classifier_hyperparameter_set_type_name
                                                                             }))))

    ## check if there is at least one classifier on the classifier list
    stopifnot(length(p_classifier_list) > 0)

    ## we need to know how many cases there are, in order to assign the cases to cross-validation folds
    num_cases <- unique(sapply(p_classifier_feature_matrices_list, nrow))
    if (length(num_cases) > 1) {
        stop("all classifier feature matrices must have equal numbers of cases")
    }

    ## create a list of length equal to the number of replications; each list contains fold assignments for all SNPs
    replications_fold_assignments_list <- replicate(p_num_cv_replications,
                                                    p_assign_cases_to_folds(p_num_folds=p_num_folds,
                                                                            p_case_label_vec=p_case_label_vec),
                                                    simplify=FALSE)

    ## make a workplan list containing triples of classifier ID, replication ID, and CV fold ID
    work_list <- lapply(setNames(lapply(data.frame(t(expand.grid(1:length(p_classifier_list),
                                                                 1:p_num_cv_replications,
                                                                 1:p_num_folds))),
                                 setNames, c("classifier_id", "replication_id", "fold_id")),
                                 NULL), as.list)

    ml_global_results_list <- p_func_lapply(work_list,
                                            function(p_vec_inds_work) {
                                                classifier_id <- p_vec_inds_work$classifier_id
                                                fold_id <- p_vec_inds_work$fold_id
                                                replication_id <- p_vec_inds_work$replication_id
                                                
                                                fold_assignments <- replications_fold_assignments_list[[replication_id]]
                                                inds_cases_test <- which(fold_id == fold_assignments)
                                                if (length(inds_cases_test) == length(fold_assignments)) {
                                                    ## this means we have num_folds=1, i.e., no cross-validation; train on all cases, so set test cases to empty vector
                                                    inds_cases_test <- c()
                                                }

                                                classifier_list <- p_classifier_list[[classifier_id]]


                                                ## need to know the feature matrix name, so we can retrieve the feature matrix
                                                classifier_feature_matrix_name <- classifier_list$classifier_feature_matrix_name

                                                ## need the classifier's hyperparameter list
                                                classifier_hyperparameter_list <- classifier_list$classifier_hyperparameter_list

                                                ## need the classifier's hyperparameter set name, which dictates what top-level list slot the results go into
                                                classifier_hyperparameter_set_type_name <- classifier_list$classifier_hyperparameter_set_type_name

                                                ## need the classifier's function name so we can look up the classifier function
                                                classifier_function_name <- classifier_list$classifier_function_name

                                                ## need the classifier function so we can run the train/test cycle
                                                classifier_function <- p_classifier_functions_list[[classifier_function_name]]
                                                
                                                save_hyperparameter_list <- classifier_hyperparameter_list
                                                
                                                if (is.null(classifier_list$feature_reducer_function_name)) {
                                                    ## this is the standard case, the classifier doesn't call for using a feature matrix reducer
                                                    
                                                    if (! is.null(classifier_feature_matrix_name)) {
                                                        ## this is the standard case, we don't have a null feature matrix (means a passthrough classifier)
                                                        feature_matrix <- p_classifier_feature_matrices_list[[classifier_feature_matrix_name]]
                                                        
                                                        if (is.null(feature_matrix)) { stop(sprintf("feature matrix %s missing",
                                                                                                    classifier_feature_matrix_name)) }
                                                    } else {
                                                        ## in this case we are using a passthrough classifier (like CADD, Eigen, or fitCons)
                                                        feature_matrix <- NULL
                                                        classifier_feature_matrix_name <- "NA"
                                                    }
                                                    
                                                } else {
                                                    
                                                    ## we are using a "supervised" feature reducer function, like PLS
                                                    base_feature_matrix <- p_classifier_feature_matrices_list[[classifier_feature_matrix_name]]
                                                    if (is.null(base_feature_matrix)) { stop(sprintf("feature matrix %s missing",
                                                                                                     classifier_feature_matrix_name)) }
                                                    feature_reducer_function_name <- classifier_list$feature_reducer_function_name
                                                    feature_reducer_function <- p_feature_reducer_functions_list[[feature_reducer_function_name]]
                                                    stopifnot( ! is.null(feature_reducer_function))
                                                    feature_reducer_input_matrix_name <- classifier_list$feature_reducer_input_matrix_name
                                                    stopifnot( ! is.null(feature_reducer_input_matrix_name))
                                                    feature_reducer_input_matrix <- p_classifier_feature_matrices_list[[feature_reducer_input_matrix_name]]
                                                    if (is.null(feature_reducer_input_matrix)) { stop(sprintf("feature matrix %s missing",
                                                                                                              feature_reducer_input_matrix_name)) }
                                                    stopifnot( ! is.null(feature_reducer_input_matrix))
                                                    feature_reducer_hyperparameters_list <- classifier_list$feature_reducer_hyperparameters_list
                                                    stopifnot( ! is.null(feature_reducer_hyperparameters_list))
                                                    
                                                    ## call the feature reducer
                                                    reduced_feature_matrix <- do.call(feature_reducer_function,
                                                                                      c(list(p_input_feature_matrix=feature_reducer_input_matrix,
                                                                                             p_case_label_vec=p_case_label_vec,
                                                                                             p_inds_cases_test=inds_cases_test),
                                                                                        feature_reducer_hyperparameters_list))
                                                    
                                                    ## combine the reduced feature matrix with the base feature matrix
                                                    if ("sparseMatrix" %in% is(base_feature_matrix)) {
                                                        if (! require(Matrix, quietly=TRUE)) { stop("package Matrix is missing") }
                                                        feature_matrix <- cBind(base_feature_matrix, reduced_feature_matrix)
                                                    } else {
                                                        feature_matrix <- cbind(base_feature_matrix, reduced_feature_matrix)
                                                    }
                                                    
                                                    classifier_feature_matrix_name <- paste(classifier_feature_matrix_name,
                                                                                            feature_reducer_input_matrix_name, sep="_")

                                                    save_hyperparameter_list <- c(save_hyperparameter_list,
                                                                                  feature_reducer_hyperparameters_list)
                                                } 

                                                classifier_custom_objective_function_parameters_list <- classifier_list$classifier_custom_objective_function_parameters_list
                                                if (! is.null(classifier_custom_objective_function_parameters_list)) {
                                                    save_hyperparameter_list <- c(save_hyperparameter_list, classifier_custom_objective_function_parameters_list)
                                                }
                                                
                                                ## train/test the classifier for the specified hyperparameters
                                                classifier_run_time <- system.time(
                                                    classifier_ret_list <- classifier_function(p_classifier_feature_matrix=feature_matrix,
                                                                                               p_classifier_hyperparameter_list=classifier_hyperparameter_list,
                                                                                               p_label_vector=p_case_label_vec,
                                                                                               p_inds_cases_test=inds_cases_test,
                                                                                               p_custom_objective_function_parameters_list=classifier_custom_objective_function_parameters_list) )

                                                feat_import_scores <- classifier_ret_list$feat_import_scores
                                                classifier_ret_list$feat_import_scores <- NULL
                                                
                                                if (is.null(save_hyperparameter_list)) {
                                                    save_hyperparameter_list <- list(classifier_hyperparameters.="")
                                                }

                                                ## create a list of results
                                                ## WARNING:  DO **NOT** ALTER THE ORDER OF THESE LIST ELEMENTS, IT WILL BREAK BRITTLE DOWNSTREAM ANALYSIS CODE:
                                                list(performance_results=data.frame(c(classifier_ret_list,
                                                                                      list(classifier_name=classifier_function_name,
                                                                                           classifier_feature_matrix_name=classifier_feature_matrix_name,
                                                                                           classifier_hyperparameter_set_type_name=classifier_hyperparameter_set_type_name,
                                                                                           classifier_hyperparameters=save_hyperparameter_list,
                                                                                           classifier_run_time=setNames(classifier_run_time[1], NULL),
                                                                                           classifier_set_name=classifier_list$classifier_set_name,
                                                                                           classifier_id=classifier_id,
                                                                                           replication_id=replication_id,
                                                                                           cv_fold_id=fold_id)),
                                                                                    stringsAsFactors=FALSE),
                                                     feat_import_scores=feat_import_scores)
                                            })

    ## invert res_list so that the classifier ID is the top level, and the fold ID is the second level
    ml_global_results_list_wp_top <- lapply(1:length(p_classifier_list),
                                            function(p_classifier_list_id) {
                                                ml_global_results_list[which(sapply(ml_global_results_list,
                                                                                    function(p_list) {
                                                                                        p_list$performance_results$classifier_id
                                                                                    })==p_classifier_list_id)]
                                            })
    
    ## get the hyperparameter set type name for each classifier
    classifier_hyperparameter_type_names <- sapply(ml_global_results_list_wp_top,
            function(p_res_list_for_classifier) {
                p_res_list_for_classifier[[1]]$performance_results$classifier_hyperparameter_set_type_name })
    
    ## divide classifier list into performance results data frames, organized by hyperparameter-set type name
    ml_global_results_list_hpts_top <- setNames(lapply(classifier_hyperparameter_set_type_names_unique,
            function(p_classifier_hyperparameter_type_name) {
                hptds_list_list <- ml_global_results_list_wp_top[which(classifier_hyperparameter_type_names == p_classifier_hyperparameter_type_name)]
                do.call(rbind, lapply(hptds_list_list, function(hptds_list) {
                    do.call(rbind, lapply(hptds_list, "[[", "performance_results"))
                }))
            }), classifier_hyperparameter_set_type_names_unique)

    ## extract feature importance scores, if they were gathered
    feat_impt_scores <- setNames(lapply(classifier_hyperparameter_set_type_names_unique,
            function(p_classifier_hyperparameter_type_name) {
                hptds_list_list <- ml_global_results_list_wp_top[which(classifier_hyperparameter_type_names == p_classifier_hyperparameter_type_name)]
                setNames(lapply(hptds_list_list, function(hptds_list) {
                    lapply(hptds_list, "[[", "feat_import_scores")
                }), sapply(hptds_list_list, function(htps_list) {
                    htps_list[[1]]$performance_results$classifier_set_name
                }))
            }), classifier_hyperparameter_set_type_names_unique)

    list(performance_results=ml_global_results_list_hpts_top,
         feature_impt_scores=feat_impt_scores)
}

g_rank_by_score_decreasing <- function(x) {
    length(x) - rank(x, ties.method="average") + 1
}

g_impute_missing_data_average_df <- function(p_df) {
    setNames(do.call(cbind, lapply(p_df,
                                   function(mycol) {
                                       mycol[is.na(mycol)] <- mean(mycol, na.rm=TRUE)
                                       data.frame(mycol)
                                   })),
             names(p_df))
}

g_make_calculate_avgrank_within_groups <- function(p_case_group_ids, p_group_to_case_ids_map_list, p_rank_by_score_decreasing) {
    stopifnot(is.list(p_group_to_case_ids_map_list))

    p_case_group_ids
    p_group_to_case_ids_map_list
    
    function(p_case_scores, p_labels, p_case_ids) {
        ## need the indices of the rSNPs in the full set of 15,331 SNPs
        inds_pos <- p_case_ids[which(p_labels==1)]

        inds_map <- match(1:length(p_case_group_ids), p_case_ids)
        
        ## for each rSNP....
        mean(sapply(inds_pos, function(p_ind_pos) {
            ## get the group ID of the rSNP
            group_id <- p_case_group_ids[p_ind_pos]

            ## get the set of all SNPs in that group
            group_case_inds <- p_group_to_case_ids_map_list[[group_id]]
            stopifnot(! is.null(group_case_inds))

            inds_cases_within_fold_set <- inds_map[group_case_inds]
            stopifnot(! is.na(inds_cases_within_fold_set))
            
            ## use group_case_inds to index into 
            scores_for_group_cases <- p_case_scores[inds_cases_within_fold_set]
            stopifnot(! is.na(scores_for_group_cases))
            
            labels_for_group_cases <- p_labels[inds_cases_within_fold_set]
            stopifnot(! is.na(labels_for_group_cases))

            group_case_inds_analyze <- c(p_ind_pos, group_case_inds[which(labels_for_group_cases==0)])
            stopifnot(! is.na(group_case_inds_analyze))
            
            scores_to_analyze <- p_case_scores[inds_map[group_case_inds_analyze]]
            stopifnot(! is.na(scores_to_analyze))

            p_rank_by_score_decreasing(scores_to_analyze)[1]
        }))
    }
}



g_interval_clipper <- function(u) {
    pmax(pmin(u, 1.0), 0.0)
}

g_make_performance_getter <- function(p_performance_auc_calculator_func, p_interval_clipper_func, p_result_slot_name) {
    function(p_prediction_scores_vec, p_labels_vec) {
        p_interval_clipper_func(p_performance_auc_calculator_func(p_prediction_scores_vec[which(p_labels_vec==1)],
                                                                  p_prediction_scores_vec[which(p_labels_vec==0)])[[p_result_slot_name]])
    }
}

g_make_get_perf_results <- function(p_calculate_aupvr,
                                    p_calculate_auroc,
                                    p_calculate_avgrank=NULL) {

    ## obtain AUROC and AUPVR performance results given a vector of scores and a numeric vector of binary class labels
    function(p_prediction_scores, p_labels, p_case_inds)  {
        stopifnot(length(p_prediction_scores) == length(p_labels))
        stopifnot(is.vector(p_prediction_scores))
        stopifnot(is.numeric(p_prediction_scores))
        stopifnot(is.vector(p_labels))
        stopifnot(is.numeric(p_labels))

        ## randomize in case of weird bias due to tied scores
        rand_order <- sample(length(p_labels))
        p_prediction_scores_rand <- p_prediction_scores[rand_order]
        p_labels_rand <- p_labels[rand_order]
        
        if (! require(PRROC, quietly=TRUE)) {
            stop("package PRROC is missing")
        }

        aupvr <- p_calculate_aupvr(p_prediction_scores_rand,
                                   p_labels_rand)

        auroc <- p_calculate_auroc(p_prediction_scores_rand,
                                   p_labels_rand)
        
        res_list <- list(aupvr=aupvr,
                         auroc=auroc)

        if (! is.null(p_calculate_avgrank)) {

            avgrank <- p_calculate_avgrank(p_prediction_scores,
                                           p_labels,
                                           p_case_inds)
                                 
            res_list <- c(res_list, list(avgrank=avgrank))
        }
        
        res_list
    }
}

## function constructor for the "ranger" classifier
## NOTE:  ranger leaks memory pretty badly when "probability=TRUE"
g_make_classifier_function_ranger <- function(p_nthread=1, p_get_perf_results, p_feature_importance_type=NULL) {

    if (! suppressWarnings( require(ranger, quietly=TRUE) ) ) {
        stop("package ranger is missing")
    }

    package_version <- function(p_package_name) {
        ip_df <- installed.packages()
        ip_df[which(ip_df[,"Package"]==p_package_name),"Version"]
    }
    
    if (package_version("ranger") == "0.6.6") {
        stop("you are running Ranger version 0.6.6 -- downgrade to version 0.6.0 (from CRAN)")
    }
    
    function(p_classifier_feature_matrix,
             p_classifier_hyperparameter_list,
             p_label_vector,
             p_inds_cases_test,
             p_custom_objective_function_parameters_list=NULL) {

        if (! require(ranger, quietly=TRUE)) {
            stop("package ranger is missing")
        }
        if (! require(methods, quietly=TRUE)) {
            stop("package methods is missing")
        }

        stopifnot( ! is.null(p_classifier_feature_matrix))
        stopifnot(length(grep("data.frame|matrix", is(p_classifier_feature_matrix))) > 0)
        stopifnot(is.vector(p_label_vector))
        stopifnot(is.numeric(p_label_vector))
        stopifnot(sort(unique(p_label_vector)) == c(0,1))
        stopifnot(colnames(p_classifier_feature_matrix) != "label")

        ncases <- nrow(p_classifier_feature_matrix)
        
        inds_cases_training <- setdiff(1:ncases, p_inds_cases_test)

        train_labels <- p_label_vector[inds_cases_training]
        test_labels <- p_label_vector[p_inds_cases_test]

        rf_data_matrix <- cbind(p_classifier_feature_matrix,
                                label=factor(p_label_vector))

        if (! is.null(p_classifier_hyperparameter_list$weight_positive_class) &&
            p_classifier_hyperparameter_list$weight_positive_class != 1) {
            weights_for_classes <- c(1, p_classifier_hyperparameter_list$weight_positive_class)
            case_weights_train <- weights_for_classes[p_label_vector[inds_cases_training] + 1]
        }
        else {
            case_weights_train <- NULL
        }

        p_classifier_hyperparameter_list$weight_positive_class <- NULL
        
        hyperparameter_list_for_ranger <- c(p_classifier_hyperparameter_list,
                                            list(case.weights=case_weights_train))

        probability_tree_setting <- p_classifier_hyperparameter_list$probability
        if (is.null(probability_tree_setting)) {
            probability_tree_setting <- FALSE
        }

        param_list <- c(list(dependent.variable.name="label",
                           data=rf_data_matrix[inds_cases_training, ],
                           classification=TRUE,
                           respect.unordered.factors=TRUE,
                           verbose=FALSE,
                           num.threads=p_nthread),
                        hyperparameter_list_for_ranger)
        
        if (! is.null(p_feature_importance_type)) {
            param_list <- c(param_list, list(importance=p_feature_importance_type))
        }
        
        rf_model <- do.call(ranger, param_list)

        if (! is.null(p_feature_importance_type)) {
            feat_import_scores <- importance(rf_model)
        } else {
            feat_import_scores <- NULL
        }
        
	train_pred_res <- predict(rf_model,
                                  data=rf_data_matrix[inds_cases_training,],
		                  predict.all=!probability_tree_setting)$predictions

        ## workaround because ranger returns per-tree scores that are "1 or 2" for probability=FALSE, and between 0 and 1 for probability=TRUE

        if (probability_tree_setting) {
            train_pred_scores <- train_pred_res[,2]
	} else {
            train_pred_scores <- apply(train_pred_res - 1, 1, mean)
        }

	train_perf_results <- p_get_perf_results(train_pred_scores, train_labels, inds_cases_training)

        if (length(test_labels) > 0) {
            test_pred_res <- predict(rf_model,
                                     data=rf_data_matrix[p_inds_cases_test,],
                                     predict.all=!probability_tree_setting)$predictions

            if (probability_tree_setting) {
                test_pred_scores <- test_pred_res[,2]
            } else {
                test_pred_scores <- apply(test_pred_res - 1, 1, mean)
            }

            if (any(is.nan(test_pred_scores))) {
                print(sprintf("number of NaN values in ranger prediction scores: %d", length(which(is.nan(test_pred_scores)))))
                print("ranger hyperparameter set: ")
                print(p_classifier_hyperparameter_list)
                test_pred_scores[is.nan(test_pred_scores)] <- 0.5
            }
        
            test_perf_results <- p_get_perf_results(test_pred_scores, test_labels, p_inds_cases_test)
            
        } else {
            test_perf_results <- list(auroc=NA, aupvr=NA)
            if (! is.null(train_perf_results$avgrank)) {
                test_perf_results <- c(test_perf_results,
                                       list(avgrank=NA))
            }
        }

        ret_list <- list(train_auroc=train_perf_results$auroc,
                         train_aupvr=train_perf_results$aupvr,
                         test_auroc=test_perf_results$auroc,
                         test_aupvr=test_perf_results$aupvr)

        if (! is.null(train_perf_results$avgrank)) {
            ret_list <- c(ret_list,
                          list(train_avgrank=train_perf_results$avgrank,
                               test_avgrank=test_perf_results$avgrank))
        }
        
        if (! is.null(feat_import_scores)) {
            ret_list <- c(ret_list,
                          list(feat_import_scores=feat_import_scores))
        }

        ret_list
    }
}       

## SAVE: this is the reference R implementation of the function "g_custom_objective_function_avgrank_cxx"
## that is implemented in the file "custom_objective_function.cc"
g_custom_objective_function_avgrank <- function(p_preds,
                                                p_labels,
                                                p_group_to_case_ids_map_list,
                                                p_case_group_ids,
                                                p_bandwidth) {
    
    sqrt_pi_bw <- sqrt(pi)*p_bandwidth
    bw2 <- p_bandwidth^2.0
    tdbw2 <- 2.0 / bw2

    case_signs <- (-1.0)^p_labels

    mixed_list <- lapply(1:length(p_preds), function(p_case_id) {
        group_case_ids <- p_group_to_case_ids_map_list[[p_case_group_ids[p_case_id]]]
        case_pred <- p_preds[p_case_id]
        cases_for_comparison <- group_case_ids[p_labels[group_case_ids] != p_labels[p_case_id]]
        if (length(cases_for_comparison) > 0) {
            delta_preds <- case_pred - p_preds[cases_for_comparison]
            intermed_vals <- exp(-(delta_preds)^(2.0) / bw2) / sqrt_pi_bw
            gradient_val <- case_signs[p_case_id] * sum(intermed_vals)
            intermed2_vals <- (abs(delta_preds) + p_bandwidth)*intermed_vals
            hessian_val <- tdbw2 * sum(intermed2_vals)
        } else {
            ## this SNP is a lone rSNP
            probability_i <- 1.0/(1.0 + exp(-p_preds[p_case_id]))
            gradient_val <- probability_i - p_labels[p_case_id]
            hessian_val <- probability_i * (1.0 - probability_i)
        }
        list(grad=gradient_val,
             hess=hessian_val)
    })            

    grad_vector <- sapply(mixed_list, "[[", "grad")
    hess_vector <- sapply(mixed_list, "[[", "hess")

    ##  calculate your loss function gradient here
    list(grad=grad_vector,
         hess=hess_vector)
}

## Requires: Rcpp
## The purpose of this function is to enable running an Rcpp-defined function in
## a "worker process" via R's parallel framework, without getting the error
## "NULL value passed as symbol address". This code is based on a code example
## that Roman Francois posted on the gmane.comp.lang.r.rcpp mailing list, 26th
## Sept. 2013.
g_make_cxx_function_with_lazy_compile <- function(p_code, p_func_get_temp_directory, ...) {
    function(...) {
        if (! require(Rcpp, quietly=TRUE)) { stop("package \"Rcpp\" is missing") }
        temp_directory <- p_func_get_temp_directory()
        do.call(cppFunction, list(code=p_code, cacheDir=temp_directory))(...)
    }
}

## Workaround because Rcpp cppFunction doesn't play nice with multicore parallel;
## by default, all worker processes use the same cache directory and that leads
## to badness because they are randomly erasing one another's cached CPP files.
g_make_make_temp_dir_if_doesnt_exist <- function(p_get_temp_directory) {
    function(...) {
        temp_directory <- p_get_temp_directory()
        if (! file.exists(temp_directory)) {
            dir.create(temp_directory)
        }
    }
}

## this function is called once, during global setup
g_make_make_custom_xgboost_objective_avgrank <- function(p_global_case_group_ids,
                                                         p_global_group_to_case_ids_map_list,
                                                         p_func_get_temp_directory,
                                                         p_make_cxx_function_with_lazy_compile,
                                                         p_func_R_version_of_objective_function_for_comparison=NULL) {

    p_global_case_group_ids
    p_global_group_to_case_ids_map_list

    cxx_code_file_name <- "custom_objective_function.cc"
    cxx_code_avgrank <- readChar(cxx_code_file_name, file.info(cxx_code_file_name)$size)
    custom_objective_function_avgrank_cxx <- p_make_cxx_function_with_lazy_compile(cxx_code_avgrank,
                                                                                   p_func_get_temp_directory)
    
    ## this function is called once for each replication/fold
    function(p_train_case_ids, p_custom_objective_function_parameters_list) {

        bandwidth <- p_custom_objective_function_parameters_list$bandwidth
        stopifnot(! is.null(bandwidth))

        ## case_group_ids:  integer locus ID corresponding to each SNP
        case_group_ids <- p_global_case_group_ids[p_train_case_ids]
        group_ids <- unique(case_group_ids)

        group_to_case_ids_map_list <- setNames(lapply(group_ids, function(group_id) {
            global_case_ids_for_group <- p_global_group_to_case_ids_map_list[[group_id]]
            match(global_case_ids_for_group, p_train_case_ids)
        }), group_ids)
        
        function(preds, dtrain) {
            labels <- getinfo(dtrain, "label")

            ret_list_cxx <- custom_objective_function_avgrank_cxx(preds,
                                                                    labels,
                                                                    group_to_case_ids_map_list,
                                                                    case_group_ids,
                                                                    bandwidth) 

            if (! is.null(p_custom_objective_function_parameters_list$debug) &&
                p_custom_objective_function_parameters_list$debug == TRUE) {
                ret_list_R <- p_func_R_version_of_objective_function_for_comparison(preds,
                                                                                    labels,
                                                                                    group_to_case_ids_map_list,
                                                                                    case_group_ids,
                                                                                    bandwidth)

                stopifnot(mean(abs(ret_list_cxx$grad - ret_list_R$grad)) < 1e-8 &&
                          mean(abs(ret_list_cxx$hess - ret_list_R$hess)) < 1e-8)
            }
            
            ret_list_cxx
         }
    }

    
}

## function constructor for the "xgboost" classifier; this is hard-coded to use "NA" to denote missing data
g_make_classifier_function_xgboost <- function(p_nthread=1,
                                               p_get_perf_results,
                                               p_feature_importance_type=NULL,
                                               p_make_objective_function=function(...){"binary:logistic"},
                                               p_case_group_ids=NULL,
                                               p_verbose=0) {

    if (! require(xgboost, quietly=TRUE)) {
        stop("package xgboost is missing")
    }
    
    nthread_third_level_list <- list(nthread=p_nthread)

    avg_group_size_ncases <- mean(table(p_case_group_ids))

    function(p_classifier_feature_matrix,
             p_classifier_hyperparameter_list,
             p_label_vector,
             p_inds_cases_test,
             p_custom_objective_function_parameters_list=NULL) {

        ## load xgboost package; we will need it to assign cases to cross-validation folds
        if (! require(xgboost, quietly=TRUE)) {
            stop("package xgboost is missing")
        }
    
        stopifnot( ! is.null(p_classifier_feature_matrix))
        stopifnot(length(grep("dgCMatrix|matrix", is(p_classifier_feature_matrix))) > 0)
        stopifnot(is.vector(p_label_vector))
        stopifnot(is.numeric(p_label_vector))
        stopifnot(sort(unique(p_label_vector)) == c(0,1))
        stopifnot(colnames(p_classifier_feature_matrix) != "label")

        ncases <- nrow(p_classifier_feature_matrix)
        
        inds_cases_training <- setdiff(1:ncases, p_inds_cases_test)
        
        train_labels <- p_label_vector[inds_cases_training]
        test_labels <- p_label_vector[p_inds_cases_test]

        ## when we upgrade to newer xgboost that has row-based subsetting of xgb.DMatrix objects, this code to move to cerenkov_ml.R
        model_data <- xgb.DMatrix(p_classifier_feature_matrix[inds_cases_training, ], label=train_labels, missing=NA)

        if (! is.null(p_custom_objective_function_parameters_list)) {
            custom_objective_function_parameters_list <- p_custom_objective_function_parameters_list
        } else {
            custom_objective_function_parameters_list <- list()
        }
        
        objective_function <- p_make_objective_function(inds_cases_training,
                                                        p_custom_objective_function_parameters_list=custom_objective_function_parameters_list)
        
        xgb_params <- c(p_classifier_hyperparameter_list,
                       list(objective=objective_function))

        xgb_model <- xgb.train(params=xgb_params,
                               data=model_data,
                               nrounds=p_classifier_hyperparameter_list$nrounds,
                               verbose=p_verbose,
                               nthread=p_nthread)
        
        if (! is.null(p_feature_importance_type)) {
            dump_file_name <- tempfile("xgb_dump")
            xgb.dump(xgb_model, fname=dump_file_name, with.stats=TRUE)
            feat_import_scores <- as.data.frame(xgb.importance(feature_names=colnames(p_classifier_feature_matrix),
                                                               filename_dump=dump_file_name))
            file.remove(dump_file_name)
        } else {
            feat_import_scores <- NULL
        }
        
        train_pred_scores <- predict(xgb_model,
                                     p_classifier_feature_matrix[inds_cases_training, ],
                                     missing=NA)
        
        ## \/\/\/\/   this is a work-around for bug # 1503 in xgboost (see GitHub); might have been fixed in latest release of R xgboost package
        n_train_pred_scores <- length(train_pred_scores)
        if (length(train_pred_scores) < length(train_labels)) {
            train_labels <- train_labels[1:length(train_pred_scores)]
        }
        ## /\/\/\/\   this is a work-around for bug # 1503 in xgboost (see GitHub); might have been fixed in latest release of R xgboost package

        stopifnot(length(train_pred_scores) == length(train_labels))

        train_perf_results <- p_get_perf_results(train_pred_scores, train_labels, inds_cases_training)

        if (length(test_labels) > 0) {
            test_pred_scores <- predict(xgb_model,
                                        p_classifier_feature_matrix[p_inds_cases_test,],
                                        missing=NA)

            ## \/\/\/\/   this is a work-around for bug # 1503 in xgboost (see GitHub); might have been fixed in latest release of R xgboost package
            n_test_pred_scores <- length(test_pred_scores)
            if (length(test_pred_scores) < length(test_labels)) {
                test_labels <- test_labels[1:length(test_pred_scores)]
            }
            ## /\/\/\/\   this is a work-around for bug # 1503 in xgboost (see GitHub); might have been fixed in latest release of R xgboost package

            stopifnot(length(test_labels) == length(test_pred_scores))
            
            test_perf_results <- p_get_perf_results(test_pred_scores, test_labels, p_inds_cases_test)
        } else {
            test_perf_results <- list(auroc=NA, aupvr=NA)
            if (! is.null(train_perf_results$avgrank)) {
                test_perf_results <- c(test_perf_results,
                                       list(avgrank=NA))
            }
       }
        
        ret_list <- list(train_auroc=train_perf_results$auroc,
                         train_aupvr=train_perf_results$aupvr,
                         test_auroc=test_perf_results$auroc,
                         test_aupvr=test_perf_results$aupvr)

        if (! is.null(train_perf_results$avgrank)) {
            ret_list <- c(ret_list,
                          list(train_avgrank=train_perf_results$avgrank,
                               test_avgrank=test_perf_results$avgrank))
        }
        
        if (! is.null(feat_import_scores)) {
            ret_list <- c(ret_list, list(feat_import_scores=feat_import_scores))
        }

        ret_list
    }
}

g_make_classifier_function_passthrough <- function(p_scores_vec, p_get_perf_results) {
    stopifnot(! is.null(p_scores_vec))
    score_vec_case_names <- names(p_scores_vec)
    stopifnot(! is.null(score_vec_case_names))

    function(p_classifier_feature_matrix,
             p_classifier_hyperparameter_list,
             p_label_vector,
             p_inds_cases_test) {

        ncases <- length(p_scores_vec)
        inds_cases_training <- setdiff(1:ncases, p_inds_cases_test)

        train_labels <- p_label_vector[inds_cases_training]
        test_labels <- p_label_vector[p_inds_cases_test]

        train_pred_scores <- p_scores_vec[inds_cases_training]
        train_perf_results <- p_get_perf_results(train_pred_scores, train_labels, inds_cases_training)

        test_pred_scores <- p_scores_vec[p_inds_cases_test]
        test_perf_results <- p_get_perf_results(test_pred_scores, test_labels, p_inds_cases_test)
         
        ret_list <- list(train_auroc=train_perf_results$auroc,
                         train_aupvr=train_perf_results$aupvr,
                         test_auroc=test_perf_results$auroc,
                         test_aupvr=test_perf_results$aupvr)

        if (! is.null(train_perf_results$avgrank)) {
            ret_list <- c(ret_list,
                          list(train_avgrank=train_perf_results$avgrank,
                               test_avgrank=test_perf_results$avgrank))
        }

        ret_list
    }
}

g_feature_reducer_pls <- function(p_input_feature_matrix,
                                  p_case_label_vec,
                                  p_inds_cases_test,
                                  p_num_components) {
    stopifnot( ! is.null(p_num_components) )
    
    if (! suppressWarnings( require(pls, quietly=TRUE) ) ) { stop("package pls is missing") }
    if (! require(methods, quietly=TRUE)) { stop("package methods is missing") }
    
    p_input_feature_matrix_as_df <- data.frame(pls.x=I(p_input_feature_matrix))
    p_input_feature_matrix_as_df$label <- I(model.matrix(~y-1, data.frame(y=factor(p_case_label_vec))))
    
    inds_cases_train <- setdiff(1:length(p_case_label_vec), p_inds_cases_test)
    
    pls_model <- cppls(label ~ pls.x, data=p_input_feature_matrix_as_df[inds_cases_train,], ncomp=p_num_components)
    
    pls_pred <- predict(pls_model, p_input_feature_matrix_as_df, type="scores")
}


g_make_hyperparameter_grid_list <- function(p_param_list_values) {
    hyperparams_df <- expand.grid(p_param_list_values, stringsAsFactors=FALSE)
    hyperparams_list <- lapply(setNames(split(hyperparams_df, seq(nrow(hyperparams_df))), NULL), as.list)
    hyperparams_list <- lapply(hyperparams_list, function(p_list) { attr(p_list, "out.attrs") <- NULL; p_list })
}

## This function makes sure that the data frame (or matrix) contains no NaN values and
## no factors with more than 64 levels.  Note:  NA values are allowed for XGboost but not RF.
g_feature_matrix_is_OK <- function(p_feature_matrix) {
    if (! require(methods, quietly=TRUE)) { stop("package methods is missing") }
    if ("matrix" %in% is(p_feature_matrix)) {
        all(! is.nan(p_feature_matrix)) &
            all(apply(p_feature_matrix, 2,
                      function(mycol) {
                          if("integer" %in% is(mycol)) {
                              length(unique(mycol)) <= 64
                          } else {
                              TRUE
                          }}))
    } else {
        all(sapply(p_feature_matrix, function(mycol) { all(! is.nan(mycol)) })) &
        all(sapply(p_feature_matrix, function(mycol) {
            col_is_ok <- TRUE
            if (is.factor(mycol)) {
                if(length(levels(mycol)) > 64) {
                    col_is_ok <- FALSE
                }
            }
            col_is_ok
        }))
    }
}

g_get_snp_locus_ids <- function(p_snp_locus_coords, 
                                p_coord_distance_cutoff_bp=50000) {
    snp_chroms <- as.character(p_snp_locus_coords$chrom)
    snp_chroms_unique <- unique(snp_chroms)
    
    unlist(lapply(snp_chroms_unique,
           function(p_snp_chrom) {
               snp_coord_data_chrom <- subset(p_snp_locus_coords, chrom==p_snp_chrom)
               snp_coord_data_chrom <- snp_coord_data_chrom[order(snp_coord_data_chrom$coord),]
               intra_snp_distances <- diff(snp_coord_data_chrom$coord)
               intra_snp_distances_thresh <- as.integer(intra_snp_distances > p_coord_distance_cutoff_bp)
               locus_vals <- cumsum(c(1, intra_snp_distances_thresh))
               locus_names <- paste(p_snp_chrom, locus_vals, sep="-")
               names(locus_names) <- rownames(snp_coord_data_chrom)
               locus_names
           }))[rownames(p_snp_locus_coords)]
}

g_check_cross_validation_num_folds_not_too_big <- function(p_num_folds, p_case_label_vec) {
    npos <- length(which(p_case_label_vec==1))
    nneg <- length(which(p_case_label_vec==0))
    p_num_folds < length(p_case_label_vec)/(nneg/npos + 1)
}

#' Assign cases to folds using stratified sampling
#'
#' Returns an integer vector (of length \code{N} equal to the number of cases)
#' in which each entry is the cross-validation fold assigment of the corresponding
#' case. 
#'
#' @param p_num_folds A positive integer specifying the number of
#'     cross-validation folds
#' @param p_case_label_vec An integer vector (of length \code{N}) containing the
#'     case labels (0 = negative case, 1 = positive case)
#' @return An integer vector of length \code{N} whose values are all in the
#'     range \code{1:p_num_folds}, giving the fold assignment for each case
#' @seealso 
#'  \code{\link[dismo]{kfold}}
#' @rdname g_assign_cases_to_folds_by_case
#' @export 
#' @importFrom dismo kfold
#' @author Stephen A. Ramsey, \email{stephen.ramsey@@oregonstate.edu}
g_assign_cases_to_folds_by_case <- function(p_num_folds,
                                            p_case_label_vec) {
    
    stopifnot(g_check_cross_validation_num_folds_not_too_big(p_num_folds, p_case_label_vec))
    
    num_cases <- length(p_case_label_vec)
    dismo::kfold(1:num_cases,
                 k=p_num_folds,
                 by=p_case_label_vec)
}

#' Assign cases to folds by group.
#'
#' Returns a function that will generate a vector of integer fold assignments
#' for cases to cross-validation folds, while maintaining class balance in the
#' folds and ensuring that all cases from a group are assigned to the same fold
#' together.
#'
#' @param p_case_group_ids Length \code{N} vector of group IDs (which can be
#'     character or integer).
#' @param p_slop_allowed A numeric scalar indicating the maximum amount by which
#'     the number of positive cases in a group can exceed the expected number
#'     (0.5 would mean that the actual number of positive cases in group can
#'     never exceed 1.5-fold the expected number). Default: 0.5.
#' @return A function with calling signature (\code{p_num_folds},
#'     \code{p_case_label_vec}). That function returns an integer vector of
#'     length \code{N} whose values are in the range \code{1:p_num_folds}, where
#'     \code{p_num_folds} is the number of folds for the cross-validation (i.e.,
#'     in ten-fold cross-validation, \code{p_num_folds=10}.
#' @author Stephen A. Ramsey, \email{stephen.ramsey@@oregonstate.edu}
g_make_assign_cases_to_folds_by_group <- function(p_case_group_ids, p_slop_allowed=0.5) {

    p_case_group_ids ## force R to evaluate the promise argument, so it is stored with the closure

    function(p_num_folds, p_case_label_vec) {

        stopifnot(g_check_cross_validation_num_folds_not_too_big(p_num_folds, p_case_label_vec))
    
        table_res <- table(p_case_group_ids, p_case_label_vec)
        group_pos_counts <- table_res[,2]
        stopifnot(group_pos_counts > 0)  ## there should be an rSNP in every group!
        ncase <- length(p_case_label_vec)
        group_counts <- apply(table_res, 1, sum)
        group_names <- rownames(table_res)
        ngroup <- length(group_names)
        group_fold_assignments <- setNames(rep(NA, ngroup), group_names)
        case_counts_for_folds <- rep(0, p_num_folds)
        pos_case_counts_for_folds <- rep(0, p_num_folds)
        fold_ids <- 1:p_num_folds
        max_fold_case_count <- ceiling((1 + p_slop_allowed) * ncase / p_num_folds)
        num_pos_cases <- length(which(p_case_label_vec==1))
        max_fold_pos_cases <- ceiling((1 + p_slop_allowed) * max_fold_case_count * num_pos_cases / ncase)
        for (i in order(group_counts, decreasing=TRUE)) {
            group_count <- group_counts[i]
            group_pos_count <- group_pos_counts[i]
            inds_allowed <- which(case_counts_for_folds + group_count <= max_fold_case_count &
                                  pos_case_counts_for_folds + group_pos_count <= max_fold_pos_cases)
            stopifnot(length(inds_allowed) > 0)
            fold_assignment <- ifelse(length(inds_allowed) > 1,
                                      sample(inds_allowed,
                                             size=1,
                                             prob=(1 - (case_counts_for_folds[inds_allowed] / max_fold_case_count)) *
                                                 (1 - (pos_case_counts_for_folds[inds_allowed] / max_fold_pos_cases))),
                                      inds_allowed)
            group_fold_assignments[i] <- fold_assignment
            case_counts_for_folds[fold_assignment] <- case_counts_for_folds[fold_assignment] + group_counts[i]
            pos_case_counts_for_folds[fold_assignment] <- pos_case_counts_for_folds[fold_assignment] + group_pos_count
        }
        ret_case_fold_assignments <- setNames(group_fold_assignments[p_case_group_ids], NULL)
        stopifnot(max(ret_case_fold_assignments)==p_num_folds)
        stopifnot(min(ret_case_fold_assignments)==1)
        stopifnot(all(! is.nan(ret_case_fold_assignments)))
        stopifnot(all(! is.na(ret_case_fold_assignments)))
        stopifnot(all(is.integer(ret_case_fold_assignments)))
        ret_case_fold_assignments
    }
}

#' Make a function that runs a classification task within an error-catching context.
#'
#' Runs a user-supplied function in a tryCatch block, and if there is a warning
#' or error, passes the warning/error text to a user-specified message
#' notification function.
#'
#' @param p_classifier_runner_func A user-supplied classification task function
#'     (that is called with no arguments)
#' @param p_send_message_notification A user-specified function for error
#'     handling (which prints the error/warning text and, depending on
#'     configuration, optionally sends the error message text in an SMS message)
#' @return The wrapper function that (when called) will run the classification
#'     task within an error-catching context.
#' @author Stephen A. Ramsey, \email{stephen.ramsey@@oregonstate.edu}
g_make_classifier_runner_func_err_handling <- function(p_classifier_runner_func,
                                                       p_send_message_notification) {
    function() {
        tryCatch( { p_classifier_runner_func() },
                 warning=function(w) { p_send_message_notification(w); NULL },
                 error=function(e) { p_send_message_notification(e); NULL })
    }
}

#' Verify that singleton groups contain only positive cases
#'
#' Checks to make sure that any singleton case groups (i.e., groups with only
#' one case belonging to each of them) are only associated with cases that are
#' positive class labels.
#'
#' @param p_labels Integer vector of length equal to the number of cases. In
#'     each vector entry, value 0 means that the case is a negative case, and
#'     value 1 means that the case is a positive case.
#' @param p_group_to_case_map_list a list of length \code{unique(p_case_groups)}
#'     (see \code{\link{g_make_gorup_to_case_ids_map_list}}) in which each
#'     element (corresponding to a single group) contains an integer vector of
#'     case IDs of the cases that belong to that group.
#' @return \code{TRUE} or \code{FALSE} indicating whether or not all singleton
#'     groups contain only positive cases.
#' @author Stephen A. Ramsey, \email{stephen.ramsey@@oregonstate.edu}
g_verify_that_singleton_groups_are_all_positive_cases <-
    function(p_labels,
             p_group_to_case_map_list) {
        num_cases_per_group <- sapply(p_group_to_case_map_list, length)
        all(p_labels[unlist(p_group_to_case_map_list[names(num_cases_per_group[num_cases_per_group==1])])]==1)
    }

#' Make the group-to-case-ids mapping list
#' 
#' From a vector containing the group IDs of a set of N cases (case IDs are
#' presumed numbered 1:N), return a list (of length equal to the number of
#' unique group IDs) mapping group ID to the case IDs of the cases that are
#' associated with the group ID.
#'
#' @param p_case_groups Length N vector of group IDs (which can be character or
#'     integer).
#' @return a list of length \code{unique(p_case_groups)} in which each element
#'     (corresponding to a single group) contains an integer vector of case IDs
#'     of the cases that belong to that group.
#' @author Stephen A. Ramsey, \email{stephen.ramsey@@oregonstate.edu}
g_make_group_to_case_ids_map_list <- function(p_case_groups) {
    unique_group_ids <- unique(p_case_groups)
    setNames(lapply(unique_group_ids, function(p_group_id) {
        which(p_case_groups == p_group_id)
    }), unique_group_ids)
}

