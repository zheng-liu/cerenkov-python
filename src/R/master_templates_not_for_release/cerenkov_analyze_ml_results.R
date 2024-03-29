## analyze_cerenkov_ml_results.R:  produce a TSV file of results of the ML and make plots of
## AUPVR and AUROC
##
## Stephen A. Ramsey

load("cerenkov_ml.Rdata")

g_logit <- function(x) {
    stopifnot(x <= 1.0 & x >= 0.0)
    log(x/(1.0 - x))
}

g_inv_logit <- function(x) {
    1.0/(1.0 + exp(-x))
}

g_ml_performance_results <- g_ml_results$performance_results

hyperparameter_set_type_names <- names(g_ml_performance_results)

library(reshape2)

## for each hyperparameter set type (HPST) name, make one super-wide data frame for each workplan; put on a per-HPST list
g_data_wide <- lapply(g_ml_performance_results, function(df) {
    df$workplan_id <- as.integer(as.character(df$workplan_id))
    unique_workplan_ids <- unique(df$workplan_id)

    col_classifier_name <- which(names(df) == "classifier_name")
    
    res_df <- data.frame(do.call(rbind,
                                 lapply(unique_workplan_ids,
                                        function(workplan_id) {
                                            inds <- which(df$workplan_id == workplan_id)
                                            df[inds[1], col_classifier_name:ncol(df)]})),
                         acast(df, workplan_id ~ replication_id + cv_fold_id, value.var="test_aupvr"),
                         acast(df, workplan_id ~ replication_id + cv_fold_id, value.var="test_auroc"),
                         acast(df, workplan_id ~ replication_id + cv_fold_id, value.var="train_aupvr"),
                         acast(df, workplan_id ~ replication_id + cv_fold_id, value.var="train_auroc"),
                         stringsAsFactors=FALSE)
    
    nfolds <- max(df$cv_fold_id)
    nreps <- max(df$replication_id)
    col_offset <- ncol(df) - col_classifier_name + 1
    inds <- which(df$workplan_id == unique_workplan_ids[1])
    repfolds <- paste(df$replication_id[inds], df$cv_fold_id[inds], sep="_")
    names(res_df)[(col_offset + 1):(col_offset + nfolds*nreps)] <- paste("test_aupvr", repfolds, sep="_")
    names(res_df)[(col_offset + nfolds*nreps + 1):(col_offset + 2*nfolds*nreps)] <- paste("test_auroc", repfolds, sep="_")
    names(res_df)[(col_offset + 2*nfolds*nreps + 1):(col_offset + 3*nfolds*nreps)] <- paste("train_aupvr", repfolds, sep="_")
    names(res_df)[(col_offset + 3*nfolds*nreps + 1):(col_offset + 4*nfolds*nreps)] <- paste("train_auroc", repfolds, sep="_")

    if ("train_avgrank" %in% names(df)) {
        res_df <- cbind(res_df,
                        acast(df, workplan_id ~ replication_id + cv_fold_id, value.var="test_avgrank"),
                        acast(df, workplan_id ~ replication_id + cv_fold_id, value.var="train_avgrank"))
        names(res_df)[(col_offset + 4*nfolds*nreps + 1):(col_offset + 5*nfolds*nreps)] <- paste("test_avgrank", repfolds, sep="_")
        names(res_df)[(col_offset + 5*nfolds*nreps + 1):(col_offset + 6*nfolds*nreps)] <- paste("train_avgrank", repfolds, sep="_")

    }
    
    res_df$replication_id <- NULL
    res_df$cv_fold_id <- NULL

    res_df
})

all_workplan_ids <- unique(unlist(lapply(g_data_wide, "[[", "workplan_id")))

## compute the average performance values
perf_strings <- c("test_aupvr","test_auroc","train_aupvr","train_auroc")
if (length(grep("avgrank", names(g_data_wide[[1]]))) > 0) {
    perf_strings <- c(perf_strings, "test_avgrank", "train_avgrank")
}

g_logit_mean <- function(x) {
    g_inv_logit(mean(g_logit(x)))
}

avg_results <- t(do.call(rbind, lapply(perf_strings,
                      function(string_to_search_with) {
                          sapply(all_workplan_ids,
                                 function(workplan_id) {
                                     unlist(lapply(g_data_wide, function(df) {
                                         ind <- which(df$workplan_id == workplan_id)
                                         if (length(ind) == 0) { return(NULL) }
                                         values_to_average <- unlist(df[ind, grep(string_to_search_with, names(df))])
                                         if (length(grep("aupvr|auroc", string_to_search_with)) > 0) {
                                             reducer_func <- g_logit_mean
                                         } else {
                                             reducer_func <- mean
                                         }
                                         avg_val <- reducer_func(values_to_average)
                                     }))
                                 })
                      })))
colnames(avg_results) <- paste("avg", perf_strings, sep="_")

lower_95ci_results <- t(do.call(rbind, lapply(perf_strings,
                                            function(string_to_search_with) {
                                                sapply(all_workplan_ids,
                                                       function(workplan_id) {
                                                           unlist(lapply(g_data_wide, function(df) {
                                                               ind <- which(df$workplan_id == workplan_id)
                                                               if (length(ind) == 0) { return(NULL) }
                                                               values_to_analyze <- unlist(df[ind, grep(string_to_search_with, names(df))])
                                                               if (length(grep("aupvr|auroc", string_to_search_with)) > 0) {
                                                                   reducer_func <- g_logit_mean
                                                               } else {
                                                                   reducer_func <- mean
                                                               }
                                                               quantile(replicate(1000, { reducer_func(sample(values_to_analyze, replace=TRUE)) }), probs=0.025)
                                                           }))
                                                       })
                                            })))

colnames(lower_95ci_results) <- paste("lower_95ci_", perf_strings, sep="_")

upper_95ci_results <- t(do.call(rbind, lapply(perf_strings,
                                            function(string_to_search_with) {
                                                sapply(all_workplan_ids,
                                                       function(workplan_id) {
                                                           unlist(lapply(g_data_wide, function(df) {
                                                               ind <- which(df$workplan_id == workplan_id)
                                                               if (length(ind) == 0) { return(NULL) }
                                                               values_to_analyze <- unlist(df[ind, grep(string_to_search_with, names(df))])
                                                               if (length(grep("aupvr|auroc", string_to_search_with)) > 0) {
                                                                   reducer_func <- g_logit_mean
                                                               } else {
                                                                   reducer_func <- mean
                                                               }
                                                               quantile(replicate(1000, { reducer_func(sample(values_to_analyze, replace=TRUE)) }), probs=0.975)
                                                           }))
                                                       })
                                            })))

colnames(upper_95ci_results) <- paste("upper_95ci_", perf_strings, sep="_")

## make a master data frame, but not yet with the performance values
g_wide_df_master <- do.call(rbind, lapply(g_data_wide, function(df) {
    inds_hyperparameter_columns <- grep("classifier_hyperparameters", names(df))
    single_column_hyperparameters <- setNames(apply(df[, inds_hyperparameter_columns, drop=FALSE], 1, function(myrow) {
        mylist <- as.list(myrow)
        paste(paste(names(mylist), mylist, sep="="), collapse=", ")
    }), NULL)
    data.frame(df[,setdiff(1:(min(grep("aupvr", names(df)))-1),inds_hyperparameter_columns)],
               single_column_hyperparameters,
               df[,min(grep("aupvr", names(df))):ncol(df)],
               stringsAsFactors=FALSE)
}))


## Insert The performance values into the master data frame (making sure the relative row order is not messed up)
stopifnot(g_wide_df_master$workplan_id == sort(g_wide_df_master$workplan_id))
g_wide_df_final <- cbind(g_wide_df_master[,1:7],
                         avg_results,
                         lower_95ci_results,
                         upper_95ci_results,
                         g_wide_df_master[,8:ncol(g_wide_df_master)])
rownames(g_wide_df_final) <- NULL

## order the rows so that highest average AUPVR is the first row
g_wide_df_final <- g_wide_df_final[order(g_wide_df_final$avg_test_aupvr, decreasing=TRUE),]

## final underscore makes sure we don't pick up "avg_test_aupvr"!
inds_test_aupvr <- grep("test_aupvr_", names(g_wide_df_final))

g_wide_df_final$pvalue <- NA

if (nrow(g_wide_df_final) > 1) {
    g_wide_df_final$pvalue[2:nrow(g_wide_df_final)] <- sapply(2:nrow(g_wide_df_final),
                                                              function(rowind) {
                                                                  data_reference_row <- unlist(g_logit(g_wide_df_final[1, inds_test_aupvr]))
                                                                  data_other_row <- unlist(g_logit(g_wide_df_final[rowind, inds_test_aupvr]))
                                                                  ret_pv <- NA
                                                                  if (length(data_reference_row) > 1 && all(! is.na(data_reference_row))) {
                                                                      ret_pv <- t.test(data_reference_row,
                                                                                       data_other_row,
                                                                                       paired=TRUE)$p.value
                                                                  }
                                                                  ret_pv
                                                              })
}

g_wide_df_final <- g_wide_df_final[, c(1:(min(grep("aupvr", names(g_wide_df_final)))-1),
                                       ncol(g_wide_df_final),
                                       (min(grep("aupvr", names(g_wide_df_final)))):(ncol(g_wide_df_final)-1))]

col_ind_workplan_id <- which(names(g_wide_df_final)=="workplan_id")

## make the workplan the first column
g_wide_df_final <- g_wide_df_final[, c(col_ind_workplan_id,
                                       setdiff(1:ncol(g_wide_df_final), col_ind_workplan_id))]

## save output to a TSV file
write.table(g_wide_df_final,
            file="cerenkov_ml.txt",
            sep="\t",
            row.names=FALSE,
            col.names=TRUE,
            quote=FALSE)

              
make_results_plot <- function(p_measure_name, p_results_df) {
    require(ggplot2)
    
    cols_for_plot <- c(which("workplan_set_name"==names(p_results_df)),
                       setdiff(grep(paste("test", p_measure_name, sep="_"), names(p_results_df)),
                               grep(paste(p_measure_name, "_", sep=""), names(p_results_df))))

# -------------- use this section if you did tuning -------------
#    hyp_best <- p_results_df$single_column_hyperparameters[1]
#    hpst_best <- p_results_df$classifier_hyperparameter_set_type_name[1]

#    rows_use <- c(which(p_results_df$single_column_hyperparameters == hyp_best &
#                        p_results_df$classifier_hyperparameter_set_type_name == hpst_best),
#                  which(p_results_df$workplan_set_name == "GWAVA_RF_published"))
# -------------- use this section if you did tuning -------------

    rows_use <- which(! (p_results_df$workplan_set_name %in% c()))  # exclude here
    
    df_plot_wide <- p_results_df[rows_use, cols_for_plot]

    library(reshape2)
    df_plot_melted <- data.frame(melt(df_plot_wide,
                                      id.vars=c("workplan_set_name"),
                                      measure.vars=c(paste("avg_test", p_measure_name, sep="_")),
                                      value.name=toupper(p_measure_name)),
                                 lower95=melt(df_plot_wide,
                                              id.vars=c("workplan_set_name"),
                                              measure.vars=c(paste("lower_95ci__test", p_measure_name, sep="_")))$value,
                                 upper95=melt(df_plot_wide,
                                              id.vars=c("workplan_set_name"),
                                              measure.vars=c(paste("upper_95ci__test", p_measure_name, sep="_")))$value)

    workplan_set_names <- as.character(df_plot_melted$workplan_set_name)
    workplan_set_names <- gsub("_published","",workplan_set_names)

    order_decreasing <- ifelse(p_measure_name == "avgrank", TRUE, FALSE)
    
    df_plot_melted$workplan_set_name <- factor(workplan_set_names,
                                               levels=workplan_set_names[order(df_plot_melted[[toupper(p_measure_name)]],
                                                                               decreasing=order_decreasing)])


    ggplot(df_plot_melted, aes_string(x="workplan_set_name", y=toupper(p_measure_name))) +
        geom_point() +
        theme_gray(base_size=18) +
        theme(axis.title.x=element_blank(),
              axis.text.x=element_text(angle=45, hjust=1)) +
        geom_errorbar(aes(ymin=lower95, ymax=upper95)) + ggsave(paste(p_measure_name, ".pdf", sep=""))

}                             

if (all(! is.na(g_wide_df_final$avg_test_aupvr))) {
    make_results_plot("aupvr", g_wide_df_final)
    make_results_plot("auroc", g_wide_df_final)
    make_results_plot("avgrank", g_wide_df_final)
}

g_ml_feature_impt_scores <- g_ml_results$feature_impt_scores

lapply(hyperparameter_set_type_names, function(p_hyperparameter_set_type_name) {
    impt_list <- g_ml_feature_impt_scores[[p_hyperparameter_set_type_name]]
    lapply(names(impt_list), function(p_workplan_name) {
        impt_df <- impt_list[[p_workplan_name]][[1]]
        if (! is.null(impt_df)) {
            write.table(impt_df,
                        file=paste("cerenkov_feature_importance_", p_hyperparameter_set_type_name, "_", p_workplan_name, ".txt", sep=""),
                        sep="\t",
                        quote=FALSE,
                        row.names=FALSE,
                        col.names=TRUE)
        }        
    })
    
})


