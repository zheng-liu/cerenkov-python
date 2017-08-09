from base_mp import *

# def logit(x):
#     if x.any() > 1.0 or x.any() < 0.0:
#         print "[ERROR] invalid value out of [0.0, 1.0]!"
#         exit(0)
#     else:
#         return math.log(x / (1.0 - x))

# def inv_logit(x):
#     return 1.0/(1.0 + (math.exp(-x)))

# def logit_mean(x):
#     return inv_logit(np.mean(logit(x)))

def logit_mean(x):
    return scipy.special.expit(np.mean(scipy.special.logit(x)))

def cerenkov_analysis(fold_assign_method):

    sns.set(style="whitegrid", color_codes=True)

    # //TODO check if all the files are there
    method_table = pickle.load(open("method_table.p", "rb"))
    data_table = pickle.load(open("data_table.p", "rb"))
    fold_table = pickle.load(open("fold_table.p", "rb"))
    hp_table = pickle.load(open("hp_table.p", "rb"))
    task_table = pickle.load(open("task_table.p", "rb"))
    
    analysis = {"hp_optimal":{}}
    
    n_iterations = 10000
    if fold_assign_method == "SNP":

        # analyze results
        # optimal_table: table of performance for the optimal hyperparameters
        optimal_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold", "auroc", "aupvr"])
        # plot_table: table for plotting (including the boostrap resampling and logit transform)
        plot_table = pd.DataFrame(columns=["method", "auroc", "aupvr"])

        for i, gb_i in task_table.groupby(["method"]):
            method = i
            hp_optimal = 0
            aupvr_optimal = -1.0

            for j, gb_j in gb_i.groupby(["hp"]):
                if gb_j["aupvr"].mean() > aupvr_optimal:
                    hp_optimal = j
                    aupvr_optimal = gb_j["aupvr"].mean()
            
            analysis["hp_optimal"][i] = hp_table[i].ix[j, "hp"]
            optimal_table = optimal_table.append(task_table.loc[(task_table["method"] == method) & (task_table["hp"] == hp_optimal)], ignore_index=True)
        
            # logit "auroc" and "aupvr"
            for _ in range(n_iterations):
                mean_logit_auroc = logit_mean(resample(gb_i, replace=True, random_state=1337)["auroc"].values)
                mean_logit_aupvr = logit_mean(resample(gb_i, replace=True, random_state=1337)["aupvr"].values)
                plot_table = plot_table.append({"method": method, "auroc": mean_logit_aupvr, "aupvr": mean_logit_aupvr}, ignore_index=True)

        analysis["optimal_table"] = optimal_table
        analysis["plot_table"] = plot_table
        pickle.dump(analysis, open("analysis.p", "wb"))
        print "[INFO] dump to analysis.p!"
        
        # plot results
        plot_logit_auroc = sns.boxplot(x="method", y="auroc", data=plot_table, whis=2.5).get_figure()
        plot_logit_auroc.savefig("auroc.pdf")
        print "[INFO] auroc figure saved to auroc.pdf!"
        
        plot_logit_auroc.clf()

        plot_logit_aupvr = sns.boxplot(x="method", y="aupvr", data=plot_table, whis=2.5).get_figure()
        plot_logit_aupvr.savefig("aupvr.pdf")
        print "[INFO] aupvr figure saved to aupvr.pdf!"

    elif fold_assign_method == "LOCUS":
        
        # analyze results
        # optimal_table: table of performance for the optimal hyperparameters
        optimal_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold", "auroc", "aupvr", "avgrank"])
        # plot_table: table for plotting (including the boostrap resampling and logit transform)
        plot_table = pd.DataFrame(columns=["method", "auroc", "aupvr", "avgrank"])

        for i, gb_i in task_table.groupby(["method"]):
            method = i
            hp_optimal = 0
            avgrank_optimal = float("inf")

            for j, gb_j in gb_i.groupby(["hp"]):
                if gb_j["avgrank"].mean() < avgrank_optimal:
                    hp_optimal = j
                    avgrank_optimal = gb_j["avgrank"].mean()
            
            analysis["hp_optimal"][i] = hp_table[i].ix[j, "hp"]
            optimal_table = optimal_table.append(task_table.loc[(task_table["method"] == method) & (task_table["hp"] == hp_optimal)], ignore_index=True)
        
            # logit "auroc" and logit "aupvr" and "avgrank"
            for i_iterations in range(n_iterations):
                mean_logit_auroc = logit_mean(resample(gb_i, replace=True, random_state=1337+i_iterations)["auroc"].values)
                mean_logit_aupvr = logit_mean(resample(gb_i, replace=True, random_state=1337+i_iterations)["aupvr"].values)
                mean_avgrank = np.mean(resample(gb_i, replace=True, random_state=1337+i_iterations)["avgrank"].values)
                plot_table = plot_table.append({"method": method, "auroc": mean_logit_auroc, "aupvr": mean_logit_aupvr, "avgrank": mean_avgrank}, ignore_index=True)
        
        analysis["optimal_table"] = optimal_table
        analysis["plot_table"] = plot_table
        pickle.dump(analysis, open("analysis.p", "wb"))
        print "[INFO] dump to analysis.p!"

        # plot results
        plot_avgrank = sns.boxplot(x="method", y="avgrank", data=plot_table, whis=2.5).get_figure()
        plot_avgrank.savefig("avgrank.pdf")
        print "[INFO] avgrank figure saved to avgrank.pdf!"
        
        plot_avgrank.clf()

        plot_logit_auroc = sns.boxplot(x="method", y="auroc", data=plot_table, whis=2.5).get_figure()
        plot_logit_auroc.savefig("auroc.pdf")
        print "[INFO] auroc figure saved to auroc.pdf!"
        
        plot_logit_auroc.clf()

        plot_logit_aupvr = sns.boxplot(x="method", y="aupvr", data=plot_table, whis=2.5).get_figure()
        plot_logit_aupvr.savefig("aupvr.pdf")
        print "[INFO] aupvr figure saved to aupvr.pdf!"

    else:
        print "Invalid fold assign method!"
        exit(0)

if __name__ == "__main__":
    cerenkov_analysis("LOCUS")