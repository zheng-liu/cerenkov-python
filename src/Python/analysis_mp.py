from base_mp import *

def logit(x):
    if x > 1.0 or x < 0.0:
        print "[ERROR] invalid value out of [0.0, 1.0]!"
        exit(0)
    else:
        return math.log(x / (1.0 - x))

def inv_logit(x):
    return 1.0/(1.0 + (math.exp(-x)))

def logit_mean(x):
    return inv_logit(mean(logit(x)))

def cerenkov_analysis(fold_assign_method):

    sns.set(style="whitegrid", color_codes=True)

    # //TODO check if all the files are there
    method_table = pickle.load(open("method_table.p", "rb"))
    data_table = pickle.load(open("data_table.p", "rb"))
    fold_table = pickle.load(open("fold_table.p", "rb"))
    hp_table = pickle.load(open("hp_table.p", "rb"))
    task_table = pickle.load(open("task_table.p", "rb"))
    
    analysis = {"hp_optimal":{}}

    if fold_assign_method == "SNP":

    	# analyze results
        plot_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold", "auroc", "aupvr"])

        for i, gb_i in task_table.groupby(["method"]):
            method = i
            hp_optimal = 0
            aupvr_optimal = -1.0

            for j, gb_j in gb_i.groupby(["hp"]):
                if gb_j["aupvr"].mean() > aupvr_optimal:
                    hp_optimal = j
                    aupvr_optimal = gb_j["aupvr"].mean()
            
            analysis["hp_optimal"][i] = hp_table[i].ix[j, "hp"]
            plot_table = plot_table.append(task_table.loc[(task_table["method"] == method) & (task_table["hp"] == hp_optimal)], ignore_index=True)
        
        # logit "auroc" and "aupvr"
        

        analysis["plot_table"] = plot_table
        pickle.dump(analysis, open("analysis.p", "wb"))
        print "[INFO] dump to analysis.p!"
        
        # plot results
        plot_auroc = sns.boxplot(x="method", y="auroc", data=plot_table).get_figure()
        plot_auroc.savefig("auroc.pdf")
        print "[INFO] auroc figure saved to auroc.pdf!"
        
        plot_auroc.clf()

        plot_aupvr = sns.boxplot(x="method", y="aupvr", data=plot_table).get_figure()
        plot_aupvr.savefig("aupvr.pdf")
        print "[INFO] aupvr figure saved to aupvr.pdf!"

    elif fold_assign_method == "LOCUS":

        plot_table = pd.DataFrame(columns=["method", "hp", "data", "fold_index", "i_rep", "i_fold", "auroc", "aupvr", "avgrank"])

        for i, gb_i in task_table.groupby(["method"]):
            method = i
            hp_optimal = 0
            avgrank_optimal = float("inf")

            for j, gb_j in gb_i.groupby(["hp"]):
                if gb_j["avgrank"].mean() < avgrank_optimal:
                    hp_optimal = j
                    avgrank_optimal = gb_j["avgrank"].mean()
            
            analysis["hp_optimal"][i] = hp_table[i].ix[j, "hp"]
            plot_table = plot_table.append(task_table.loc[(task_table["method"] == method) & (task_table["hp"] == hp_optimal)], ignore_index=True)

        analysis["plot_table"] = plot_table
        pickle.dump(analysis, open("analysis.p", "wb"))
        print "[INFO] dump to analysis.p!"

        # plot results
        plot_avgrank = sns.boxplot(x="method", y="avgrank", data=plot_table).get_figure()
        plot_avgrank.savefig("avgrank.pdf")
        print "[INFO] avgrank figure saved to avgrank.pdf!"
        
        plot_avgrank.clf()

        plot_auroc = sns.boxplot(x="method", y="auroc", data=plot_table).get_figure()
        plot_auroc.savefig("auroc.pdf")
        print "[INFO] auroc figure saved to auroc.pdf!"
        
        plot_auroc.clf()

        plot_aupvr = sns.boxplot(x="method", y="aupvr", data=plot_table).get_figure()
        plot_aupvr.savefig("aupvr.pdf")
        print "[INFO] aupvr figure saved to aupvr.pdf!"

    else:
        print "Invalid fold assign method!"
        exit(0)

if __name__ == "__main__":
    cerenkov_analysis("LOCUS")