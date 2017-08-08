# main function for base.py
from base_mp import *

################################
# data preprocess
################################

# data_osu = pd.read_csv("features_OSU.tsv", sep="\t")
# coord = pd.read_csv("coord.txt", sep="\t")
# print coord
# # TODO to ensure data_osu is already the "SNP ID secure" version
# data_osu.index = coord.index  # Set 1st index
# data_osu.index.name = "rsid"
# data_osu["coord"] = coord.coord
# data_osu.set_index("coord", append=True, inplace=True)  # Set 2nd index

# data_gwava = pd.read_csv("GWAVA.csv", sep=",")
# data_gwava.index = coord.index
# data_gwava.index.name = "rsid"
# data_gwava["coord"] = coord.coord
# data_gwava.set_index("coord", append=True, inplace=True)

data_osu = pd.read_csv("features_OSU.tsv", sep="\t")
data_gwava = pd.read_csv("GWAVA.csv", sep=",", index_col=0)
data_gwava = data_gwava.ix[data_osu.index]
coord = pd.read_csv("coord.txt", sep="\t")
data_osu.index = coord.index  # Set 1st index
data_osu.index.name = "rsid"
data_gwava.index = coord.index
data_gwava.index.name = "rsid"
data_gwava["coord"] = coord.coord
data_osu["coord"] = coord.coord
data_osu.set_index("coord", append=True, inplace=True)  # Set 2nd index
data_gwava.set_index("coord", append=True, inplace=True)

one_hot_gwava = pd.get_dummies(data_gwava["chr"], prefix="chrom")
data_gwava = data_gwava.drop(["chr", "end", "start"], axis=1)
data_gwava = data_gwava.join(one_hot_gwava)
chromSet = ['chr' + str(i) for i in range(1, 23)]+["chrX"]
for chrom in chromSet:
    data_gwava.rename(columns={"chrom_"+chrom : "chrom"+chrom}, inplace=True)
data_gwava.rename(columns={"cls": "label"}, inplace=True)

pos_neg_ratio = len(data_osu[data_osu["label"] == 1])*1.0 / len(data_osu[data_osu["label"] == 0])
sampling_method = "LOCUS"
kwargs={}
kwargs["cutoff_bp"] = 50000
kwargs["slope_allowed"] = 0.5
kwargs["seed"] = 1337
method_table = method_generate()
hp_table = hp_generate(pos_neg_ratio)
fold_table = fold_generate(data_osu, n_rep=200, n_fold=10, fold_assign_method=sampling_method, **kwargs)
data_table = data_generate(["cerenkov17", "gwava_xgb", "gwava_rf"], [data_osu, data_gwava, data_gwava], fold_assign_method=sampling_method, **kwargs)
task_table = task_pool(method_table, hp_table, data_table, fold_table)


################################
# machine learning
################################
s_time = time.time()
result_task_table = cerenkov_ml(task_table, method_table, fold_table, hp_table, data_table, fold_assign_method=sampling_method, ncpus=-1, feature_reduced=False)
e_time = time.time()
print result_task_table
print "cerenkov_ml TIME= ", e_time - s_time

################################
# data analysis
################################
# cerenkov_analysis(fold_assign_method=sampling_method)