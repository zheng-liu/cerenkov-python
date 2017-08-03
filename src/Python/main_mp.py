# main function for base.py
from base_mp import *

data_osu = pd.read_csv("features_OSU.tsv", sep="\t")
coord = pd.read_csv("coord.txt", sep="\t")
# TODO to ensure data_osu is already the "SNP ID secure" version
data_osu.index = coord.index  # Set 1st index
data_osu.index.name = "rsid"
data_osu["coord"] = coord.coord
data_osu.set_index("coord", append=True, inplace=True)  # Set 2nd index
pos_neg_ratio = len(data_osu[data_osu["label"] == 1])*1.0 / len(data_osu[data_osu["label"] == 0])
print pos_neg_ratio

kwargs={}
kwargs["cutoff_bp"] = 50000
kwargs["slope_allowed"] = 0.5
kwargs["seed"] = 1337
method_table = method_generate()
hp_table = hp_generate(pos_neg_ratio)
fold_table = fold_generate(data_osu, n_rep=2, n_fold=5, fold_assign_method="LOCUS", **kwargs)
data_table = data_generate(["cerenkov17", "gwava_xgb", "gwava_rf"], [data_osu, data_osu, data_osu], fold_assign_method = "LOCUS", **kwargs)


task_table = task_pool(method_table, hp_table, data_table, fold_table)

s_time = time.time()
result_task_table = cerenkov_ml(task_table, method_table, fold_table, hp_table, data_table, fold_assign_method="LOCUS", ncpus=-1, feature_reduced=False)
e_time = time.time()
print result_task_table
print "cerenkov_ml TIME= ", e_time - s_time

cerenkov_analysis(fold_assign_method="LOCUS")