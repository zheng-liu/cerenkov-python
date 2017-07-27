# main function for base.py
from base import *

data_osu = pandas.read_csv("features_OSU.tsv", sep="\t")
coord = pandas.read_csv("coord.txt", sep="\t")
data_osu.index = coord.index # //TODO to ensure data_osu is already the "SNP ID secure" version
data_osu["coord"] = coord.coord
data_osu.index.name = "rsid"
data_osu.set_index("coord", append=True, inplace=True)


start_time = time.time()
fold_kwargs={}
fold_kwargs["cutoff_bp"] = 50000
fold_kwargs["slope_allowed"] = 0.5
fold_kwargs["seed"] = 1337
method_table = method_generate()
hp_table = hp_generate()
data_table = data_generate(["cerenkov17"], [data_osu])
fold_table = fold_generate(data_osu, n_rep=2, n_fold=5, fold_assign_method="SNP", **fold_kwargs)
task_table = task_pool(method_table, hp_table, data_table, fold_table)
result_task_table = cerenkov_ml(task_table, method_table, fold_table, hp_table, data_table, fold_assign_method="SNP", ncpus=-1, feature_reduced=False)
print result_task_table
