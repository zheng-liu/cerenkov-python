from cerenkov_ml_base import *

original_time = time.time()
n_rep = 2
n_fold = 5

data_osu = pandas.read_csv("features_OSU.tsv", sep="\t")
coord = pandas.read_csv("coord.txt", sep="\t")
data_osu.index = coord.index # //TODO to ensure data_osu is already the "SNP ID secure" version
data_osu["coord"] = coord.coord
data_osu.index.name = "rsid"
data_osu.set_index("coord", append=True, inplace=True)

hyperparameter = dict(
    learning_rate=0.1,
    n_estimators=30,
    gamma=10,
    subsample=1,
    colsample_bytree=0.85,
    base_score=0.1082121,
    scale_pos_weight=1,
    max_depth=6,
    seed=1337,
    nthread = 1
    )

method_list = [cerenkov17]*n_rep
dataset_list = [data_osu]*n_rep
hyperparameter_list = [hyperparameter]*n_rep

start_time = time.time()
result_list_parallel = cerenkov_ml(method_list, dataset_list, hyperparameter_list, n_rep, n_fold, "LOCUS", ncpus=-1)
end_time = time.time()
print "with parallelization: ", end_time - start_time, end_time - original_time
print "**********************parallel result**********************\n"
print "parallel result: "
for result in result_list_parallel:
    print result["avgrank"]

gc.collect()

result_list_unparallel = []
start_time = time.time()
for i in range(n_rep):
    result_list_unparallel.extend(method_list[i](feature_list[i], label_vec, hyperparameter_list[i], fold_assignments[i], "LOCUS"))
end_time = time.time()
test_time = end_time - start_time
print "**********************without parallel result**********************\n"
print "without parallelization: ", test_time
for result in result_list_unparallel:
    print result["avgrank"]