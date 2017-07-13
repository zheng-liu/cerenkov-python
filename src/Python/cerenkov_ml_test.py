from cerenkov_ml_base import *

n_rep = 4
n_fold = 10

data_osu = pandas.read_csv("features_OSU.tsv", sep="\t")
coord = pandas.read_csv("coord.txt", sep="\t")
data_osu.index = coord.index # //TODO to ensure data_osu is already the "SNP ID secure" version
data_osu["coord"] = coord.coord
data_osu.index.name = "rsid"
data_osu.set_index("coord", append=True, inplace=True)

hyperparameter_osu = dict(
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


hyperparameter_gwava_rf = dict(
    n_estimators=300,
    oob_score=True,
    n_jobs=-1,
    random_state=1337
    )

# method_list = [cerenkov17, gwava_rf]
# dataset_list = [data_osu]*n_rep
# hyperparameter_list = [hyperparameter_osu, hyperparameter_gwava_rf]

method_list = [gwava_rf, cerenkov17, gwava_rf, cerenkov17]
dataset_list = [data_osu, data_osu, data_osu, data_osu]
hyperparameter_list = [hyperparameter_gwava_rf, hyperparameter_osu, hyperparameter_gwava_rf, hyperparameter_osu]



start_time = time.time()
result_list_parallel = cerenkov_ml(method_list, dataset_list, hyperparameter_list, n_rep, n_fold, "LOCUS", ncpus=-1)
end_time = time.time()
print "**********************parallel result**********************\n"
print "with parallelization: ", end_time - start_time
for result in result_list_parallel:
    print result["avgrank"]



gc.collect()
start_time = time.time()
fold_list = locus_sampling(data_osu, n_rep, n_fold)
result_list_unparallel = []
for i in range(4):
    result_list_unparallel.extend(method_list[i](dataset_list[i], hyperparameter_list[i], fold_list[i], "LOCUS"))
end_time = time.time()
test_time = end_time - start_time
print "**********************without parallel result**********************\n"
print "without parallelization: ", test_time
for result in result_list_unparallel:
    print result["avgrank"]