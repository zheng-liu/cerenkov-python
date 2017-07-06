from cerenkov_ml_base import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

# np.random.seed(1337)
n_rep = 8
n_fold = 10

data_osu = pd.read_csv("features_OSU.tsv", sep="\t")
coord = pd.read_csv("coord.txt", sep="\t")
data_osu.index = coord.index # //TODO to ensure data_osu is already the "SNP ID secure" version
feat_osu = data_osu.drop("label", axis=1)
label = data_osu["label"]

fold_list = locus_sampling(data_osu, coord, n_rep, n_fold)
# print fold_list
# fold_list = snp_sampling(data_osu, n_rep, n_fold)
# print fold_list

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
feature_list = [feat_osu]*n_rep
label_vec = label
hyperparameter_list = [hyperparameter]*n_rep
fold_assignments = fold_list

start_time = time.time()
result_list_parallel = cerenkov_ml_test(method_list, feature_list, label_vec, hyperparameter_list, fold_assignments, ncpus=-1)
end_time = time.time()
print "with parallelization: ", end_time - start_time
print "**********************parallel result**********************\n"
print "parallel result: "
for result in result_list_parallel:
    print result["auroc"]

gc.collect()

result_list_unparallel = []
start_time = time.time()
for i in range(n_rep):
    result_list_unparallel.append(method_list[i](feature_list[i], label_vec, hyperparameter_list[i], fold_assignments[i], "SNP"))
end_time = time.time()
test_time = end_time - start_time
print "**********************without parallel result**********************\n"
print "without parallelization: ", test_time
for result in result_list_unparallel:
    print result["auroc"]

# result_parallel = np.concatenate(result_list_parallel)
# result_unparallel = np.concatenate(result_list_unparallel)
# np.savetxt("parallel.txt",result_parallel, delimiter="\t")
# np.savetxt("unparallel.txt",result_unparallel, delimiter="\t")