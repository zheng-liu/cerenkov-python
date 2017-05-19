from cerenkov_ml_base import *
n_rep = 8

data_osu = pd.read_csv("features_OSU.tsv", sep="\t")

feat_osu = data_osu.drop("label", axis=1)

label = data_osu["label"]

fold0 = [(x%10+1) for x in range(len(feat_osu))]
fold_list = []
for _ in range(n_rep):
    np.random.shuffle(fold0)
    fold = pd.DataFrame(data=fold0, columns=["fold_id"])
    fold.index = feat_osu.index
    fold_list.append(fold)


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


method_list = [cerenkov17_test]*n_rep
feature_list = [feat_osu]*n_rep
label_vec = label
hyperparameter_list = [hyperparameter]*n_rep
fold_assignments = fold_list


# cerenkov_ml_test(method_list, feature_list, label_vec, hyperparameter_list, fold_assignments, ncpus=1)

# gc.collect()

# cerenkov_ml_test(method_list, feature_list, label_vec, hyperparameter_list, fold_assignments, ncpus=2)

# gc.collect()

cerenkov_ml_test(method_list, feature_list, label_vec, hyperparameter_list, fold_assignments, ncpus=-1)

gc.collect()

start_time = time.time()
for i in range(n_rep):
    method_list[i](feature_list[i], label_vec, hyperparameter_list[i], fold_assignments[i])
end_time = time.time()
test_time = end_time - start_time
print "without parallelization: ", test_time