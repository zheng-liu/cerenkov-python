import math, time, thread, sys
import pp

import pandas as pd
import numpy as np
import xgboost as xgb
import time
import gc
from cerenkov_ml_base import *
warnings.filterwarnings("ignore", category=DeprecationWarning)

n_rep = 2

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

start_time = time.time()
result_list_parallel = cerenkov_ml_test(method_list, feature_list, label_vec, hyperparameter_list, fold_assignments, ncpus=-1)
end_time = time.time()
print "with parallelization: ", end_time - start_time
print "**********************parallel result**********************\n"
# print "parallel result: ", result_list_parallel

gc.collect()

result_list_unparallel = []
start_time = time.time()
for i in range(n_rep):
    result_list_unparallel.append(method_list[i](feature_list[i], label_vec, hyperparameter_list[i], fold_assignments[i]))
end_time = time.time()
test_time = end_time - start_time
print "**********************without parallel result**********************\n"
print "without parallelization: ", test_time
print result_list_unparallel




# file = open("parallel_unparallel.txt", "w")
# file.write("parallel:\n")
# for r in result_list_parallel:
#     file.write("%s\n" % r)
# file.write("unparallel:\n")
# for r in result_list_unparallel:
#     file.write("%s\n" % r)
# file.close()

print np.concatenate(result_list_parallel, axis=1)

result_parallel = np.concatenate(result_list_parallel, axis=1)
result_unparallel = np.concatenate(result_list_unparallel, axis=1)
np.savetxt("parallel.txt",result_parallel, delimiter="\t")
np.savetxt("unparallel.txt",result_unparallel, delimiter="\t")

# start_time = time.time()
# result = cerenkov17_test(feature_list[0], label_vec, hyperparameter_list[0], fold_assignments[0])
# print result
# end_time = time.time()
# test_time = end_time - start_time
# print "without parallelization: ", test_time