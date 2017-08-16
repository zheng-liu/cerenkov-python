# main function for base.py
from base_mp import *

################################
# data preprocess
################################

# one-hot-encode OSU18 dataset
data_osu18 = pd.read_csv("feature_matrix_osu18_2.tsv", sep="\t", keep_default_na=False)
data_gwava18 = pd.read_csv("GWAVA_osu18.csv", sep="\t", keep_default_na=False)

data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="chrom", new_col_prefix="chrom", integrate=True)
print "chrom"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="majorAllele", new_col_prefix="majorAllele_", integrate=True)
print "majorAllele"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="minorAllele", new_col_prefix="minorAllele_", integrate=True)
print "minorAllele"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="ChromhmmGm12878", new_col_prefix="ChromhmmGm12878_", integrate=True)
print "ChromhmmGm12878"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="ChromhmmH1hesc", new_col_prefix="ChromhmmH1hesc_", integrate=True)
print "ChromhmmH1hesc"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="ChromhmmHelas3", new_col_prefix="ChromhmmHelas3_", integrate=True)
print "ChromhmmHelas3"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="ChromhmmHepg2", new_col_prefix="ChromhmmHepg2_", integrate=True)
print "ChromhmmHepg2"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="ChromhmmHuvec", new_col_prefix="ChromhmmHuvec_", integrate=True)
print "ChromhmmHuvec"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="ChromhmmK562", new_col_prefix="ChromhmmK562_", integrate=True)
print "ChromhmmK562"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="SegwayGm12878", new_col_prefix="SegwayGm12878_", integrate=True)
print "SegwayGm12878"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="SegwayH1hesc", new_col_prefix="SegwayH1hesc_", integrate=True)
print "SegwayH1hesc"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="SegwayHelas3", new_col_prefix="SegwayHelas3_", integrate=True)
print "SegwayHelas3"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="SegwayHepg2", new_col_prefix="SegwayHepg2_", integrate=True)
print "SegwayHepg2"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="SegwayHuvec", new_col_prefix="SegwayHuvec_", integrate=True)
print "SegwayHuvec"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="SegwayK562", new_col_prefix="SegwayK562_", integrate=True)
print "SegwayK562"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="pwm", new_col_prefix="pwm_", integrate=True)
print "pwm"
data_osu18 = one_hot_encode(dfm=data_osu18, cat_col_name="geneannot", new_col_prefix="geneannot_", integrate=True)
print "geneannot"
gc.collect()

#one-hot-encode GWAVA dataset
data_gwava18 = data_gwava18.drop(["start", "end"], axis=1)
data_gwava18 = one_hot_encode(dfm=data_gwava18, cat_col_name="chr", new_col_prefix="chrom", integrate=True)
print "chr"
data_gwava18.rename(columns={"cls": "label"}, inplace=True)
gc.collect()

# setup the double-index
coord_csnp = pd.read_csv("RSID_C_osu18.tsv", sep="\t")
coord_rsnp = pd.read_csv("RSID_R_osu18.tsv", sep="\t")
coord = pd.concat([coord_csnp, coord_rsnp])[["name", "chromStart"]]
coord.rename(columns={"chromStart": "coord"}, inplace=True)
data_osu18 = data_osu18.merge(coord, on="name")
data_osu18.set_index(["name", "coord"], inplace=True)
data_gwava18 = data_gwava18.merge(coord, on="name")
data_gwava18.set_index(["name", "coord"], inplace=True)

# setup all parameters
pos_neg_ratio = len(data_osu18[data_osu18["label"] == 1])*1.0 / len(data_osu18[data_osu18["label"] == 0])
sampling_method = "LOCUS"
kwargs={}
kwargs["cutoff_bp"] = 50000
kwargs["slope_allowed"] = 0.5
kwargs["seed"] = 1337
method_table = method_generate()
hp_table = hp_generate(pos_neg_ratio)
fold_table = fold_generate(data_osu18, n_rep=200, n_fold=5, fold_assign_method=sampling_method, **kwargs)
data_table = data_generate(["cerenkov17", "gwava_xgb", "gwava_rf"], [data_osu18, data_gwava18, data_gwava18], fold_assign_method=sampling_method, **kwargs)
task_table = task_pool(method_table, hp_table, data_table, fold_table)


################################
# machine learning
################################
s_time = time.time()
result_task_table = cerenkov_ml(task_table, method_table, fold_table, hp_table, data_table, fold_assign_method=sampling_method, ncpus=-1, feature_reduced=False)
e_time = time.time()
print result_task_table
print "cerenkov_ml TIME= ", e_time - s_time