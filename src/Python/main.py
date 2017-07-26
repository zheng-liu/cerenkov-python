# main function for base.py
from base import *

data_osu = pandas.read_csv("features_OSU.tsv", sep="\t")
coord = pandas.read_csv("coord.txt", sep="\t")
data_osu.index = coord.index # //TODO to ensure data_osu is already the "SNP ID secure" version
data_osu["coord"] = coord.coord
data_osu.index.name = "rsid"
data_osu.set_index("coord", append=True, inplace=True)

start_time = time.time()
fold = locus_sampling(data_osu, n_rep=200, n_fold=5, cutoff_bp=50000, slope_allowed=0.5, seed=1337)
end_time = time.time()
print fold
print "time=", end_time - start_time