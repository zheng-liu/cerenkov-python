'''

dependencies:
    progressbar
    scikit-learn-?
    xgboost-?
    pandas-?
    aws ecs?

//TODO check which versions are those softwares

'''

from cerenkov_ml_base import *

params = {
    "num_folds_cross_validation": 10,
    "num_cv_replications": 2,
    "flag_create_fork_cluster": False,  ## for fig. 3:  TRUE
    "override_num_fork_processes": 50,  ## for EC2, set to 64; for my MBP, set to 8
    "notify_by_text_msg": False,
    "show_progress_bar": True,
    "flag_locus_sampling": True,        ## set to false if you want SNP-level sampling
    "flag_xgb_importance": False,       ## if you set this to TRUE, make sure you set num_cv_replications=1 and num_folds_cross_validation=1
    "random_number_seed": 1337,
    "nthreads_per_process": 1,
    "flag_randomize_classifier_order": False,
    "flag_create_ec2_instances": False  ## don't set this to true if you set "flag_create_fork_cluster" to true
}

# //TODO omit the msg_test details "us-west-2:315280700912:ramseylab" and document here
params["aws_snps_topic_arn"] = "arn:aws:sns:us-west-2:315280700912:ramseylab" if params["notify_by_text_msg"] else ""

# //TODO add EC2 configuration code here


# set random seed
random.seed(params["random_number_seed"])

# read data
print "---------- load OSU data ----------"
osu_feature = pd.read_csv("XXXXXX.txt") # //TODO update the data file name
feature_check(osu_feature) # check NaN in feature matrix


