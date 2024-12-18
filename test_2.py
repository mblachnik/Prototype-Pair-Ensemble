import pandas as pd
import numpy as np
# from cascade_svm import CascadeSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import time
import os
from ppelib import classifiers as  ppe
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from ppelib.sampler.kmeans_sampler import SimpleClusterCentroids
from imblearn.under_sampling import ClusterCentroids
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearnex import patch_sklearn
from ppelib.lvqClusters import lvqClusters
from sklearn.neighbors import KNeighborsClassifier
from src.main.python.iSel.drop3 import DROP3
from src.main.python.iSel.icf import ICF
from src.main.python.iSel.ldis import LDIS
from src.main.python.iSel.cdis import CDIS
from src.main.python.iSel.xldis import XLDIS
from src.main.python.iSel.lssm import LSSm
from src.main.python.iSel.egdis import EGDIS
from src.main.python.iSel.base import InstanceSelectionMixin
from ppelib import resampler as rppe

def get_selector(selector:str)->InstanceSelectionMixin:
    if selector == "ICF":
        return ICF()
    if selector == "LDIS":
        return LDIS()
    if selector == "CDIS":
        return CDIS()
    if selector == "XLDIS":
        return XLDIS()
    if selector == "LSSm":
        return LSSm()
    if selector == "EGDIS":
        return EGDIS()
    

# patch_sklearn()
# folder = 'Y:\\Datasets\\datasets_ppe\\'
# folder = "Y:\Datasets\Datasets\KeelNormCV"
# folder = "Y:\Datasets\Datasets\MetaIS\corrected\Filtered by CCIS"
#folder = "Y:\Datasets\Datasets\KeelNorm"
folder = "Data"
#"D:\\mblachnik\\datasets\\Datasets\\KeelNormCV"
#'D:\\mblachnik\\datasets\\large'
resFolder = "Data/Results/"
files = [
    # ["Agrawal1",100],#9e5
    ["Banana",100],#5e3
    # ["banana",500],#5e3
    # ["twonorm",500],#6,6e3
    # ["spambase",500],#4e3
    # ["phoneme",500],#5e3
    # ["ring",500],#6,6e3
    # ["coil2000",500],#8,8e3
    # ["magic",500],#1,7e4
    # ["electricity-normalized",1500],#4e4
    # ["shuttle2",100],#5,2e4
    # ["codrnaNorm",100],#4,4e5
    # ["covtype",100],#5,3e5
    # ["php89ntbG",100],#4,4e5
    # ["Stagger1",100],#?
    # ["BayesianNetworkGenerator_spambase",100],#?
    # ["BNG_sonar",100],#9e5
    # ["titanic",100],#?
]



MAX_DEPTH = 8
RANDOM_STATE = 42
fold_number = 10
res = {}
n_clusters = 10
prune_regions = True
PPE2 = True

sel = ["ICF", "LDIS", "CDIS", "XLDIS", "LSSm", "EGDIS"]


for selector in sel:
    mname = f"{selector}"
    if PPE2:
        mname = "PPE2_" + mname
    models_names = [mname]
    print(mname)
    for model_name in models_names:
        dName = f"{resFolder}\\{model_name}"
        if not os.path.isdir(dName):
            os.mkdir(f"{resFolder}\\{model_name}")
        for f_data in files:
            try:
                file=f_data[0]
                print(f"#### {file} ####")
                dName = f"{resFolder}\\{model_name}\\{file}"
                if not os.path.isdir(dName):
                    os.mkdir(f"{resFolder}\\{model_name}\\{file}")
                for i in range(1, fold_number +1):
                    #tr = pd.read_csv(f"{folder}\\{file}\\{file}-{fold_number}-{i}tra.dat",sep=";")
                    #te = pd.read_csv(f"{folder}\\{file}\\{file}-{fold_number}-{i}tst.dat",sep=";")
                    tr = pd.read_csv(f"{folder}\\{file}.csv",sep=";")
                    te = pd.read_csv(f"{folder}\\{file}.csv",sep=";")

                    cols = [col for col in tr.columns if col not in ["LABEL", "id"]]
                    Xtr = tr.loc[:, cols].values
                    ytr = tr.loc[:, "LABEL"].values
            
                    Xte = te.loc[:, cols].values
                    yte = te.loc[:, "LABEL"].values
            
                    ohe = LabelEncoder()
                    ytr = ohe.fit_transform(ytr)
                    yte = ohe.fit_transform(yte)
            
                    instance_selection = get_selector(selector)
                    selection_start = time.time()
                    resampler = rppe.ppeResample(instance_selection,
                                                ppe_type="ppe2",
                                                min_support=f_data[1],
                                                proto_selection=ClusterCentroids(sampling_strategy={0: n_clusters, 1: n_clusters}),
                                                prune_regions=prune_regions,
                                                unbalanced_rate=0.2,
                                                minimum_regions=2,
                                                n_jobs=10)
                    ## ----IS----- ##
                    if PPE2:
                        sXtr, sytr = resampler.fit_resample(Xtr,ytr)
                    else:
                        instance_selection.fit(Xtr,ytr)
                        idx = instance_selection.sample_indices_
                        sXtr, sytr =  Xtr[idx], ytr[idx]
                    ## ----RESAMPLER----- ##
                    ## --- ##
                    selection_stop = time.time()
                    model = KNeighborsClassifier(n_neighbors=1)
                    t0 = time.time()
                    model.fit(sXtr, sytr)
                    t1 = time.time()
                    yte_p = model.predict(Xte)
                    t2 = time.time()
                    
                    res = {"average(ModelOptimizationExecutionTime)": [],
                        "average(ModelPredictionTime)": [],
                        "average(ACC)": [],
                        "average(BACC)": [],
                        "standard_deviation(ModelOptimizationExecutionTime)": [],
                        "standard_deviation(ModelPredictionTime)": [],
                        "standard_deviation(ACC)": [],
                        "standard_deviation(BACC)": [],
                        "n_regions":[],
                        "max_regions":[],
                        "min_regions":[],
                        "dataset_size":[],
                        "selected_dataset_size":[],
                        "selection_time": []}
            
                    res["average(ModelOptimizationExecutionTime)"].append(t1-t0)
                    res["average(ModelPredictionTime)"].append(t2-t1)
                    res["average(ACC)"].append(accuracy_score(yte, yte_p))
                    res["average(BACC)"].append(balanced_accuracy_score(yte, yte_p))
                    res["standard_deviation(ModelOptimizationExecutionTime)"].append(0)
                    res["standard_deviation(ModelPredictionTime)"].append(0)
                    res["standard_deviation(ACC)"].append(0)
                    res["standard_deviation(BACC)"].append(0)
                    res["ppe_init_time"] = resampler.ppe_init_time
                    res["ppe_region_time"] = resampler.ppe_region_time
                    res["ppe_fit_time"] = resampler.ppe_fit_time
                    res["n_regions"] = 0
                    res["max_regions"] = 0
                    res["min_regions"] = 0
                    try:
                        res["n_regions"] = len(model.region_stats)
                        res["max_regions"] = model.region_stats["Vectors"].iloc[-1]
                        res["min_regions"] = model.region_stats["Vectors"].iloc[0]
                    except:pass
                    res["dataset_size"].append(len(ytr))
                    res["selected_dataset_size"].append(len(sytr))
                    res["selection_time"].append(selection_stop - selection_start)

                    resdf = pd.DataFrame(res)
            
                    resdf.to_csv(f"{resFolder}\\{model_name}\\{file}-{i}.dat.log",index=False,sep=";")
                    # res_stat = pd.DataFrame(resampler.region_stats)
                    # res_stat.to_csv(f"{resFolder}\\{model_name}\\{file}\\{file}-{fold_number}-{i}tra.dat_CSVM_NonEnsemble_region_stat.log",index=False,sep=";")
            except Exception as e:
                print(e)
