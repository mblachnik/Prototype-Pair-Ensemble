import pandas as pd
import matplotlib.pyplot as plt
import os

def collectData(directory):
    offFiles = ["BayesianNetworkGenerator_spambase"]
    dd = os.walk(directory)
    ignoreFirsttRow = True
    resultsFinal = {}
    for dName,subDirs,fNames in dd:
        # print(dName)
        # print(subDirs)
        # print(fNames)
        if ignoreFirsttRow:
            ignoreFirsttRow = False
            continue
        fNames = [fName for fName in fNames if "CV" not in fName]
        results = []
        ts = [True for f in offFiles if f in dName]
        if len(ts)>0 and ts[0]: continue
        for fName in fNames:
            if "NonEnsemble" not in fName:
                continue 
            print(fName)
            df = pd.read_csv(dName + "\\" + fName,sep=";")
            if ("NonEnsemble" not in fName) and  all([ False if id in directory else True for id in ["14","15"]]):
                df2 = pd.read_csv(dName + "\\" + fName[:-4] + "_CV_size_log.csv", sep="	",skiprows=[0])
                df2d = df2.describe()
                df["ProtoN"] = df2d.loc["count",'# size']
                df["FullSiz"] = df2d.loc["mean",'mainSize']
                df["TrainSize"] = df2d.loc["mean",'# size']
            results.append(df)
        resultsT = pd.concat(results)
        dataName = dName.split("\\")[-1]
        resultsFinal[dataName] = resultsT.describe()
    return resultsFinal

res = []
# process_id_main = '8'
# process_id_ref = '15'
SAMPLE_NAME = "PPE2_ICF2"
path = f"Data\\Results\\{SAMPLE_NAME}\\"
resultsRef  = collectData(path)
print(resultsRef)
for key in resultsRef.keys():
    res.append({
        "Dataset":key,
    "ACC Ref": resultsRef[key].loc["mean", "average(ACC)"],
    # "ACC Main": resultsBase[key].loc["mean", "average(ACC)"],
    "ACC STD Ref": resultsRef[key].loc["std", "average(ACC)"],
    # "ACC STD Main": resultsBase[key].loc["std", "average(ACC)"],
    "BACC Ref": resultsRef[key].loc["mean", "average(BACC)"],
    # "BACC Main": resultsBase[key].loc["mean", "average(BACC)"],
    "BACC STD Ref": resultsRef[key].loc["std", "average(BACC)"],
    # "BACC STD Main": resultsBase[key].loc["std", "average(BACC)"],
    "time Train Ref":resultsRef[key].loc["mean", "average(ModelOptimizationExecutionTime)"],
    # "time Train Main": resultsBase[key].loc["mean", "average(ModelOptimizationExecutionTime)"],
    "time Test Ref": resultsRef[key].loc["mean", "average(ModelPredictionTime)"],
    # "time Test Main": resultsBase[key].loc["mean", "average(ModelPredictionTime)"],
    "# samples Batch Ref": resultsRef[key].loc["mean", 'TrainSize'] if 'TrainSize' in resultsRef[key].columns else None,
    # "# samples Batch Main": resultsBase[key].loc["mean", 'TrainSize'] if 'TrainSize' in resultsBase[key].columns else None,
    "# proto Ref": resultsRef[key].loc["mean", 'ProtoN'] if 'ProtoN' in resultsRef[key].columns else None,
    # "# proto Main": resultsBase[key].loc["mean", 'ProtoN'] if 'ProtoN' in resultsBase[key].columns else None,
    "# train Ref": resultsRef[key].loc["mean", 'FullSiz'] if 'FullSiz' in resultsRef[key].columns else None,
    # "# train Main": resultsBase[key].loc["mean", 'FullSiz'] if 'FullSiz' in resultsBase[key].columns else None,
    "# dataset_size": resultsRef[key].loc["mean", 'dataset_size'] if 'dataset_size' in resultsRef[key].columns else None,
    "# selected_dataset_size": resultsRef[key].loc["mean", 'selected_dataset_size'] if 'selected_dataset_size' in resultsRef[key].columns else None,
    "# selection_time": resultsRef[key].loc["mean", 'selection_time'] if 'selection_time' in resultsRef[key].columns else None,
    "# ppe_init_time": resultsRef[key].loc["mean", 'ppe_init_time'] if 'ppe_init_time' in resultsRef[key].columns else None,
    "# ppe_region_time": resultsRef[key].loc["mean", 'ppe_region_time'] if 'ppe_region_time' in resultsRef[key].columns else None,
            })

path = "Data\\Results\\IS2"
wyniki = f"wyniki_{SAMPLE_NAME}"
wyn = pd.DataFrame(res)
# print(wyn)
# wyn["ACC Main/Ref"] = (wyn["ACC Main"]/wyn["ACC Ref"])
# wyn["BACC Main/Ref"] = (wyn["BACC Main"]/wyn["BACC Ref"])
# wyn["time Train Main/Ref"] = (wyn["time Train Main"]/wyn["time Train Ref"])
# wyn["time Test Main/Ref"] = (wyn["time Test Main"]/wyn["time Test Ref"])
print(f"{path}\\{wyniki}.csv")
wyn.to_csv(f"{path}\\{wyniki}.csv")
wyn.to_excel(f"{path}\\{wyniki}.xlsx")






