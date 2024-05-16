from sklearn.tree import DecisionTreeClassifier

from ppelib.classifiers import RandomOracle, RandomOracle_Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import scipy.stats as stats

dataDir = 'D:\\Projects\\DataMining\\Data\\CSV_Experiments\\Above_1k\\'
datasets = [

    # "Agrawal1",
    # "Stagger1", #100% dokładności
    # "BayesianNetworkGenerator_spambase",
    # "BNG_sonar",
    # (dataDirLarge,"codrnaNorm"),
    # (dataDirLarge,"electricity-normalized"),
    # (dataDirLarge,"covtype"),
    # (dataDirLarge,"php89ntbG"),

    (dataDir, "banana"),
    (dataDir, "coil2000"),
    #(dataDir, "magic"),
    (dataDir, "phoneme"),
    (dataDir, "ring"),
    (dataDir, "spambase"),
    (dataDir, "twonorm"),

    # "shuttle2"
]

res = []
tts = {}
iter = 3
for dirName,fName in datasets:
    df = pd.read_csv(dirName + fName + "\\" + fName + ".dat", sep=";")
    X = df[[column for column in df.columns if column not in ["LABEL", "id"]]]
    y = np.squeeze(df[['LABEL']].values)
    print("Start prediction")
    models = [
                ("VRO", RandomOracle_Classifier(RandomOracle(base_estimator=DecisionTreeClassifier(max_features="sqrt"), min_size=50),n_estimators=100, n_jobs=1)),
              #("RO ", RandomOracle(base_estimator=RandomForestClassifier(n_jobs=5),min_size=100)),
                ("RF ",RandomForestClassifier(n_jobs=5)),

               ]
    accs = []
    for n,m in models:
        #m = RandomOracle_Classifier(RandomOracle(base_estimator=RandomForestClassifier()), n_estimators=20)
        acc = cross_val_score(m, X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42))
        print(f"Model: {n}, Accuracy: {acc.mean() * 100:.4f}% +- {acc.std() * 100:.4f}%")
        accs.append(acc)
        res.append( { 'dataset':fName,
                      'model':n,
                      'accuracy': acc.mean() * 100,
                      'std': acc.std() * 100,
                      }
                    )
    d1 = accs[0]
    pvs = []
    for i in range(1,len(accs)):
        d2 = accs[i]
        tt = stats.ttest_rel(d1, d2)
        pvs.append((tt.pvalue,tt.pvalue<0.05))
    tts[fName] = pvs

df_res = pd.DataFrame(res)
df_tt = pd.DataFrame(tts)
df_res.to_excel(f'Data/Results/random_oracle_{iter}.xlsx')
df_tt.to_excel(f'Data/Results/random_oracle_tt_{iter}.xlsx')
