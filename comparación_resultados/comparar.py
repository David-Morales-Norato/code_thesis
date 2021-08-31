import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dataset = "mnist"
df_none = pd.read_csv("data_"+dataset+"_none.csv").iloc[:30]
df_back = pd.read_csv("data_"+dataset+"_back.csv").iloc[:30]
df_fsi = pd.read_csv("data_"+dataset+"_fsi.csv").iloc[:30]

df_none.columns = [col + "_none" for col in df_none.columns]
df_back.columns = [col + "_back" for col in df_back.columns]
df_fsi.columns = [col + "_fsi" for col in df_fsi.columns]

cols = ['test_binary_accuracy', 'test_precision_1', 'test_recall_1', 'test_f1_score']

results = "results_"+dataset

if not os.path.exists(results):
    os.mkdir(results)

pd_comp = pd.concat([df_none, df_back, df_fsi], axis=1)
print("Dataset: ", dataset)
for col in cols:
    print(col)
    print("none:", pd_comp[[col+"_none"]].values[-1],"back:", pd_comp[[col+"_back"]].values[-1],"fsi:", pd_comp[[col+"_fsi"]].values[-1])
    ax = pd_comp.plot(y = [col+"_none", col+"_back", col+"_fsi"])
    ax.figure.savefig(os.path.join(results, col))
    fig = ax.get_figure(); plt.close(fig)



