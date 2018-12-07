import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from draw_cm import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

path = 'csv/val_res_2nd.csv'

class_names = ['OK','colorbeam','ghost','innerDust','largerLight','longbeam','mulitibeam','shorbeam']
def get_confusionmatrix(path):
    df = pd.read_csv(path)

    class_names = list(df.columns.values)[1:]

    cm = np.array(df[class_names].values.tolist())
    return cm,

def get_cm2(path):
    df = pd.read_csv(path)
    label = list(df.true)
    pred = list(df.pred)
    return label,pred

# get_cm2(path)

# Compute confusion matrix
# cnf_matrix,class_names = get_confusionmatrix(path)
label,pred = get_cm2(path)

print(label)
print(pred)
cnf_matrix = confusion_matrix(label,pred,list(np.arange(0,8)))
print(cnf_matrix)
# cnf_matrix = np.where(cnf_matrix=='nan',cnf_matrix,0)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='confusion matrix')

plt.show()