import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from  matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
import cv2


############################
#	SOMENTE OS VÍDEOS ANÕMALOS
##############################

# ROC curve is (X,Y) = (FPR, TPR)
fpr_wsal_only_abnormal_path = "fpr_wsal_only_abnormal.npy"
tpr_wsal_only_abnormal_path = "tpr_wsal_only_abnormal.npy"

fpr_rtfm_only_abnormal_path = "fpr_rtfm_only_abnormal.npy"
tpr_rtfm_only_abnormal_path = "tpr_rtfm_only_abnormal.npy"

tpr1 = np.load(tpr_wsal_only_abnormal_path)
fpr1 = np.load(fpr_wsal_only_abnormal_path)

tpr2 = np.load(tpr_rtfm_only_abnormal_path)
fpr2 = np.load(fpr_rtfm_only_abnormal_path)


plt.figure()

data1 = pd.DataFrame({'True Positive Rate' : tpr1, 'False Positive Rate': fpr1})
data2 = pd.DataFrame({'True Positive Rate' : tpr2, 'False Positive Rate': fpr2})

#sns.set_theme(style="ticks")
sns.set_context("paper", rc={'figure.figsize':(20.7,8.27),"font.size":8,"axes.titlesize":8,"axes.labelsize":8})   


ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data1, color='r')
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data2, color='b')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))	# Range do eixo y apenas com inteiros


ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.legend(title='Methods', loc='upper left', labels=['WSAL', 'RTFM'])

#fig = ax.get_figure()
#fig.savefig("roc_curve_only_abnormal.png") 
plt.savefig("roc_curve_only_abnormal.png")

############################
#	TODO O DATASET
##############################

plt.figure()


fpr_wsal_only_abnormal_path = "fpr_wsal.npy"
tpr_wsal_only_abnormal_path = "tpr_wsal.npy"

fpr_rtfm_only_abnormal_path = "fpr_rtfm.npy"
tpr_rtfm_only_abnormal_path = "tpr_rtfm.npy"

tpr1 = np.load(tpr_wsal_only_abnormal_path)
fpr1 = np.load(fpr_wsal_only_abnormal_path)

tpr2 = np.load(tpr_rtfm_only_abnormal_path)
fpr2 = np.load(fpr_rtfm_only_abnormal_path)



data1 = pd.DataFrame({'True Positive Rate' : tpr1, 'False Positive Rate': fpr1})
data2 = pd.DataFrame({'True Positive Rate' : tpr2, 'False Positive Rate': fpr2})

#sns.set_theme(style="ticks")
sns.set_context("paper", rc={'figure.figsize':(20.7,8.27),"font.size":8,"axes.titlesize":8,"axes.labelsize":8})   


ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data1, color='r')
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data2, color='b')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))	# Range do eixo y apenas com inteiros

plt.legend(title='Methods', loc='upper left', labels=['WSAL', 'RTFM'])

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#fig = ax.get_figure()
#fig.savefig("roc_curve_all_dataset.png") 
plt.savefig("roc_curve_all_dataset.png")