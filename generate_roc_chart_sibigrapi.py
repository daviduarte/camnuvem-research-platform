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

# I3D
# ROC curve is (X,Y) = (FPR, TPR)
i3d_fpr_wsal_only_abnormal_path = "graficos/ucf/i3d/fpr_wsal_only_abnormal_ucf_crime.npy"
i3d_tpr_wsal_only_abnormal_path = "graficos/ucf/i3d/tpr_wsal_only_abnormal_ucf_crime.npy"

i3d_fpr_rtfm_only_abnormal_path = "graficos/ucf/i3d/fpr_rtfm_only_abnormal_ucf_crime.npy"
i3d_tpr_rtfm_only_abnormal_path = "graficos/ucf/i3d/tpr_rtfm_only_abnormal_ucf_crime.npy"

i3d_fpr_rads_only_abnormal_path = "graficos/ucf/i3d/fpr_ucf_crime_sultani_only_abnormal.npy"
i3d_tpr_rads_only_abnormal_path = "graficos/ucf/i3d/tpr_ucf_crime_sultani_only_abnormal.npy"

i3d_tpr1 = np.load(i3d_tpr_wsal_only_abnormal_path)
i3d_fpr1 = np.load(i3d_fpr_wsal_only_abnormal_path)

i3d_tpr2 = np.load(i3d_tpr_rtfm_only_abnormal_path)
i3d_fpr2 = np.load(i3d_fpr_rtfm_only_abnormal_path)

i3d_tpr3 = np.load(i3d_tpr_rads_only_abnormal_path)
i3d_fpr3 = np.load(i3d_fpr_rads_only_abnormal_path)


# I3D + SSOHC
# ROC curve is (X,Y) = (FPR, TPR)
i3d_ssohc_fpr_wsal_only_abnormal_path = "graficos/ucf/ssohc-i3d/fpr_wsal_only_abnormal_ucf_crime.npy"
i3d_ssohc_tpr_wsal_only_abnormal_path = "graficos/ucf/ssohc-i3d/tpr_wsal_only_abnormal_ucf_crime.npy"

i3d_ssohc_fpr_rtfm_only_abnormal_path = "graficos/ucf/ssohc-i3d/fpr_rtfm_only_abnormal_ucf_crime.npy"
i3d_ssohc_tpr_rtfm_only_abnormal_path = "graficos/ucf/ssohc-i3d/tpr_rtfm_only_abnormal_ucf_crime.npy"

i3d_ssohc_fpr_rads_only_abnormal_path = "graficos/ucf/ssohc-i3d/fpr_ucf_crime_sultani_only_abnormal.npy"
i3d_ssohc_tpr_rads_only_abnormal_path = "graficos/ucf/ssohc-i3d/tpr_ucf_crime_sultani_only_abnormal.npy"

i3d_ssohc_tpr1 = np.load(i3d_ssohc_tpr_wsal_only_abnormal_path)
i3d_ssohc_fpr1 = np.load(i3d_ssohc_fpr_wsal_only_abnormal_path)

i3d_ssohc_tpr2 = np.load(i3d_ssohc_tpr_rtfm_only_abnormal_path)
i3d_ssohc_fpr2 = np.load(i3d_ssohc_fpr_rtfm_only_abnormal_path)

i3d_ssohc_tpr3 = np.load(i3d_ssohc_tpr_rads_only_abnormal_path)
i3d_ssohc_fpr3 = np.load(i3d_ssohc_fpr_rads_only_abnormal_path)

plt.figure()

data1 = pd.DataFrame({'True Positive Rate' : i3d_tpr1, 'False Positive Rate': i3d_fpr1})
data2 = pd.DataFrame({'True Positive Rate' : i3d_tpr2, 'False Positive Rate': i3d_fpr2})
data3 = pd.DataFrame({'True Positive Rate' : i3d_tpr3, 'False Positive Rate': i3d_fpr3})

data4 = pd.DataFrame({'True Positive Rate' : i3d_ssohc_tpr1, 'False Positive Rate': i3d_ssohc_fpr1})
data5 = pd.DataFrame({'True Positive Rate' : i3d_ssohc_tpr2, 'False Positive Rate': i3d_ssohc_fpr2})
data6 = pd.DataFrame({'True Positive Rate' : i3d_ssohc_tpr3, 'False Positive Rate': i3d_ssohc_fpr3})

#sns.set_theme(style="ticks")
sns.set_context("paper", rc={'figure.figsize':(20.7,8.27),"font.size":8,"axes.titlesize":8,"axes.labelsize":8})   


ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data1, color='r', linewidth=2.5)
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data2, color='b', linewidth=2.5)
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data3, color='g', linewidth=2.5)
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data4, color='r', linestyle='dotted', linewidth=2.5)
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data5, color='b', linestyle='dotted', linewidth=2.5)
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data6, color='g', linestyle='dotted', linewidth=2.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))	# Range do eixo y apenas com inteiros


ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.legend(title='Methods', loc='lower right', labels=['WSAL I3D', 'RTFM I3D', 'RADS I3D', 'WSAL SSOC+I3D', 'RTFM SSOC+I3D', 'RADS SSOC+I3D'])

#fig = ax.get_figure()
#fig.savefig("roc_curve_only_abnormal.png") 
plt.savefig("roc_curve_only_abnormal_10c.png")





exit()
############################
#	TODO O DATASET
##############################

plt.figure()


fpr_wsal_only_abnormal_path = "fpr_wsal_camnuvem.npy"
tpr_wsal_only_abnormal_path = "tpr_wsal_camnuvem.npy"

fpr_rtfm_only_abnormal_path = "fpr_rtfm_camnuvem.npy"
tpr_rtfm_only_abnormal_path = "tpr_rtfm_camnuvem.npy"

fpr_rads_only_abnormal_path = "fpr_camnuvem_sultani.npy"
tpr_rads_only_abnormal_path = "tpr_camnuvem_sultani.npy"

tpr1 = np.load(tpr_wsal_only_abnormal_path)
fpr1 = np.load(fpr_wsal_only_abnormal_path)

tpr2 = np.load(tpr_rtfm_only_abnormal_path)
fpr2 = np.load(fpr_rtfm_only_abnormal_path)

tpr3 = np.load(tpr_rads_only_abnormal_path)
fpr3 = np.load(fpr_rads_only_abnormal_path)



data1 = pd.DataFrame({'True Positive Rate' : tpr1, 'False Positive Rate': fpr1})
data2 = pd.DataFrame({'True Positive Rate' : tpr2, 'False Positive Rate': fpr2})
data3 = pd.DataFrame({'True Positive Rate' : tpr3, 'False Positive Rate': fpr3})

#sns.set_theme(style="ticks")
sns.set_context("paper", rc={'figure.figsize':(20.7,8.27),"font.size":8,"axes.titlesize":8,"axes.labelsize":8})   


ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data1, color='r')
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data2, color='b')
ax = sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=data3, color='g')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))	# Range do eixo y apenas com inteiros

plt.legend(title='Methods', loc='upper left', labels=['WSAL', 'RTFM', 'RADS'])

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#fig = ax.get_figure()
#fig.savefig("roc_curve_all_dataset.png") 
plt.savefig("roc_curve_all_dataset_10c.png")