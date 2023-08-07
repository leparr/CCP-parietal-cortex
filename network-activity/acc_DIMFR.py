"""

@author: lparrilla
"""


# =============================================================================
# ACCURACIES
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

wta_25 = np.load ('mean_acc_wta_25')
wta_24 = np.load ('mean_acc_wta_24')
ff_25 = np.load ('mean_acc_ff_25')
ff_24 = np.load ('mean_acc_ff_24')
control_25 = np.load ('mean_acc_control_25')
control_24 = np.load ('mean_acc_control_24')

wta_25_std = np.load ('std_wta_25')
wta_24_std = np.load ('std_wta_24')
ff_25_std = np.load ('std_wta_25')
ff_24_std = np.load ('std_wta_24')
control_25_std = np.load ('std_wta_25')
control_24_std = np.load ('std_wta_24')
x = [1, 2, 3]
plt.errorbar(x, wta_25, yerr=wta_25_std, color='r', lw=2, marker='o', markersize=10, label='Stacked learning WTA M1', capsize=5)
plt.errorbar(x, wta_24, yerr=wta_24_std, color='b', lw=2, marker='o', markersize=10, label='Stacked learning WTA M2', capsize=5)
plt.errorbar(x, ff_25, yerr=ff_25_std, color='r', lw=2, marker='s', markersize=10, label='Feedforward M1', capsize=5)
plt.errorbar(x, ff_24, yerr=ff_24_std, color='b', lw=2, marker='s', markersize=10, label='Feedforward M2', capsize=5)
plt.errorbar(x, control_25, yerr=control_25_std, color='r', lw=2, marker='v', markersize=10, label='Control M1', capsize=5)
plt.errorbar(x, control_24, yerr=control_24_std, color='b', lw=2, marker='v', markersize=10, label='Control M2', capsize=5)
legend_handles = [
    plt.Line2D([], [], color='r', lw=4),
    plt.Line2D([], [], color='b', lw=4),
    plt.Line2D([], [], color='k', marker='o', markersize=15, linestyle='None'),
    plt.Line2D([], [], color='k', marker='v', markersize=15, linestyle='None'),
    plt.Line2D([], [], color='k', marker='s', markersize=15, linestyle='None'),
]
legend_labels = [
    'M1',
    'M2',
    'Stacked learning WTA',
    'Feedforward',
    'Control',
]
plt.legend(handles=legend_handles, labels=legend_labels, fontsize=20, bbox_to_anchor=(0.1, 0.1))
plt.ylabel('Accuracy [%]', fontsize=30)
plt.xlabel('Behavioral epochs', fontsize=30)
plt.yticks(fontsize=20)
plt.xticks([1, 2, 3], ['1', '2', '3'], fontsize=20)
plt.rcParams['lines.linewidth'] = 1
plt.show()

# =============================================================================
# DIMFR
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

dimfr_wta_ep1 = np.load ('dimfr_wta_ep1')
dimfr_wta_ep2 = np.load ('dimfr_wta_ep2')
dimfr_wta_ep3 = np.load ('dimfr_wta_ep3')
dimfr_ff_ep1 = np.load ('dimfr_ff_ep1')
dimfr_ff_ep2 = np.load ('dimfr_ff_ep2')
dimfr_ff_ep3 = np.load ('dimfr_ff_ep3')

fig, ax = plt.subplots()
boxprops = dict(linestyle='-',linewidth=1, color='black')
medianprops = dict(linestyle='-', linewidth=1, color='black')
meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='purple')
whiskerprops = dict(linestyle='--', linewidth=1, color='black')
capprops = dict(linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='o', markersize=8)
bp = ax.boxplot([dimfr_wta_ep1, dimfr_wta_ep2, dimfr_wta_ep3], labels=['1', '2', '3'], showmeans=False, meanline=True, widths=0.8, boxprops=boxprops, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
bp2 = ax.boxplot([dimfr_ff_ep1, dimfr_ff_ep2, dimfr_ff_ep3], labels=['1', '2', '3'], showmeans=False, meanline=True, widths=0.8, boxprops=boxprops, medianprops=medianprops, meanprops=meanprops, whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
colors = ['orange', 'blue']
for i, data in enumerate([dimfr_wta_ep1, dimfr_wta_ep2, dimfr_wta_ep3]):
    for j in range(5):
        ax.scatter(i+1, data[j], color=colors[0], s=100, edgecolors=colors[0], linewidths=1.5, zorder=3)
        ax.scatter(i+1, data[j+5], color=colors[1], s=100, facecolors='white', edgecolors=colors[0], linewidths=2, zorder=3)
for i, data in enumerate([dimfr_ff_ep1, dimfr_ff_ep2, dimfr_ff_ep3]):
    for j in range(5):
        ax.scatter(i+1, data[j], color=colors[1], s=100, edgecolors=colors[1], linewidths=1.5, zorder=3)
        ax.scatter(i+1, data[j+5], color=colors[1], s=100, facecolors='white', edgecolors=colors[1], linewidths=2, zorder=3)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='k', markersize=15, label='M1',linestyle='None'),
    plt.Line2D([0], [0], marker='o', color='k', markerfacecolor='w', markersize=15, label='M2',linestyle='None'),
    plt.Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', markersize=15, label='FF',linestyle='None'),
    plt.Line2D([0], [0], marker='o', color='orange', markerfacecolor='orange', markersize=15, label='WTA',linestyle='None'),
]
ax.legend(handles=legend_elements, fontsize = 20)
ax.set_xlabel('Decoding epochs', fontsize = 30)
ax.set_ylabel('Mean firing difference [%]', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.show()
