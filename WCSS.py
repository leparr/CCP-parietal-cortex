#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:04:13 2023

@author: lparrilla
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from sklearn.cluster import KMeans

folder_path = '/home/lparrilla/Documents/Dynap_simulation/clustering_1'
file_list = os.listdir(folder_path)
WCSS = []
perc_drop = []
# LOADING FILES
for file_name in file_list:
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)   
        data = data.reshape(int(len (data)/450),450)
        scaler = StandardScaler()
        segmentation_std = scaler.fit_transform(data.transpose())
        pca = PCA(n_components = 3)
        pca.fit(segmentation_std)
        scores_pca = pca.transform (segmentation_std)
        # CALCULATING WCSS SCORES
        wcss = []
        for i in range (1,21):
            kmeans_pca = KMeans (n_clusters = i, init ='k-means++', random_state = 42)
            kmeans_pca.fit (scores_pca)
            wcss.append (kmeans_pca.inertia_)
        WCSS.append (wcss)
        # CALCULATING PERCENTAGE DROP WCSS ITERATIONS
        percentage_decrease = []
        for i in range(len(wcss) - 1):
            decrease = wcss[i] - wcss[i + 1]
            percentage = (decrease / wcss[i]) * 100
            percentage_decrease.append(percentage)
        y = percentage_decrease
        perc_drop.append (y)

# WCSS SCORES PLOT
plt.figure (figsize = (10,8))
plt.plot (WCSS[2], 'blue', label = 'FF M1', marker='o', markersize = 6, lw = 3)
plt.plot (WCSS[3], 'red', label = 'FF M2', marker='o', markersize = 6, lw = 3)

plt.plot (WCSS[1], 'blue', label = 'HR M1', marker='o', ls='dashdot', markersize = 6, lw = 3)
plt.plot (WCSS[4], 'red', label = 'HR M2', marker='o', ls='dashdot', markersize = 6, lw = 3)

plt.plot (WCSS[0], 'blue', label = 'REC M1', marker='o', ls='dotted', markersize = 6, lw = 3)
plt.plot (WCSS[5], 'red', label = 'REC M2', marker='o', ls='dotted', markersize = 6, lw = 3)

lines = []
lines.append(plt.Line2D([], [], color='blue', marker='o', markersize=6, linewidth=3))
lines.append(plt.Line2D([], [], color='red', marker='o', markersize=6, linewidth=3))
lines.append(plt.Line2D([], [], color='blue', marker='o', markersize=6, ls='dashdot',linewidth=3))
lines.append(plt.Line2D([], [], color='red', marker='o', markersize=6, ls='dashdot',linewidth=3))
lines.append(plt.Line2D([], [], color='blue', marker='o', markersize=6, ls='dotted',linewidth=3))
lines.append(plt.Line2D([], [], color='red', marker='o', markersize=6, ls='dotted',linewidth=3))

labels = ['FF M1', 'FF M2', 'HR M1', 'HR M2', 'REC M1', 'REC M2']
legend = plt.legend(handles=lines, labels=labels, fontsize=20)
for legline in legend.get_lines():
    legline.set_markersize(15)  
    legline.set_linewidth(4)  
labels = ['', '0', '', '', '', '','','','','40000']
plt.gca().set_yticklabels(labels,fontsize=20)
x_ticks = np.array(range(len(WCSS[0])))
even_x_ticks = x_ticks[x_ticks % 2 == 0]
plt.xticks(even_x_ticks, even_x_ticks, fontsize=18)
plt.xlabel('Number of clusters', fontsize = 30, labelpad=5)
plt.ylabel('WCSS score', fontsize = 30, labelpad = -40)
plt.show()

# PERCENTAGE DROP WCSS PLOT

plt.figure (figsize = (10,8))
plt.plot (perc_drop[2], 'blue', label = 'FF M1', marker='o', markersize = 6, lw = 3)
plt.plot (perc_drop[3], 'red', label = 'FF M2', marker='o', markersize = 6, lw = 3)

plt.plot (perc_drop[1], 'blue', label = 'HR M1', marker='o', ls='dashdot', markersize = 6, lw = 3)
plt.plot (perc_drop[4], 'red', label = 'HR M2', marker='o', ls='dashdot', markersize = 6, lw = 3)

plt.plot (perc_drop[0], 'blue', label = 'REC M1', marker='o', ls='dotted', markersize = 6, lw = 3)
plt.plot (perc_drop[5], 'red', label = 'REC M2', marker='o', ls='dotted', markersize = 6, lw = 3)

lines = []
lines.append(plt.Line2D([], [], color='blue', marker='o', markersize=6, linewidth=3))
lines.append(plt.Line2D([], [], color='red', marker='o', markersize=6, linewidth=3))
lines.append(plt.Line2D([], [], color='blue', marker='o', markersize=6, ls='dashdot',linewidth=3))
lines.append(plt.Line2D([], [], color='red', marker='o', markersize=6, ls='dashdot',linewidth=3))
lines.append(plt.Line2D([], [], color='blue', marker='o', markersize=6, ls='dotted',linewidth=3))
lines.append(plt.Line2D([], [], color='red', marker='o', markersize=6, ls='dotted',linewidth=3))

labels = ['FF M1', 'FF M2', 'HR M1', 'HR M2', 'REC M1', 'REC M2']
legend = plt.legend(handles=lines, labels=labels, fontsize=20)
for legline in legend.get_lines():
    legline.set_markersize(15)  
    legline.set_linewidth(4)  
    
x_ticks = np.array(range(len(WCSS[0])))
even_x_ticks = x_ticks[x_ticks % 2 == 0]
plt.xticks(even_x_ticks, even_x_ticks, fontsize=18)
plt.yticks (fontsize=18)
plt.xlabel('Number of iterations', fontsize = 30, labelpad=5)
plt.ylabel('Drop in WCSS [%]', fontsize = 30)
plt.show()
