"""

@author: lparrilla
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import seaborn as sns
from sklearn.cluster import KMeans

def PCA_plot (folder_path):
    '''
    Method to plot learned plastic weights onto their first 3 Principal Components
    
    Parameters :
        folder_path : folder where plastic weights are saved
    
    Returns :
        the plots of the learned weights onto first 3 principal components
    '''
    
    file_list = os.listdir(folder_path)
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
            test = scores_pca.transpose()
            comp1 = test [0]
            comp2 = test [1]
            comp3 = test [2]
            num_groups = 9
            group_size = 50
            colors = np.repeat(np.arange(num_groups), group_size)
            fig = plt.figure(figsize = (9,8))
            ax = fig.add_subplot(111, projection='3d')
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='z', labelsize=20)
            ax.set_xlabel('PC 1', fontsize=25, labelpad=20)
            ax.set_ylabel('PC 2', fontsize=25, labelpad=20)
            ax.set_zlabel('PC 3', fontsize=25, labelpad=20)
            ax.scatter(comp1, comp2, comp3, c=colors, cmap='rainbow', edgecolor='k')
            ax.set_title(file_name)

def plot_IV_CV (folder_path):
    '''
    Method to plot individual and cumulative variance explained onto different principal components for the loaded plastic weights
    
    Parameters :
        folder_path : folder from where to load the plastic weights
    
    Returns :
        the plot of indiviudal and cumulative vairance explained onto first 10 principal components of the loaded weights
    '''
    
    file_list = os.listdir(folder_path)
    sorted_file_names = sorted(file_list)
    PCA_scores = []
    
    for file_name in sorted_file_names:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            data = data.reshape(int(len(data) / 450), 450)
            pca = PCA(n_components=10)
            pca.fit(data)
            variance_explained = pca.explained_variance_ratio_
            PCA_scores.append(variance_explained)
    variance_explained_dataset1 = PCA_scores[0]
    variance_explained_dataset2 = PCA_scores[1]
    variance_explained_dataset3 = PCA_scores[2]
    variance_explained_dataset4 = PCA_scores[3]
    variance_explained_dataset5 = PCA_scores[4]
    variance_explained_dataset6 = PCA_scores[5]
    cumulative_variance_dataset1 = np.cumsum(variance_explained_dataset1)
    cumulative_variance_dataset2 = np.cumsum(variance_explained_dataset2)
    cumulative_variance_dataset3 = np.cumsum(variance_explained_dataset3)
    cumulative_variance_dataset4 = np.cumsum(variance_explained_dataset4)
    cumulative_variance_dataset5 = np.cumsum(variance_explained_dataset5)
    cumulative_variance_dataset6 = np.cumsum(variance_explained_dataset6)
    n_components = len(variance_explained_dataset1)
    bar_width = 0.35
    x = np.arange(1, n_components + 1)
    body_alpha = 0.4
    contour_alpha = 0.8
    plt.bar(x, variance_explained_dataset1, width=bar_width, edgecolor=(0, 0, 1, contour_alpha), facecolor=(0, 0, 1, body_alpha), linewidth=3, label='IV FF M1')
    plt.bar(x + bar_width, variance_explained_dataset2, width=bar_width, edgecolor=(1, 0, 0, contour_alpha), facecolor=(1, 0, 0, body_alpha), linewidth=3, label='IV FF M2')
    plt.bar(x, variance_explained_dataset3, width=bar_width, ls = 'dashdot', edgecolor=(0, 0, 1, contour_alpha), facecolor=(0, 0, 1, body_alpha), linewidth=3, label='IV HR M1')
    plt.bar(x + bar_width, variance_explained_dataset4, ls = 'dashdot', width=bar_width, edgecolor=(1, 0, 0, contour_alpha), facecolor=(1, 0, 0, body_alpha), linewidth=3, label='IV HR M2')
    plt.bar(x, variance_explained_dataset5, width=bar_width, ls = 'dotted', edgecolor=(0, 0, 1, contour_alpha), facecolor=(0, 0, 1, body_alpha), linewidth=3, label='IV REC M1')
    plt.bar(x + bar_width, variance_explained_dataset6, ls = 'dotted', width=bar_width, edgecolor=(1, 0, 0, contour_alpha), facecolor=(1, 0, 0, body_alpha), linewidth=3, label='IV REC M2')
    plt.plot(x + bar_width/2, cumulative_variance_dataset1, marker='o', markersize = 8, lw = 4, color='blue', label='CV FF M1')
    plt.plot(x + bar_width/2, cumulative_variance_dataset2, marker='o', markersize = 8,lw = 4,color='red', label='CV FF M2')
    plt.plot(x + bar_width/2, cumulative_variance_dataset3, marker='o', markersize = 8,lw = 4,ls = 'dashdot', color='blue', label='CV HR M1')
    plt.plot(x + bar_width/2, cumulative_variance_dataset4, marker='o', markersize = 8,lw = 4,ls = 'dashdot', color='red', label='CV HR M2')
    plt.plot(x + bar_width/2, cumulative_variance_dataset5, marker='o', markersize = 8,lw = 4,ls = 'dotted', color='blue', label='CV REC M1')
    plt.plot(x + bar_width/2, cumulative_variance_dataset6, marker='o', markersize = 8,lw = 4,ls = 'dotted', color='red', label='CV REC M2')
    plt.xlabel('Principal Component',fontsize=25)
    plt.ylabel('Variance Explained',fontsize = 25)
    plt.xticks(x + bar_width, x, fontsize=20)
    plt.yticks (fontsize=20)
    plt.legend(fontsize = 15)
    plt.show()

def WCSS (folder_path):
    '''
    Method to calculate the Within Cluster Sum of Squares for loaded plastic weights onto their first 3 principal components
    
    Parameters :
        folder_path : directory where to load the plastic weights from
        
    Reutrns :
        WCSS : WCSS scores for the loaded plastic weights
        perc_drop : drop in percentage of WCSS between subsequent number of clusters taken into account
    '''
    
    file_list = os.listdir(folder_path)
    sorted_file_names = sorted(file_list)
    WCSS = []
    perc_drop = []
    for file_name in sorted_file_names:  
        if file_name.endswith('.npy'):
            print(file_name)
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)   
            data = data.reshape(int(len(data) / 450), 450)
            scaler = StandardScaler()
            segmentation_std = scaler.fit_transform(data.transpose())
            pca = PCA(n_components=3)
            pca.fit(segmentation_std)
            scores_pca = pca.transform(segmentation_std)
            
            wcss = []
            for i in range(1, 21):
                kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans_pca.fit(scores_pca)
                wcss.append(kmeans_pca.inertia_)
            WCSS.append(wcss)
            percentage_decrease = []
            for i in range(len(wcss) - 1):
                decrease = wcss[i] - wcss[i + 1]
                percentage = (decrease / wcss[i]) * 100
                percentage_decrease.append(percentage)
            perc_drop.append(percentage_decrease)   
    return (WCSS, perc_drop)

def plot_WCSS (WCSS):
    '''
    Method to plot the WCSS for loaded plastic weights
    
    Parameters :
        WCSS : scores of WCSS on loaded plastic weights
    Returns :
        plot with the WCSS scores
    '''
    
    plt.figure (figsize = (10,8))
    plt.plot (WCSS[0], 'blue', label = 'FF M1', marker='o', markersize = 6, lw = 3)
    plt.plot (WCSS[1], 'red', label = 'FF M2', marker='o', markersize = 6, lw = 3)
    
    plt.plot (WCSS[2], 'blue', label = 'HR M1', marker='o', ls='dashdot', markersize = 6, lw = 3)
    plt.plot (WCSS[3], 'red', label = 'HR M2', marker='o', ls='dashdot', markersize = 6, lw = 3)
    
    plt.plot (WCSS[4], 'blue', label = 'REC M1', marker='o', ls='dotted', markersize = 6, lw = 3)
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
    labels = ['', '0', '', '', '', '','','','30000']
    plt.gca().set_yticklabels(labels,fontsize=20)
    x_ticks = np.array(range(len(WCSS[0])))
    even_x_ticks = x_ticks[x_ticks % 2 == 0]
    plt.xticks(even_x_ticks, even_x_ticks, fontsize=18)
    plt.xlabel('Number of clusters', fontsize = 30, labelpad=5)
    plt.ylabel('WCSS score', fontsize = 30, labelpad = -40)
    plt.show()

def plot_perc_drop_WCSS (perc_drop):
    '''
    Method to plot the percentage drop in WCSS between consecutive number of clusters taken into acocunt
    
    Parameters :
        perc_drop : drop in percentage of WCSS scores 
        
    Returns :
        the plot of the drop in percentage of WCSS
    '''
    
    plt.figure (figsize = (10,8))
    plt.plot (perc_drop[0], 'blue', label = 'FF M1', marker='o', markersize = 6, lw = 3)
    plt.plot (perc_drop[1], 'red', label = 'FF M2', marker='o', markersize = 6, lw = 3)
    
    plt.plot (perc_drop[2], 'blue', label = 'HR M1', marker='o', ls='dashdot', markersize = 6, lw = 3)
    plt.plot (perc_drop[3], 'red', label = 'HR M2', marker='o', ls='dashdot', markersize = 6, lw = 3)
    
    plt.plot (perc_drop[4], 'blue', label = 'REC M1', marker='o', ls='dotted', markersize = 6, lw = 3)
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

folder_path = '/home/lparrilla/Documents/Dynap_simulation/clustering'
