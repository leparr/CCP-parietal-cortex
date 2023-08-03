"""

@author: lparrilla
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def shape_w_rec (w_plast):
    '''
    Method to appropriately reshape loaded plastic weight for recurrent synapses
    
    Parameters :
        w_plast : loaded plastic weight
        
    Returns :
        w_plast_matrix : reshaped loaded plastic weight
    '''
    
    num_clusters = 9
    num_pre_neurons = 50
    num_post_neurons = 49
    w_plast_3d = np.reshape(w_plast, (num_clusters, num_pre_neurons, num_post_neurons))
    w_plast_matrix = np.transpose(w_plast_3d, (0, 2, 1))
    w_plast_matrix = np.reshape(w_plast_matrix, (num_clusters, -1))
    return w_plast_matrix

def plot_1_rec_cluster (w_plast, conn_type):
    '''
    Method to plot the recurrent weights of 1 single cluster
    
    Parameters :
        w_plast : loaded plastic weight
        conn_type : specifiy if the cluster is fully connected or without self recurrence, acceptable parameters are: 'full' and 'no_self'
        
    Returns :
        the plot of learned synaptic weights for 1 excitatory cluster
    '''
    
    re_w_plast = shape_w_rec(w_plast)
    if conn_type == 'full':
        cluster_w = re_w_plast[0]
    elif conn_type == 'no_self':
        cluster_w = list(re_w_plast[0])
        for i in np.arange (0,len(re_w_plast[0])+50,51):
            cluster_w.insert (i,0)
    else:
        print ('Provide a valid connectivity type')
    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap (np.array (cluster_w).reshape(50,50))
    cbar = ax.collections[0].colorbar
    text_height = 0.5 * (cbar.vmax - cbar.vmin)
    text_position = cbar.vmin + text_height
    cbar.ax.text(1.6, text_position, 'Weight magnitude', rotation=90, va='center', fontsize = 25)
    minimo = round(np.min(cluster_w),2)
    massimo = round(np.max (cluster_w),2)
    cbar.set_ticks([minimo,massimo])
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel ('Post neurons', fontsize = 20, labelpad = -10)
    ax.set_ylabel ('Pre neurons', fontsize = 20, labelpad = -20)
    ticks = [0,50]
    ax.set_yticks(ticks,ticks,fontsize = 20, rotation = 0)
    ax.set_xticks(ticks,ticks,fontsize = 20, rotation = 0)
    
def plot_weights (w_plast, synapse_type):
    '''
    Method to plot final learned plastic weight for the three types of synapses
    
    Parameters :
        w_plast : loaded plastic weight
        synapse_type : type of synapse of the loaded plastic weight, acceptable parameters are 'ff', 'hr' and 'rec'
    
    Returns :
        the plot of the learned weights
    '''
    
    if synapse_type == 'ff':
        fig = plt.figure(figsize=(8, 6))
        ax = sns.heatmap(w_plast.reshape(int(len(w_plast)/450),450))
        yticks = [0, 30, 60, 90, 120]
        xticks = [0,100,200,300,400]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=20, rotation = 0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=20, rotation = 0)
        cbar = ax.collections[0].colorbar
        text_height = 0.5 * (cbar.vmax - cbar.vmin)
        text_position = cbar.vmin + text_height
        cbar.ax.text(1.6, text_position, 'Weight magnitude', rotation=90, va='center', fontsize = 25)
        minimo = round(min(w_plast),2)
        massimo = round(max (w_plast),2)
        cbar.set_ticks([minimo,massimo])
        cbar.ax.tick_params(labelsize=20)
        ax.set_xlabel('Postsynaptic index', fontsize = 25)
        ax.set_ylabel('Presynaptic index', fontsize = 25, labelpad = -10)
        plt.show()
    elif synapse_type == 'hr':
        fig = plt.figure(figsize=(8, 6))
        ax = sns.heatmap(w_plast.reshape(int(len(w_plast)/450),450))
        yticks = [0, 49]
        xticks = [0,100,200,300,400]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=20, rotation = 0)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=20, rotation = 0)
        cbar = ax.collections[0].colorbar
        text_height = 0.5 * (cbar.vmax - cbar.vmin)
        text_position = cbar.vmin + text_height
        cbar.ax.text(1.6, text_position, 'Weight magnitude', rotation=90, va='center', fontsize = 25)
        minimo = round(min(w_plast),2)
        massimo = round(max (w_plast),2)
        cbar.set_ticks([minimo,massimo])
        cbar.ax.tick_params(labelsize=20)
        ax.set_xlabel('Postsynaptic index', fontsize = 25)
        ax.set_ylabel('Presynaptic index', fontsize = 25, labelpad = -10)
        plt.show()
    elif synapse_type == 'rec':
            fig = plt.figure(figsize=(8, 6))
            minimo = round(min(w_plast),2)
            massimo = round(max (w_plast),2)
            w_plast = shape_w_rec(w_plast)
            ax = sns.heatmap(w_plast.transpose())
            yticks = [0, 2400]
            xticks = np.arange (9)
            xtick_labels = ['0','1','2','3','4','5','6','7','8']
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, fontsize=20, rotation = 0)
            ax.set_xticks([x + 0.5 for x in range(len(xtick_labels))])
            ax.set_xticklabels (xtick_labels, fontsize = 25)
            cbar = ax.collections[0].colorbar
            text_height = 0.5 * (cbar.vmax - cbar.vmin)
            text_position = cbar.vmin + text_height
            cbar.ax.text(1.6, text_position, 'Weight magnitude', rotation=90, va='center', fontsize = 25)
            cbar.set_ticks([minimo,massimo])
            cbar.ax.tick_params(labelsize=20)
            ax.set_xlabel('Postsynaptic index', fontsize = 25)
            ax.set_ylabel('Cluster index', fontsize = 25, labelpad = -10)
            plt.show()

def plot_w_evo(w_evo):
    '''
    Method to plot the evolution of learned weight in time
    
    Parameters :
        w_evo : loaded matrix of plastic weight evolution in time
    
    Returns :
        the plot of the evolution of learned weights in time
    '''

    w_max = np.max(w_evo)
    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(w_evo)
    xticks = [0, 30, 60]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=17, rotation=0)
    yticks = [0, w_evo.shape[0] - 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=20)
    ax.set_xlabel('Time [ms]', fontsize=25, labelpad=0)
    ax.set_ylabel('Synaptic index', fontsize=25, labelpad=-40)
    cbar = ax.collections[0].colorbar        
    text_height = 0.5 * (cbar.vmax - cbar.vmin)
    text_position = cbar.vmin + text_height
    cbar.ax.text(1.6, text_position, 'Weight magnitude', rotation=90, va='center', fontsize=25)
    cbar.set_ticks([0, w_max])
    cbar.ax.tick_params(labelsize=20)
    plt.show()

