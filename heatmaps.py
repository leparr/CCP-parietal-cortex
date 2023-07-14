#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lparrilla
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def shape_w_rec (w_plast):
    num_clusters = 9
    num_pre_neurons = 50
    num_post_neurons = 49
    w_plast_3d = np.reshape(w_plast, (num_clusters, num_pre_neurons, num_post_neurons))
    w_plast_matrix = np.transpose(w_plast_3d, (0, 2, 1))
    w_plast_matrix = np.reshape(w_plast_matrix, (num_clusters, -1))
    return w_plast_matrix

def plot_weights (w_plast, synapse_type):
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

def shape_w_rec (w_plast):
    num_clusters = 9
    num_pre_neurons = 50
    num_post_neurons = 49
    w_plast_3d = np.reshape(w_plast, (num_clusters, num_pre_neurons, num_post_neurons))
    w_plast_matrix = np.transpose(w_plast_3d, (0, 2, 1))
    w_plast_matrix = np.reshape(w_plast_matrix, (num_clusters, -1))
    return w_plast_matrix


# LOAD W_PLAST
w_plast = np.load ('w_plast')

def plot_heatmaps (w_plast, synapse_type):
    if synapse_type == 'ff' or synapse_type == 'hr':

        w_plast_min = round(min(w_plast),3)
        w_plast_max = round(max(w_plast),3)
        w_plast_mean = round(np.mean(w_plast),3)
        w_plast_std = round (np.std(w_plast),3)
        print (w_plast_std)
        w_plast = w_plast.reshape(int(len(w_plast) / 450), 450)
        n = w_plast.shape[0]
        m = w_plast.shape[1]
        group_size = 50
        elements_per_group = group_size * n
        grouped_arr = np.split(w_plast, 9, axis=1)
        grouped_arr = np.concatenate(grouped_arr, axis=0)
        grouped_arr = grouped_arr.reshape(9, elements_per_group)
        fig = plt.figure(figsize = (8,6))
        for i in range(9):
            sns.kdeplot(grouped_arr[i], linewidth=4, label='Class ' + str(i))
            sns.despine()
            plt.legend(fontsize=15)
            plt.xlim(w_plast_min, w_plast_max)
            plt.xticks([w_plast_min, w_plast_mean, w_plast_max], [w_plast_min, w_plast_mean, w_plast_max],fontsize=15)
            plt.yticks(fontsize=15)
        plt.xlabel('Synaptic weights', fontsize = 20)
        plt.ylabel('KDE', fontsize = 20)
        plt.show()
    elif synapse_type == 'rec':
        w_plast_min = round(min(w_plast),3)
        w_plast_max = round(max(w_plast),3)
        w_plast_mean = round(np.mean(w_plast),3)
        w_plast_std = round (np.std(w_plast),3)
        print (w_plast_std)
        w_plast = shape_w_rec(w_plast)
        fig = plt.figure(figsize = (8,6))
        for i in range(9):
            sns.kdeplot(w_plast[i], linewidth=4, label='Class ' + str(i))
            sns.despine()
            plt.legend(fontsize=15)
            plt.xlim(w_plast_min, w_plast_max)
            plt.xticks([w_plast_min, w_plast_mean, w_plast_max], [w_plast_min, w_plast_mean, w_plast_max],fontsize=15)
            plt.yticks(fontsize=15)
        plt.xlabel('Synaptic weights', fontsize = 20)
        plt.ylabel('KDE', fontsize = 20)
        plt.show()

# PLOT WEIGHT EVOLUTION IN TIME
def get_max(w_evo):
    w_max = []
    for i in range(len(w_evo)):
        w_max.append(max(w_evo[i]))
    return round(max(w_max), 2)
   
def plot_w_evo(w_evo):
    w_max = get_max(w_evo)
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


