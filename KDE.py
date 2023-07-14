#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: lparrilla
"""


import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

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
def plot_kde (w_plast, synapse_type):
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


