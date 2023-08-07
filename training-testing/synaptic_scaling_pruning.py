#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:09:59 2023

@author: lparrilla
"""
import numpy as np
import os

def pruning(matrix):
    m, n = matrix.shape
    top_20_indices = np.argpartition(matrix, -30, axis=0)[-30:]
    pruned = np.zeros((m, n))
    pruned[top_20_indices, np.arange(n)] = matrix[top_20_indices, np.arange(n)]
    return pruned.flatten()

def ss (w_plast):
    targets = []
    for i in range (int(len (w_plast)/450)):
        for j in range (450):
            targets = np.concatenate ([targets,[j]])
    w = []
    n_read = 50
    for j in np.arange (0,9,1):
        g = []
        for i in range (len(w_plast)):    
            if targets[i] >= j*n_read and targets[i] < j*n_read+n_read :
                g.append(w_plast[i])
        w.append(g)    
    x = []
    for i in range (len(w)): 
        x.append (sum(w[i]))
    perc = []
    for i in range (len(w)):    
        perc.append (x[i]*100/sum(x))
    coeff = []
    for i in range (len(perc)):
          coeff.append(11.11/perc[i])  
    idx_c = []
    for i in range (len(coeff)):
        idx_c.append ([coeff[i],i])
    for i in range (len(w)):
        for j in range (len(w[i])):
            w[i][j] = w[i][j]*coeff[i]
    for i in range (len(w_plast)):
        for j in np.arange (0,9,1):
            if targets[i] >= j*n_read and targets[i] < j*n_read+n_read:            
                w_plast[i] = w_plast[i]*coeff[j]
    return (w_plast)

