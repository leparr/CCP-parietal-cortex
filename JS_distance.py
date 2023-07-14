#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lparrilla
"""

import os
import numpy as np
from scipy.spatial.distance import jensenshannon
from matplotlib import pyplot as plt

def load_np_arrays(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            var_name = os.path.splitext(filename)[0]  # Remove file extension
            np_array = np.load(file_path)
            globals()[var_name] = np_array

directory_path = '/home/lparrilla/Documents/Dynap_simulation/final_w'
load_np_arrays(directory_path)

arr_24_ff = []
arr_25_ff = []

arr_24_hr = []
arr_25_hr = []

arr_24_rec = []
arr_25_rec = []

global_vars = globals().copy()

for var_name in global_vars:
    if var_name.startswith('24'):
        if 'FF' in var_name:
            arr_24_ff.append(global_vars[var_name])
        elif 'HR' in var_name:
            arr_24_hr.append(global_vars[var_name])
        elif 'REC' in var_name:
            arr_24_rec.append(global_vars[var_name])
        elif var_name.startswith('24_ep'):
            arr_24_ff.append (global_vars[var_name])
            
    elif var_name.startswith('25'):
        if 'FF' in var_name:
            arr_25_ff.append(global_vars[var_name])
        elif 'HR' in var_name:
            arr_25_hr.append(global_vars[var_name])
        elif 'REC' in var_name:
            arr_25_rec.append(global_vars[var_name])
        elif var_name.startswith('25_ep'):
            arr_25_ff.append (global_vars[var_name])

js_ff = []
js_hr = []
js_rec = []

def monte_carlo_resample(a, b, num_samples):
    indices_a = np.random.choice(len(a), num_samples, replace=True)
    indices_b = np.random.choice(len(b), num_samples, replace=True)
    resampled_a = a[indices_a]
    resampled_b = b[indices_b]
    return resampled_a, resampled_b
    
full_js_ff = []
for i in np.arange(6):
    for sample_size in np.arange (0,10000,1):
        ff1, ff2 = monte_carlo_resample(arr_24_ff[i],arr_25_ff[i], sample_size)
        js_ff.append(jensenshannon(ff1,ff2))
    full_js_ff.append (np.mean(js_ff))
    js_hr.append (np.mean(jensenshannon(arr_24_hr[i], arr_25_hr[i])))
    js_rec.append (np.mean(jensenshannon(arr_24_rec[i], arr_25_rec[i])))

plt.plot(full_js_ff, color= 'black', marker='o', markersize = 10, label='FF')
plt.plot(js_hr, marker='s', color= 'black', markersize = 10,label='HR')
plt.plot(js_rec, marker='^', color= 'black', markersize = 10,label='REC')

plt.legend(fontsize = 15)
plt.xlabel('Behavioral epochs',fontsize=30)
plt.ylabel('Jensen-Shannon distance',fontsize=30)
plt.xticks(np.arange(6))
plt.yticks(np.arange(0,0.6,0.1))

labels = ['0', '', '', '', '','5']
plt.gca().set_xticklabels(labels,fontsize=20)
labels = ['', '0.1', '', '', '0.4','']
plt.gca().set_yticklabels(labels,fontsize=20)
plt.show()     
        



