"""

@author: lparrilla
"""

import os
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import kruskal
from matplotlib import pyplot as plt 

def progressive_pruning(matrix, filename):
    '''
    Method do apply progressive pruning depending on which training iteration the loaded plastic weight is from
    
    Parameters : 
        matrix : loaded plastic weight
        filename : name of the loaded plastic weight
    Returns :
        matrix : input plastic weight if the filename does not end wit ha digit
        pruned_matrix : plastic weight with progressive pruning applied
    '''
    
    base_name, _ = os.path.splitext(filename)
    if not base_name[-1].isdigit():
        return matrix

    matrix = matrix.reshape(int(len(matrix) / 450), 450)
    file_num = int(base_name[-1]) 
    k = 100 - (file_num * 10)
    top_k_indices = np.argsort(matrix, axis=0)[-k:]
    mask = np.ones_like(matrix, dtype=bool)
    mask[top_k_indices, np.arange(matrix.shape[1])] = False
    pruned_matrix = np.where(mask, 0, matrix)
    return pruned_matrix

def load_np_arrays(directory):
    '''
    Method to load plastic weights from a directory
    
    Parameters : 
        directory : directory from where to load weights
    Returns :
        it loads in the workspace the plastic weights with progressive pruning applied
    '''
    
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    for filename in os.listdir(subfolder_path):
                        if filename.endswith('.npy'):
                            file_path = os.path.join(subfolder_path, filename)
                            var_name = os.path.splitext(filename)[0] 
                            np_array = np.load(file_path)
                            normalized_array = progressive_pruning(np_array,filename)
                            globals()[var_name] = normalized_array.flatten()
                            
directory_path = 'directory'
load_np_arrays(directory_path)

# SORTING W_PLASTS
arr_24_ff_ep0 = []
arr_24_ff_ep1 = []
arr_24_ff_ep2 = []
arr_24_ff_ep3 = []
arr_24_ff_ep4 = []
arr_24_ff_ep5 = []

arr_24_hr_ep0 = []
arr_24_hr_ep1 = []
arr_24_hr_ep2 = []
arr_24_hr_ep3 = []
arr_24_hr_ep4 = []
arr_24_hr_ep5 = []

arr_24_rec_ep0 = []
arr_24_rec_ep1 = []
arr_24_rec_ep2 = []
arr_24_rec_ep3 = []
arr_24_rec_ep4 = []
arr_24_rec_ep5 = []

arr_25_ff_ep0 = []
arr_25_ff_ep1 = []
arr_25_ff_ep2 = []
arr_25_ff_ep3 = []
arr_25_ff_ep4 = []
arr_25_ff_ep5 = []

arr_25_hr_ep0 = []
arr_25_hr_ep1 = []
arr_25_hr_ep2 = []
arr_25_hr_ep3 = []
arr_25_hr_ep4 = []
arr_25_hr_ep5 = []

arr_25_rec_ep0 = []
arr_25_rec_ep1 = []
arr_25_rec_ep2 = []
arr_25_rec_ep3 = []
arr_25_rec_ep4 = []
arr_25_rec_ep5 = []

global_vars = globals().copy()

for var_name in global_vars:
    if var_name.startswith('24'):
        if var_name.startswith('24_ep'):
            if 'ep_0_' in var_name:
                arr_24_ff_ep0.append (global_vars[var_name])
            elif 'ep_1_' in var_name:
                arr_24_ff_ep1.append (global_vars[var_name])
            elif 'ep_2_' in var_name:
                arr_24_ff_ep2.append (global_vars[var_name])
            elif 'ep_3_' in var_name:
                arr_24_ff_ep3.append (global_vars[var_name])
            elif 'ep_4_' in var_name:
                arr_24_ff_ep4.append (global_vars[var_name])
            elif 'ep_5_' in var_name:
                arr_24_ff_ep5.append (global_vars[var_name])
                
        elif var_name.startswith('24_HR'):
            if 'ep_0_' in var_name:
                arr_24_hr_ep0.append (global_vars[var_name])
            elif 'ep_1_' in var_name:
                arr_24_hr_ep1.append (global_vars[var_name])
            elif 'ep_2_' in var_name:
                arr_24_hr_ep2.append (global_vars[var_name])
            elif 'ep_3_' in var_name:
                arr_24_hr_ep3.append (global_vars[var_name])
            elif 'ep_4_' in var_name:
                arr_24_hr_ep4.append (global_vars[var_name])
            elif 'ep_5_' in var_name:
                arr_24_hr_ep5.append (global_vars[var_name])

        elif var_name.startswith('24_REC'):
            if 'ep_0_' in var_name:
                arr_24_rec_ep0.append (global_vars[var_name])
            elif 'ep_1_' in var_name:
                arr_24_rec_ep1.append (global_vars[var_name])
            elif 'ep_2_' in var_name:
                arr_24_rec_ep2.append (global_vars[var_name])
            elif 'ep_3_' in var_name:
                arr_24_rec_ep3.append (global_vars[var_name])
            elif 'ep_4_' in var_name:
                arr_24_rec_ep4.append (global_vars[var_name])
            elif 'ep_5_' in var_name:
                arr_24_rec_ep5.append (global_vars[var_name])  
   
    elif var_name.startswith('25'):
        
         if var_name.startswith('25_ep'):
             if 'ep_0_' in var_name:
                 arr_25_ff_ep0.append (global_vars[var_name])
             elif 'ep_1_' in var_name:
                 arr_25_ff_ep1.append (global_vars[var_name])
             elif 'ep_2_' in var_name:
                 arr_25_ff_ep2.append (global_vars[var_name])
             elif 'ep_3_' in var_name:
                 arr_25_ff_ep3.append (global_vars[var_name])
             elif 'ep_4_' in var_name:
                 arr_25_ff_ep4.append (global_vars[var_name])
             elif 'ep_5_' in var_name:
                 arr_25_ff_ep5.append (global_vars[var_name])
                 
         elif var_name.startswith('25_HR'):
             if 'ep_0_' in var_name:
                 arr_25_hr_ep0.append (global_vars[var_name])
             elif 'ep_1_' in var_name:
                 arr_25_hr_ep1.append (global_vars[var_name])
             elif 'ep_2_' in var_name:
                 arr_25_hr_ep2.append (global_vars[var_name])
             elif 'ep_3_' in var_name:
                 arr_25_hr_ep3.append (global_vars[var_name])
             elif 'ep_4_' in var_name:
                 arr_25_hr_ep4.append (global_vars[var_name])
             elif 'ep_5_' in var_name:
                 arr_25_hr_ep5.append (global_vars[var_name])

         elif var_name.startswith('25_REC'):
             if 'ep_0_' in var_name:
                 arr_25_rec_ep0.append (global_vars[var_name])
             elif 'ep_1_' in var_name:
                 arr_25_rec_ep1.append (global_vars[var_name])
             elif 'ep_2_' in var_name:
                 arr_25_rec_ep2.append (global_vars[var_name])
             elif 'ep_3_' in var_name:
                 arr_25_rec_ep3.append (global_vars[var_name])
             elif 'ep_4_' in var_name:
                 arr_25_rec_ep4.append (global_vars[var_name])
             elif 'ep_5_' in var_name:
                 arr_25_rec_ep5.append (global_vars[var_name])

def monte_carlo_resample(array1, array2, num_samples):
    '''
    Method to apply the MonteCarlo resample to arrays of different lenghts
    
    Parameters : 
        array1 : first array to resample
        array2 : second array to resample
        num_samples : lenght of the resampled arrays
        
    Returns :
        resampled_1 : first resampled arrays 
        resampled_2 : second resampled array 
    '''
    
    indices_a = np.random.choice(len(array1), num_samples, replace=True)
    indices_b = np.random.choice(len(array2), num_samples, replace=True)
    resampled_1 = array1[indices_a]
    resampled_2 = array2[indices_b]
    return resampled_1, resampled_2

# COMPUTING COSINE SIMILARITY
cs_0_ff_hr = [1]
cs_1_ff_hr = [1]
cs_2_ff_hr = [1]
cs_3_ff_hr = [1]
cs_4_ff_hr = [1]
cs_5_ff_hr = [1]

cs_0_ff_rec = [1]
cs_1_ff_rec = [1]
cs_2_ff_rec = [1]
cs_3_ff_rec = [1]
cs_4_ff_rec = [1]
cs_5_ff_rec = [1]

cs_0_hr_rec = [1]
cs_1_hr_rec = [1]
cs_2_hr_rec = [1]
cs_3_hr_rec = [1]
cs_4_hr_rec = [1]
cs_5_hr_rec = [1]

sample_size = 10000
box_cs_ff_hr = []
box_cs_ff_rec = []

cs_m_ff_hr = [1]
cs_std_ff_hr = [0]

cs_m_ff_rec = [1]
cs_std_ff_rec = [0]

cs_m_hr_rec = [1]
cs_std_hr_rec = [0]

for i in np.arange(10):
    # FF HR
    box_cs_ff_hr = []
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep0[i],arr_24_hr_ep0[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep0[i],arr_25_hr_ep0[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep1[i],arr_24_hr_ep1[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep1[i],arr_25_hr_ep1[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))

    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep2[i],arr_24_hr_ep2[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep2[i],arr_25_hr_ep2[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep3[i],arr_24_hr_ep3[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep3[i],arr_25_hr_ep3[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))

    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep4[i],arr_24_hr_ep4[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep4[i],arr_25_hr_ep4[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep5[i],arr_24_hr_ep5[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep5[i],arr_25_hr_ep5[i], sample_size)
    box_cs_ff_hr.append (1-cosine(ff0,hr0))
    cs_m_ff_hr.append (np.mean(box_cs_ff_hr))
    cs_std_ff_hr.append (np.std(box_cs_ff_hr))
    
    # FF REC
    box_cs_ff_rec = []
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep0[i],arr_24_rec_ep0[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep0[i],arr_25_rec_ep0[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep1[i],arr_24_rec_ep1[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep1[i],arr_25_rec_ep1[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))

    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep2[i],arr_24_rec_ep2[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep2[i],arr_25_rec_ep2[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep3[i],arr_24_rec_ep3[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep3[i],arr_25_rec_ep3[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))

    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep4[i],arr_24_rec_ep4[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep4[i],arr_25_rec_ep4[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_ff_ep5[i],arr_24_rec_ep5[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_ff_ep5[i],arr_25_rec_ep5[i], sample_size)
    box_cs_ff_rec.append (1-cosine(ff0,hr0))
    cs_m_ff_rec.append (np.mean(box_cs_ff_rec))
    cs_std_ff_rec.append (np.std(box_cs_ff_rec))
   
    # HR REC
    box_cs_hr_rec = []
    ff0, hr0 = monte_carlo_resample(arr_24_hr_ep0[i],arr_24_rec_ep0[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_hr_ep0[i],arr_25_rec_ep0[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_hr_ep1[i],arr_24_rec_ep1[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_hr_ep1[i],arr_25_rec_ep1[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))

    ff0, hr0 = monte_carlo_resample(arr_24_hr_ep2[i],arr_24_rec_ep2[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_hr_ep2[i],arr_25_rec_ep2[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_hr_ep3[i],arr_24_rec_ep3[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_hr_ep3[i],arr_25_rec_ep3[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))

    ff0, hr0 = monte_carlo_resample(arr_24_hr_ep4[i],arr_24_rec_ep4[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_hr_ep4[i],arr_25_rec_ep4[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_24_hr_ep5[i],arr_24_rec_ep5[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    ff0, hr0 = monte_carlo_resample(arr_25_hr_ep5[i],arr_25_rec_ep5[i], sample_size)
    box_cs_hr_rec.append (1-cosine(ff0,hr0))
    cs_m_hr_rec.append (np.mean(box_cs_hr_rec))
    cs_std_hr_rec.append (np.std(box_cs_hr_rec))

# PLOTTING
x = np.arange(len(cs_m_ff_hr))
cs_m_ff_hr = np.array(cs_m_ff_hr)
cs_std_ff_hr = np.array(cs_std_ff_hr)
cs_m_ff_rec = np.array(cs_m_ff_rec)
cs_std_ff_rec = np.array(cs_std_ff_rec)
cs_m_hr_rec = np.array(cs_m_hr_rec)
cs_std_hr_rec = np.array(cs_std_hr_rec)
plt.plot(x, cs_m_ff_hr, lw=3,color='orange', label='FF-HR')
plt.fill_between(x, cs_m_ff_hr - cs_std_ff_hr, cs_m_ff_hr + cs_std_ff_hr, color='orange', alpha=0.3)
plt.plot(x, cs_m_ff_rec, lw=3,color='blue', label='FF-REC')
plt.fill_between(x, cs_m_ff_rec - cs_std_ff_rec, cs_m_ff_rec + cs_std_ff_rec, color='blue', alpha=0.3)
plt.plot(x, cs_m_hr_rec, lw=3,color='green', label='HR-REC')
plt.fill_between(x, cs_m_hr_rec - cs_std_hr_rec, cs_m_hr_rec + cs_std_hr_rec, color='green', alpha=0.3)
plt.xlabel('Training Iteration',fontsize=30)
plt.ylabel('Cosine Similarity',fontsize=30)
plt.gca().yaxis.set_label_coords(-0.02, 0.55)
plt.xticks(fontsize=20, ticks=[0,5,10]) 
plt.yticks(fontsize=20, ticks=[1,0.3])
plt.legend(fontsize = 15)
plt.show()

# PRINT DROP IN COSINE SIMILARITY FROM FIRST TO LAST TRAINING ITERATION
print ((cs_m_ff_hr[10]-cs_m_ff_hr[1])/cs_m_ff_hr[1])
print((cs_m_ff_rec[10]-cs_m_ff_rec[1])/cs_m_ff_rec[1])
print((cs_m_hr_rec[10]-cs_m_hr_rec[1])/cs_m_hr_rec[1])

def kruskal_wallis (arr):
    '''
    Method to compute Kruskal Wallis test
    
    Parameters :
        arr : array of arrays on which to compute Kruskal Wallis test
    Returns :
        statistic : Kruskal Wallis value  between input arrays
        p_value : probability to observe the computed statistics given true null hypothesis
    '''
    
    statistic, p_value = kruskal(*arr)
    alpha = 0.01
    return (statistic,p_value)
    print("Kruskal-Wallis test results:")
    print(f"Kruskal-Wallis statistic: {statistic}")
    print(f"p-value: {p_value}")
    if p_value < alpha:
        print("There is a significant difference among the groups.")
    else:
        print("There is no significant difference among the groups.")
    









