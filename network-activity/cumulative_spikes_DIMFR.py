"""

@author: lparrilla
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

directory_path = '/home/lparrilla/Documents/Dynap_simulation/heat'
loaded_arrays = {}
for filename in os.listdir(directory_path):
    if filename.endswith('.npy'):
        full_path = os.path.join(directory_path, filename)
        array_name = os.path.splitext(filename)[0]
        loaded_arrays[array_name] = np.load(full_path)
# ARRANGE LOADED DATA    
sp_i_ff = []
sp_t_ff = []
sp_i_rec = []
sp_t_rec = []
DIMFR_ff = []
DIMFR_rec = []
for i in loaded_arrays.keys():
    if 'ff' in i:
        sp_i_ff = np.concatenate ([sp_i_ff,loaded_arrays[i][0]])
        sp_t_ff = np.concatenate ([sp_t_ff,loaded_arrays[i][1]])
    elif 'rec' in i:
        sp_i_rec = np.concatenate ([loaded_arrays[i][0],sp_i_rec])
        sp_t_rec = np.concatenate ([sp_t_rec,loaded_arrays[i][1]])
    elif 'DIMFR' in i:
        DIMFR_ff.append (loaded_arrays[i][0])
        DIMFR_rec.append (loaded_arrays[i][1])
m_DIMFR_ff = (np.mean(DIMFR_ff, axis=0) -1)*100
std_DIMFR_ff = np.std(DIMFR_ff, axis=0)*100
m_DIMFR_rec = (np.mean(DIMFR_rec, axis=0) -1)*100
std_DIMFR_rec = np.std(DIMFR_rec, axis=0)*100

def from_spikes_to_heat (spike_i,spike_t):
    '''
    Method to convert neural indices and spike times to a cumulative heatmap for every class presentation
    
    Parametars :
        spike_i : array of neural indices
        spike_t : array of spike times
        
    Returns :
        cumulative_spikes/num_neurons_per_bin : cumulative spikes in each time bins, normalized per number of neurons
    '''
    
    bin_width = 100
    num_neurons_per_bin = 50
    def group_indices(indices, group_size):
        return [indices[i:i + group_size] for i in range(0, len(indices), group_size)]
    neuron_groups = group_indices(np.unique(spike_i), num_neurons_per_bin)
    duration = 9000
    num_bins = int(duration / bin_width)
    num_groups = len(neuron_groups)
    spike_counts = np.zeros((num_groups, num_bins))
    for spike_t_val, spike_i_val in zip(spike_t, spike_i):
        bin_index = int(spike_t_val / bin_width)
        for group_index, neuron_group in enumerate(neuron_groups):
            if spike_i_val in neuron_group:
                spike_counts[group_index, bin_index] += 1
                break
    cumulative_spike_counts = np.cumsum(spike_counts, axis=1)
    cumulative_spikes = np.zeros_like(cumulative_spike_counts)
    for group_index in range(num_groups):
        for bin_index in range(num_bins):
            cumulative_spikes[group_index, bin_index] = cumulative_spike_counts[group_index, bin_index] - cumulative_spike_counts[group_index, bin_index // 10 * 10]
    return (cumulative_spikes/num_neurons_per_bin)

num_bins = 90
num_groups = 9
heat_ff = from_spikes_to_heat(sp_i_ff, sp_t_ff)/5
heat_rec = from_spikes_to_heat(sp_i_rec, sp_t_rec)/5

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(12, 4, width_ratios=[2, 2, 2, 3])  

ax1 = plt.subplot(gs[:6, :3]) 
sns.heatmap(heat_rec, ax=ax1, cmap='plasma')
ax1.set_yticks(np.arange(9) + 0.5)
ax1.set_yticklabels(np.arange(0, 9), fontsize=15, rotation=0)
ax1.set_xticks(np.arange(0, num_bins, 20) + 0.5)
ax1.set_xticklabels(np.arange(0, num_bins, 20), fontsize=15, rotation=0)
ax1.hlines(np.arange(1, 9), xmin=0, xmax=num_bins, colors='black', linewidth=2)
ax1.vlines(np.arange(0, num_bins + 1, 10), ymin=0, ymax=num_groups, colors='black', linewidth=2)
ax1.set_ylabel ('WTA Populations', fontsize = 20)
cbar = ax1.collections[0].colorbar
text_height = 0.5 * (cbar.vmax - cbar.vmin)
text_position = cbar.vmin + text_height
cbar.ax.text(1.6, text_position, 'Spikes/neuron', rotation=90, va='center', fontsize = 20)
minimo = round(np.min(heat_rec),1)
massimo = round(np.max (heat_rec),1)
cbar.set_ticks([minimo,massimo])
cbar.ax.tick_params(labelsize=20)

ax2 = plt.subplot(gs[6:, :3]) 
sns.heatmap(heat_ff, ax=ax2, cmap='plasma')
ax2.set_yticks(np.arange(9) + 0.5)
ax2.set_yticklabels(np.arange(0, 9), fontsize=15, rotation=0)
ax2.set_xticks(np.arange(0, num_bins, 20) + 0.5)
ax2.set_xticklabels(np.arange(0, num_bins, 20), fontsize=15, rotation=0)
ax2.hlines(np.arange(1, 9), xmin=0, xmax=num_bins, colors='black', linewidth=2)
ax2.vlines(np.arange(0, num_bins + 1, 10), ymin=0, ymax=num_groups, colors='black', linewidth=2)
ax2.set_xlabel ('Time bins', fontsize = 20)
ax2.set_ylabel ('FF Populations', fontsize = 20)
cbar = ax2.collections[0].colorbar
text_height = 0.5 * (cbar.vmax - cbar.vmin)
text_position = cbar.vmin + text_height
cbar.ax.text(1.6, text_position, 'Spikes/neuron', rotation=90, va='center', fontsize = 20)
minimo = round(np.min(heat_ff),1)
massimo = round(np.max (heat_ff),1)
cbar.set_ticks([minimo,massimo])
cbar.ax.tick_params(labelsize=20)

ax3 = plt.subplot(gs[:, 3:]) 
x = np.arange(len(m_DIMFR_rec))
ax3.plot(m_DIMFR_rec, label='WTA', color='orange')
ax3.plot(m_DIMFR_ff, label='FF', color= 'b')
plt.fill_between(x, m_DIMFR_rec - std_DIMFR_rec, m_DIMFR_rec + std_DIMFR_rec, alpha=0.2, color='orange')
plt.fill_between(x, m_DIMFR_ff - std_DIMFR_ff, m_DIMFR_ff + std_DIMFR_ff, alpha=0.2, color='b')
ax3.set_xlabel('Correct classifications', fontsize=20)
ax3.set_ylabel('DIMFR [%]', fontsize = 20)
xlabpos = np.arange (8)
xlab = ['1','2','3','4','5','6','7','8']
ax3.set_xticks(xlabpos)
ax3.set_xticklabels(xlab, fontsize=15)
ax3.tick_params(axis='y', labelsize=15)
ax3.tick_params(axis='x', labelsize=15)
ax3.legend(loc = 'upper left', fontsize = 20)
ax3.grid(True)
plt.tight_layout()
plt.show()

