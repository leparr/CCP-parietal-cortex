"""

@author: lparrilla
"""

import pandas as pd
from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
import time
from DynapSE import DynapSE
from equations.dynapse_eq import *
from parameters.dynapse_param import *
import warnings
import seaborn as sns
import ast
from brian2 import prefs

prefs.use_cache = False
BrianLogger.suppress_name('base')
warnings.filterwarnings("ignore")
BrianLogger.suppress_name('brian2.codegen.generators.base')
BrianLogger.log_level_error()

def spikenize (chunk, l_bin):
    '''
    Method to create neural indices and spike times from sorted csv
    
    Parameters :
        chunk : segment of the dataset to convert into neural indices and spike times
        l_bin : temporal lenght of each bin in the csv
        
    Returns :
        spike_i : neural indices
        spike_t : spike times
    '''
    
    spike_i = []
    spike_t = []    
    for ep in range (len (chunk)):        
        spike_i.append([])
        spike_t.append([])        
        for col in range (len(chunk[ep])):            
            for row in range (len (chunk[ep][col])):                 
                if chunk[ep][col][row] != 0:
                    spike_i[ep].append(col)
                    spike_t[ep].append(row*l_bin)                    
                    
    return (spike_i,spike_t)

def spikenize_dataset (dataset, nep, l_bin):
    '''
    Method to convert a csv into neural indices and spike times to be passed to the input layer
    
    Parameters :
        dataset : dataset to train/test the network on
        nep : number of behavioral epoch to train/test the network on
        
    Returns :
        spike_i : neural indices for input layer
        spikes_t : spike times for input layer
        duration : complete duration for the presentation of one class
    '''

    if dataset == 24:
        n_in = 138
        file = "/home/lparrilla/Documents/Dynap_simulation/Parietal_datasets/reach_led1_24V6A.1ms.csv"
    elif dataset == 25:
        n_in = 124
        file = "/home/lparrilla/Documents/Dynap_simulation/Parietal_datasets/reach_led1_25V6A.1ms.csv"
    else:
        print ('Provide a valid dataset number')
    string = []
    for i in range (n_in):
        box = ('n'+str(i))
        string.append (box)
    df = pd.read_csv(file,skiprows=[0],dtype='float')
    with open(file) as fd:
        for n, line in enumerate(fd):
            intestazione = line.strip("# ")
            break
    epochdef = ast.literal_eval(intestazione[intestazione.find('{'):intestazione.find('}')+1]) 
    n_bin = epochdef[nep]
    duration = l_bin*n_bin*ms
    spikebox = []
    for i in range (9):
        boxbox = []
        for j in range (10):
            box = df[(df.epoch == nep)&(df.condition == i)&(df.trial == j)]
            n_activity = box.loc [:,string].values
            n_activity = n_activity.transpose()
            boxbox.append(n_activity)
        spikebox.append(boxbox)
    sorted_cond = spikebox
    spike_i = []
    spike_t = []
    for i in range (9):
        print ('Spikenizing condition '+str(i))    
        (sp_i,sp_t) = spikenize (sorted_cond[i], 1)          
        spike_i.append (sp_i)
        spike_t.append (sp_t)
    return (n_in, spike_i, spike_t, duration)

def extract_DIMFR (mean_rates):
    '''
    Method to extract the Difference in Mean Increase in Firing Rate between cluster classifying corrrectly and other clusters
    
    Parameters :
        mean_rates : average firing rate of the 9 clusters
    
    Returns :
        DIMFR : Difference in Mean Firing Rate Increase
    '''
    
    highest_number = max(mean_rates)
    other_numbers = [number for number in mean_rates if number != highest_number]
    mean = sum(other_numbers) / len(other_numbers)
    DIMFR =  highest_number / mean    
    return (DIMFR)

mm_lvl = 0.05
n_class = 50
n_batch = 1
nep = 2
dataset = 25
lag = 150
trial_cond = np.tile(np.arange(9), (10, 1))
t1 = time.time()        
n_in, spike_i,spike_t,duration = spikenize_dataset (dataset,nep, l_bin=1)
t2 = time.time ()
print ('Time to preprocess = ' +str (t2-t1))
tot_time = duration+lag*ms

def get_rates(spikeMon, measurement_period = (duration)):
    '''
    Method to get firing rates of readout neurons
    
    Parameters :
        spikeMon : spikemonitor object recording from the neural population
        duration : duration of the stimulus
        
    Returns :
        rates : neural firing rates over the duration of the input stimulus
    '''
    
    rates = np.zeros(len(spikeMon.event_trains()))
    rates = [len(spikeMon.event_trains()[i][spikeMon.event_trains()[i] > spikeMon.clock.t - measurement_period]) / measurement_period
             for i in range(len(spikeMon.event_trains()))]
    return rates        

# =============================================================================
# NEURAL NETWORK
# =============================================================================

'''
Instantiate a Spiking Neural Network to decode 9 spatial positions given an input dataset of single cells recordings

The network is composed by:
    1 Input layer as virtual neurons (L_in)
    1 Hidden layer (L_hid)
    1 Recurrent pools (L_rec)
    1 Inhibitory cluster (L_inh)    
'''

network = Network()
chip = DynapSE(network)

post_lr_ff = 0.0000001
pre_lr_ff = post_lr_ff/10000
post_lr_hr = 0.0000001
pre_lr_hr = post_lr_hr/10000
post_lr_rec = 0.0000001
pre_lr_rec = post_lr_rec/10000

indices = []
times = []*ms
L_in = SpikeGeneratorGroup (n_in, indices, times, name='L_in')
L_hid = chip.get_neurons(n_class*9, 'Core_0')
L_rec = chip.get_neurons(n_class*9, 'Core_1')
L_inh = chip.get_neurons(90, 'Core_2')
rates_noise = np.random.uniform(15, 25, size=n_class*9)*Hz
PG_noise = PoissonGroup (n_class*9, rates = rates_noise)

s_in = chip.add_connection (L_in, L_hid, synapse_type = 'TRI_STDP')
s_in.connect (True)
s_in.weight = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*550
s_in.w_plast = np.load ('ff_w_plast')
s_in.nuEEpre = pre_lr_ff
s_in.nuEEpost = post_lr_ff
s_in.I_tau_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*50*pA
s_in.I_g_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*50*pA
s_in.I_wo_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*50*pA
s_in.C_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*1.5 * pF
    
s_hr = chip.add_connection(L_hid, L_rec, synapse_type = 'TRI_STDP')
s_hr.connect (condition='int(i/n_class) == int(j/n_class)')
s_hr.weight = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*400
s_hr.w_plast = np.load ('hr_w_plast')
s_hr.nuEEpre = pre_lr_hr
s_hr.nuEEpost = post_lr_hr
s_hr.I_tau_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*50*pA
s_hr.I_g_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*50*pA
s_hr.I_wo_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*50*pA
s_hr.C_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*1.5 * pF

s_rec = chip.add_connection(L_rec, L_rec, synapse_type = 'TRI_STDP')
s_rec.connect (condition='int(i/n_class) == int(j/n_class) and i!=j')
s_rec.weight = np.random.normal(loc=1, scale=mm_lvl, size=len(s_rec))*20
s_rec.w_plast = np.load ('rec_w_plast')
s_rec.nuEEpre = pre_lr_rec
s_rec.nuEEpost = post_lr_rec
s_rec.I_tau_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_rec))*50*pA
s_rec.I_g_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_rec))*50*pA
s_rec.I_wo_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_rec))*50*pA
s_rec.C_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_rec))*1.5 * pF

s_ei = chip.add_connection (L_rec, L_inh, synapse_type = 'AMPA')
s_ei.connect (True)
s_ei.weight = np.random.normal(loc=1, scale=mm_lvl, size=len(s_ei))*40
s_ei.C_syn_exc2 = np.random.normal(loc=1, scale=0.05, size=len(s_ei))* 1.5*pF  
s_ei.I_tau_syn_exc2 = np.random.normal(loc=1, scale=0.05, size=len(s_ei))* 50*pA      
s_ei.I_wo_syn_exc2 = np.random.normal(loc=1, scale=0.05, size=len(s_ei))* 50*pA       
s_ei.I_g_syn_exc2 = np.random.normal(loc=1, scale=0.05, size=len(s_ei))* 50*pA

s_inh=chip.add_connection(L_inh,L_rec,synapse_type = 'GABA_B')
s_inh.connect (True)
s_inh.weight = np.random.normal(loc=1, scale=mm_lvl, size=len(s_inh))*-10000
s_inh.C_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_inh))* 1.5*pF             
s_inh.I_tau_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_inh))* 15*pA    
s_inh.I_tau_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_inh))* 150*pA      
s_inh.I_wo_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_inh))* 50*pA       
s_inh.I_g_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_inh))* 10*pA

s_noise  = chip.add_connection(PG_noise, L_rec, synapse_type = 'AMPA')
s_noise.connect (True)
s_noise.weight = np.random.normal(loc=1, scale=mm_lvl, size=len (s_noise))*9
s_noise.C_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_noise))* 1.5*pF             
s_noise.I_tau_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_noise))* 0.1*pA      
s_noise.I_wo_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_noise))* 50*pA       
s_noise.I_g_syn_shunt = np.random.normal(loc=1, scale=mm_lvl, size=len(s_noise))* 10*pA

spikes_in = SpikeMonitor(L_in)
spikes_hid = SpikeMonitor (L_hid)
spikes_rec = SpikeMonitor(L_rec)
spikes_inh = SpikeMonitor(L_inh)
state_rec = StateMonitor (L_rec,variables=['Iin_clip'], record=(True),dt = 0.48828125*ms)
network.add (L_in, L_hid, L_rec, L_inh, PG_noise,
             spikes_in, spikes_hid, spikes_rec, 
             s_in, s_hr, s_rec, s_ei, s_inh,s_noise)

print ('Time to create the network = ' + str (time.time()-t2))

counter = 0
counter_acc_rec = 0
DIMFR = []
DIMFR_ff = []
for j in range (1):
    for i in range (len(trial_cond[j+7])):
        indices = spike_i[i][j+7]
        times = spike_t[i][j+7]*ms        
        L_in.set_spikes(indices,times+(duration*i)+j*(duration*9))
        
        network.run (duration)
        
        ff_spikes = get_rates(spikes_hid)
        mean_rates = []
        for h in np.arange (0,9,1):
            box_sum = 0
            for t in np.arange (0,n_class,1):
                box_sum += ff_spikes [h*n_class+t]
            box_sum = box_sum/n_class
            mean_rates = np.concatenate ([mean_rates,[box_sum]])
        mean_rates = list (mean_rates)
        max_index = mean_rates.index(max(mean_rates))
        if max_index == trial_cond[j+7][i]:
            DIMFR_ff.append (extract_DIMFR(mean_rates))
        rec_spikes = get_rates(spikes_rec)
        mean_rates = []
        for h in np.arange (0,9,1):
            box_sum = 0
            for t in np.arange (0,n_class,1):
                box_sum += rec_spikes [h*n_class+t]
            box_sum = box_sum/n_class
            mean_rates = np.concatenate ([mean_rates,[box_sum]])
        mean_rates = list (mean_rates)
        max_index = mean_rates.index(max(mean_rates))
        if max_index == trial_cond[j+7][i]:
            counter_acc_rec += 1
            DIMFR.append(extract_DIMFR(mean_rates))
        counter+=1
        print("Showing %d, recognized as %d" % (trial_cond[j+7][i], max_index))
        
        
        