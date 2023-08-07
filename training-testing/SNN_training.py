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

def filter_net (network, syn2train):
    '''
    Method to optimize network time to train
    
    Parameters:
        network : SNN object
        syn2train : type of synapse to train
        
    Returns :
        network : network with essential synapses and locked initial learning rate for synapses alredy trained
    '''
    if syn2train == 'ff':
        network.remove (L_rec, L_teach_rec, s_hr, s_rec, s_teach_rec, spikes_rec, spikes_teach_rec, state_hr, state_rec)
        network['TRI_STDP0'].nuEEpost = 0.001

    elif syn2train == 'hr':
        network.remove (s_rec, spikes_teach_ff, s_teach_ff, state_ff, state_rec)
        network['TRI_STDP0'].nuEEpost = 0.0000001
        network['TRI_STDP0'].nuEEpre = 0.0000001
        network['TRI_STDP0'].weight = 550
        network['TRI_STDP0'].w_plast = np.load ('w_plast_ff.npy')
        network['TRI_STDP1'].nuEEpost = 0.001
    elif syn2train == 'rec':
        network.remove (state_ff, state_hr, spikes_teach_ff, s_teach_ff)
        network['TRI_STDP0'].nuEEpost = 0.0000001
        network['TRI_STDP0'].nuEEpre = 0.0000001
        network['TRI_STDP0'].weight = 300
        network['TRI_STDP0'].w_plast = np.load ('w_plast_ff.npy')
        network['TRI_STDP1'].nuEEpost = 0.0000001
        network['TRI_STDP1'].nuEEpre = 0.0000001
        network['TRI_STDP1'].weight = 250
        network['TRI_STDP1'].w_plast = np.load ('w_plast_hr.npy')
        network['TRI_STDP2'].nuEEpost = 0.001
    else:
        print ('Provide a valid synapse to train')
    return (network)

def set_teach (network, syn2train, class2train, n_class):
    '''
    Method to set teacher signal firing rate according to class presented and synapse to train
    
    Parameter:
        network : SNN object
        syn2train : type of synapse to train 'ff', 'hr', 'rec'
        class2train : number of class currently presented to the network
        n_class : how many neurons code for one class
    
    Returns :
        rate_matrix : rate matrix to be passed to the teacher signal
    '''
    rate_matrix = np.zeros (n_class*9)
    for i in range (n_class*9):
        if i >= class2train*(n_class) and i < class2train*(n_class)+n_class: 
            rate_matrix[i] = 1000
        else:
            rate_matrix[i] = 0
    return (rate_matrix*Hz)

def save_intermediate (w_plast, idx, syn):
    '''
    Method to save trained plastic weight after every trial presentation
    
    Parameters :
        w_plast : plastic weight to be saved
        idx : number of training tial
        syn : type of synapse that is currently trained
        
    Returns :
        none : output is the trained plastic weight saved as an .npy into the current working directory
    '''
    
    if syn == 'ff':
        np.save (str(dataset)+'_FF_ep'+str(nep)+'_step'+str(idx))
    elif syn == 'hr':
        np.save (str(dataset)+'_HR_ep'+str(nep)+'_step'+str(idx))
    elif syn == 'rec':
        np.save (str(dataset)+'_REC_ep'+str(nep)+'_step'+str(idx))
    else:
        print ('Provide a valid synapse type')    

mm_lvl = 0.05
n_class = 50
n_batch = 1
nep = 2
dataset = 25
lag = 150
trial_cond = np.tile(np.arange(9), (10, 1))
trial2train  = 7
t1 = time.time()        
n_in, spike_i,spike_t,duration = spikenize_dataset (dataset,nep, l_bin=1)
t2 = time.time ()
print ('Time to preprocess = ' +str (t2-t1))
tot_time = duration+lag*ms
syn2train = 'ff'
# =============================================================================
# NEURAL NETWORK
# =============================================================================

'''
Instantiate a Spiking Neural Network to decode 9 spatial positions given an input dataset of single cells recordings

The network is composed by:
    1 Input layer as virtual neurons (L_in)
    1 Hidden layer (L_hid)
    1 Recurrent pools (L_rec)
    2 Teaching layers as virtual neurons
    
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
L_hid = chip.get_neurons(n_class*9, 'Core_2')
L_rec = chip.get_neurons(n_class*9, 'Core_1')
L_teach_ff = PoissonGroup(n_class*9, rates = 0*Hz, name='L_teach_ff')
L_teach_rec = PoissonGroup(n_class*9, rates=0*Hz, name='L_teach_rec')

s_in = chip.add_connection (L_in, L_hid, synapse_type = 'TRI_STDP')
s_in.connect (True)
s_in.weight = 1
s_in.w_plast = 0
s_in.nuEEpre = pre_lr_ff
s_in.nuEEpost = post_lr_ff
s_in.I_tau_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*50*pA
s_in.I_g_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*50*pA
s_in.I_wo_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*50*pA
s_in.C_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_in))*1.5 * pF
    
s_hr = chip.add_connection(L_hid, L_rec, synapse_type = 'TRI_STDP')
s_hr.connect (condition='int(i/n_class) == int(j/n_class)')
s_hr.weight = 1
s_hr.w_plast = 0
s_hr.nuEEpre = pre_lr_hr
s_hr.nuEEpost = post_lr_hr
s_hr.I_tau_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*50*pA
s_hr.I_g_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*50*pA
s_hr.I_wo_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*50*pA
s_hr.C_syn_exc_tripstdp = np.random.normal(loc=1, scale=mm_lvl, size=len(s_hr))*1.5 * pF

s_rec = chip.add_connection(L_rec, L_rec, synapse_type = 'TRI_STDP')
s_rec.connect (condition='int(i/n_class) == int(j/n_class) and i!=j')
s_rec.weight = 1
s_rec.w_plast = 0
s_rec.nuEEpre = pre_lr_rec
s_rec.nuEEpost = post_lr_rec
s_rec.I_tau_syn_exc_tripstdp = np.random.normal(loc=1, scale=0.2, size=len(s_rec))*50*pA
s_rec.I_g_syn_exc_tripstdp = np.random.normal(loc=1, scale=0.2, size=len(s_rec))*50*pA
s_rec.I_wo_syn_exc_tripstdp = np.random.normal(loc=1, scale=0.2, size=len(s_rec))*50*pA
s_rec.C_syn_exc_tripstdp = np.random.normal(loc=1, scale=0.2, size=len(s_rec))*1.5 * pF

s_teach_ff = chip.add_connection (L_teach_ff, L_hid, synapse_type = 'AMPA')
s_teach_ff.connect ('i==j')
s_teach_ff.weight = 1000

s_teach_rec = chip.add_connection(L_teach_rec,L_rec, synapse_type='AMPA')
s_teach_rec.connect ('i==j')
s_teach_rec.weight = 1000

spikes_in = SpikeMonitor(L_in)
spikes_hid = SpikeMonitor (L_hid)
spikes_rec = SpikeMonitor(L_rec)
spikes_teach_ff = SpikeMonitor (L_teach_ff)
spikes_teach_rec = SpikeMonitor(L_teach_rec)

state_ff = StateMonitor (s_in, variables='w_plast', record=True, dt = duration)
state_hr = StateMonitor (s_hr, variables='w_plast', record=True, dt=duration)
state_rec = StateMonitor(s_rec, variables='w_plast', record=True, dt=duration)

network.add (L_in, L_hid, L_rec, L_teach_ff, L_teach_rec,
             spikes_in, spikes_hid, spikes_rec, spikes_teach_ff, spikes_teach_rec,
             state_ff, state_hr, state_rec,
             s_in, s_hr, s_rec, s_teach_ff, s_teach_rec)

print ('Time to create the network = ' + str (time.time()-t2))

network = filter_net (network, syn2train)

for batch in np.arange (n_batch):
    print ('Training on batch n' + str(batch))
    for i in range (len(trial_cond)):
        if i == trial2train :
            break
        else:
            print ('TRAINING ON TRIAL'+str(i))
            for j in trial_cond[i]:
                print ('Presenting condition ' + str(j))
                t4 = time.time()
                indices = spike_i[j][i]
                times = spike_t[j][i]*ms 
                L_in.set_spikes (indices, 
                               times+(j*tot_time
                            +tot_time*9*i
                            +batch*tot_time*9*trial2train 
                              ))
                if syn2train == 'ff':
                    L_teach_ff.rates = set_teach(network, syn2train, j, n_class)
                elif syn2train == 'hr' or syn2train == 'rec':
                    L_teach_rec.rates = set_teach (network,syn2train, j, n_class)
    
                network.run (duration)
                L_teach_ff.rates = 0*Hz
                network.run(lag*ms)
                t5 = time.time() 
                
                if syn2train == 'ff':
                    save_intermediate (s_in.w_plast,i,'ff')
                elif syn2train == 'hr':
                    save_intermediate (s_hr.w_plast,i,'hr')
                elif syn2train == 'rec':
                    save_intermediate (s_rec.w_plast,i,'rec')

                print ('Time to train on this sample = ' +str(t5-t4))
            
         