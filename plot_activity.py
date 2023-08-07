"""

@author: lparrilla
"""
from matplotlib import pyplot as plt
import numpy as np

def get_activity_exp_continuous_normalized(timestamps_ms, exp_time_constant_ms=40, dt_ms = 0.1):
    '''
    Method to convolve neural spiketimes with an exponential decaying kernel to preserve dimensionality
    
    Parameters : 
        timestamps_ms : neural spike times
        
    Returns :
        x,y : temporal evolution of mean firing rate
    '''
    t_curr = timestamps_ms[0]
    x = []
    y = []
    a_curr = 0
    for timestamp in timestamps_ms:
        while timestamp >= t_curr:
            x.append(t_curr)
            y.append(a_curr)
            t_curr += dt_ms
            a_curr = a_curr*np.exp(-dt_ms/exp_time_constant_ms)        
        a_curr += 1        
    x = np.array(x)
    y = np.array(y)
    return [x, y/(exp_time_constant_ms*1e-03)]

# EXTRACT NEURAL INDICES AND SPIKE TIMES FROM MONITORS AFTER THE SIMULATON HAS BEEN RUN

sp_i = spikemon.i
sp_t = spikemon.t/ms
sp_i_rec = recmon.i
sp_t_rec = recmon.t/ms
div_box_i = [[],[],[],[],[],[],[],[],[]]
div_box_t = [[],[],[],[],[],[],[],[],[]]
for i in range (len(sp_i)):
    div_box_i[int(sp_i[i]/50)].append (sp_i[i])
    div_box_t[int(sp_i[i]/50)].append (sp_t[i])
div_box_i_rec = [[],[],[],[],[],[],[],[],[]]
div_box_t_rec = [[],[],[],[],[],[],[],[],[]]
for i in range (len(sp_i_rec)):
    div_box_i_rec[int(sp_i_rec[i]/50)].append (sp_i_rec[i])
    div_box_t_rec[int(sp_i_rec[i]/50)].append (sp_t_rec[i])
    
# PLOTTING

plot1 = plt.subplot2grid((24, 3), (0, 0), rowspan = 12, colspan=3)
plot1.plot (spikemon.t/ms,spikemon.i+124,'k',marker=',',linestyle='none',markersize = 1)
plot1.plot (recmon.t/ms,recmon.i+124,'r',marker=',',linestyle='none',markersize = 1)
plot1.plot (inhmon.t/ms,inhmon.i+450+124,'b',marker=',',linestyle='none',markersize = 1)
plot1.plot (inpmon.t/ms,inpmon.i,'green',marker=',',linestyle='none',markersize = 1)
for i in range (10):
    plt.axhline(50*i+124)
    plt.axvline((duration/ms)*i)
plt.axhline (0)
plt.axhline(450+124+90)
y_ticks = [0,124,124+50,124+100,124+150,124+200,124+250,124+300,124+350,124+400,124+450]
for i in range (len(y_ticks)):
    if i == 0:
        y_ticks[i]=y_ticks[i]+62
    else:
        y_ticks[i]=y_ticks[i]+25
y_lab = ['Inp',0,1,2,3,4,5,6,7,8,'Inh']
x_t = np.arange(0,9000,1000)
x_lab = np.arange (0,9,1)
focus = 3
box_str = []
for i in range (9):
    box_str.append (str(trial_cond[0][i]))  
ticks = [i * (duration/ms) for i in range(9)]
plt.xticks (ticks,box_str, fontsize = 20)
plt.yticks(y_ticks,y_lab)
plt.yticks (fontsize = 15)
plot1.set_ylabel ('Neural Groups', fontsize = 20)
plot2 = plt.subplot2grid((24, 3), (13, 0), rowspan=3, colspan=3)
plt.yticks (fontsize = 15)
x,y = get_activity_exp_continuous_normalized(inhmon.t/ms)
plot2.plot (x,y/90,'b')
y_ticks = [20]
y_lab = [20]
plt.yticks(y_ticks,y_lab,fontsize=15)
plot3 = plt.subplot2grid((24, 3), (16, 0), rowspan=3, colspan=3)
plt.yticks(y_ticks,y_lab,fontsize=15)
plot4 = plt.subplot2grid((24, 3), (19, 0), rowspan=3, colspan=3)
plt.yticks(y_ticks,y_lab,fontsize=15)
plt.ylabel('Firing Rates [Hz]', fontsize = 20)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.035, 1.2)
for i in range (9):
    if i == focus:
        x, y = get_activity_exp_continuous_normalized(div_box_t[i])
        plot4.plot(x, y/50,'k',alpha=1,lw=1, label='FF')
        x, y = get_activity_exp_continuous_normalized(div_box_t_rec[i])
        plot3.plot(x, y/50,'r',alpha=1,lw=1, label = 'WTA')
    else:
        x, y = get_activity_exp_continuous_normalized(div_box_t[i])
        plot4.plot(x, y/50,'k',alpha = 0.2,lw=0.5)
        x, y = get_activity_exp_continuous_normalized(div_box_t_rec[i])
        plot3.plot(x, y/50,'r',alpha = 0.2,lw=0.5)
plot5 = plt.subplot2grid((24, 3), (22, 0), rowspan=3, colspan=3)
plt.yticks(y_ticks,y_lab,fontsize=15)
x,y = get_activity_exp_continuous_normalized(inpmon.t/ms)
plot5.plot (x,y/124,'g')
plot1.get_shared_x_axes().join(plot1, plot2, plot3, plot4,plot5)
for i in range (10):
    plot1.axvline((duration/ms)*i)
    plot2.axvline((duration/ms)*i)
    plot3.axvline((duration/ms)*i)
    plot4.axvline((duration/ms)*i)
    plot5.axvline((duration/ms)*i)
plot5.set_xlabel ('Time [ms]', fontsize = 20)
plt.xticks(fontsize=15)



