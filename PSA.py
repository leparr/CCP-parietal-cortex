"""

@author: lparrilla
"""

from matplotlib import pyplot as plt
import seaborn as sns

EPSP = state.Iin_clip

def PSD_grand_average (EPSP):   
    '''
    Method to extract the Power Spectral Density from EPSP traces
    
    Parameters :
        EPSP : the excitatory post synaptic potentials recorded from the excitatory clusters of the WTA
        
    Returns : 
        g_average : grand average of PSD performed on all neurons 
        signal :
        frequencies : frequencies on which the PSA is performed
        lenght :
        up : upper confidence interval for PSD grand average
        down : lower confidence interval for PSD grand average
        low : low frequencies PSD 0-30Hz
        central : low gamma frequencies PSD 30-50Hz
        high : high gamma frequencies PSD 50-100Hz
        very_high : very high frequencise PSD 100-200Hz
    '''
    g_average,signal,frequencies,lenght,up,down,low,central,high,very_high = PSD_grand_average(EPSP)

    fs = 4096
    signal = []
    d = []
    for i in range (450):
        frequencies = EPSP[i]
        (frequencies, S) = scipy.signal.periodogram(frequencies, fs, scaling='density')
        signal.append (S)
        d.append (frequencies)
    g_average = []
    std_box = []
    for i in range (len(signal[0])):
        p = []
        for j in range (len(signal)):
            p.append (signal[j][i])
        g_average.append (np.mean(p))
        std_box.append (np.std(p))
    lenght = 900
    main = array (g_average[:lenght])
    up = []
    down = []
    for i in range (lenght):
        up.append (g_average[i]+std_box[i])
        down.append (g_average[i]-std_box[i])
    for i in range (len(down)):
        if down[i]<0:
            down[i] = 0
    up = array (up)
    down = array(down)
    tot = []
    low = []
    central = []
    high = []
    very_high = []
    for i in range (len(frequencies)):
        tot.append (g_average[i])
        if frequencies[i]<30:
            low.append (g_average[i])
        elif frequencies[i]>30 and frequencies[i]<50:
            central.append(g_average[i])
        elif frequencies[i]> 50 and frequencies[i]<100:
            high.append(g_average[i])
        elif frequencies[i]>100 and frequencies[i]<200:
            very_high.append (g_average[i])
    lenght = 0
    for i in range (len(frequencies)):
        if frequencies[i]<200:
            lenght+=1
        else:
            break  
    return (g_average,signal,frequencies,lenght,up,down,low,central,high,very_high)
 
def PSD_heat (EPSP):
    '''
    Method to calulate individual PSD for every excitatory cluster
    
    Parameters : 
        EPSP : the excitatory post synaptic potentials recorded from the excitatory clusters of the WTA
    Returns :
        s : divided PSA for each WTA excitatory clusters
    '''
    
    b6 = []
    for k in range(9):
        box5 = []
        for i in range (len(EPSP[0])):
            p = []
            for j in range (50):
                p.append (EPSP[j+k*50][i])
            box5.append (np.mean(p))
        b6.append (box5)
    b7 = []
    for i in range (9):
        b7.append (b6[i][:200])
    box3 = []
    std_box = []
    p = []
    for i in range (9):
        p.append (signal[i*50:i*50+50])
    box3 = []
    for i in range (9):        
        boxx = [] 
        for k in range (len(p[0][0])):
            t = []
            for j in range (50):
                t.append (p[i][j][k])
            boxx.append (np.mean(t))    
        box3.append (boxx)
    s = []
    for i in range (9):
        s.append (box3[i][:450])
    return (s)

g_average,signal,frequencies,lenght,up,down,low,central,high,very_high = PSD_grand_average(EPSP)
s = PSD_heat (EPSP)

# =============================================================================
# PLOT POWER SPECTRAL ANALYSIS
# =============================================================================

ax1 = plt.subplot2grid((4, 9), (0, 0), rowspan = 4, colspan=4)
ax1.fill_between(frequencies[:lenght],up[:lenght],down[:lenght],alpha=.5)
ax1.plot (frequencies[:lenght],g_average[:lenght])
ax1.yaxis.offsetText.set_fontsize(15)
plt.xlabel('Frequency [Hz]',fontsize = 20)
plt.ylabel ('PSD[V**2/Hz]',fontsize = 20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)

ax2 = ax1.twinx()
ax2.yaxis.offsetText.set_fontsize(15)
ax2 = plt.fill_between (np.arange(0,31,1),(sum(low)/len(low)),alpha=0.5,color = 'gray',label='Lower Frequencies')
ax2 = plt.fill_between (np.arange(30,51,1),(sum(central)/len(central)),alpha=0.5,color='r',label='Low Gamma band')
ax2 = plt.fill_between (np.arange(50,101,1),(sum(high)/len(high)),alpha=0.5,color ='orange',label='High Gamma Band')
ax2 = plt.fill_between (np.arange(100,201,1),(sum(very_high)/len(very_high)),alpha=0.5, color ='purple',label='Higher Frequencies')
plt.yticks(fontsize=17)
legend(loc = 'upper right', fontsize = 15)

ticklab = []
for i in range (450):
    ticklab.append (round(frequencies[i],0))
tl = []
ar = np.arange (0,450,80)
for i in range (len(ar)):
    tl.append (ticklab[ar[i]])
tic = np.array(np.arange (0,450,80))
tl=[]
for i in tic:
    tl.append(ticklab[i])
ax3 = plt.subplot2grid((4, 9), (0, 5), rowspan = 4, colspan=4)
sns.heatmap (s,cmap='coolwarm',xticklabels = tl)
plt.xticks(tic,tl)
ax3.hlines([1,2,3,4,5,6,7,8],*ax3.get_xlim(),colors = 'k')
plt.xlabel('Frequency[Hz]',fontsize=20)
plt.ylabel('Class Averge',fontsize=20)
plt.xticks(rotation = 0, fontsize=20)
plt.yticks(rotation = 0, fontsize=20)
cbar = ax3.collections[0].colorbar
cbar = ax3.collections[0].colorbar
text_height = 0.5 * (cbar.vmax - cbar.vmin)
text_position = cbar.vmin + text_height
cbar.ax.text(1.6, text_position, 'PSD[V**2/Hz]', rotation=90, va='center', fontsize = 20)
minimo = round(np.min(s),1)
massimo = round(np.max (s),100)
cbar.set_ticks([minimo,massimo])
cbar.ax.tick_params(labelsize=20)
cbar.ax.yaxis.get_offset_text().set_fontsize(15)
plt.show()

