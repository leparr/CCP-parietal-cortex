"""

@author: lparrilla
"""

import numpy as np
import matplotlib.pyplot as plt

'''
Code to plot the synaptic kernel of triplet STDP

'''
t1 = np.linspace(-50, 0, 500)  
t2 = np.linspace(0, 50, 500)  
tau_p = 15  
tau_n = 20  
A_p = 0.8  
A_n = -0.8  
kernel1 = A_n * np.exp(t1 / tau_n)*100
kernel2 = A_p * np.exp(-t2 / tau_p)*100
plt.plot(t1, kernel1, color='black', label=r'$\Delta t_2$ = 1', lw=3)
plt.plot(t2, kernel2, color='black', lw=3)
delta_t2_1 = 5
delta_t2_2 = 10
line1 = A_p * np.exp(-(t2 + delta_t2_1) / tau_p)*100
line2 = A_p * np.exp(-(t2 + delta_t2_2) / tau_p)*100
plt.plot(t2, line1, color='blue', label=r'$\Delta t_2 = {}$'.format(delta_t2_1), lw=3)
plt.plot(t2, line2, color='red', label=r'$\Delta t_2 = {}$'.format(delta_t2_2), lw=3)
plt.xlabel('$\Delta t_1$ [ms]', fontsize = 20)
plt.ylabel('$\Delta w$ [%]', fontsize = 20, labelpad=-20)
plt.axhline(y=0, color='black', ls = '--', linewidth=1.5)
plt.axvline(x=0, color='black', ls = '--', linewidth=1.5)
plt.xlim(-50, 50)
plt.xticks(fontsize=20, ticks=[-50,0,50]) 
plt.yticks(fontsize=20, ticks=[-80,0,80])  
plt.legend(fontsize = 15)
plt.show()
