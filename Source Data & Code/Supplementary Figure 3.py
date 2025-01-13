import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

def normal (x,mu,sigma):
    pdf = np.exp( -1*((x-mu)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)*sigma)
    return pdf

path = '...\\Supplementary Figure 3\\'
path_save = '...\\Supplementary Figure 3\\'

# In[SUPP Fig3 c]
data = pd.read_csv(path+'SUPP Figure 3(c).csv')

plt.figure(figsize=(5,3),dpi=400)
plt.subplot(211)
plt.plot(data.iloc[:,0])
plt.ylabel('Current (A)')
plt.xticks([])
plt.subplot(212)
plt.plot(data.iloc[:,1])
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 3(c).png', format='png')
# plt.savefig(path+'SUPP Figure 3(c).pdf', format='pdf')
plt.show()


# In[SUPP Fig3 d]
data = pd.read_csv(path+'SUPP Figure 3(d).csv')

plt.figure(figsize=(4.5,3.5),dpi=400)
plt.plot(data.iloc[:,1],'o',color = '#ED708D',label='Discharge capacity')
plt.plot(data.iloc[:,0],'*',color = '#922091',label='Capacity labels')
plt.ylim([52,58])
plt.xlabel('Sample')
plt.ylabel('Capacity (Ah)')
plt.legend(loc=1)
plt.twinx()
plt.plot(data.iloc[:,2],color = '#068E38',label='Error')
plt.ylabel('Error (Ah)')
plt.legend(loc=3)
plt.ylim([-0.5,2])
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 3(d).png', format='png')
# plt.savefig(path+'SUPP Figure 3(d).pdf', format='pdf')
plt.show()







