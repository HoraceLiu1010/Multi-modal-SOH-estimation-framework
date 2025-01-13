import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

path = '...\\Fig 3\\'
path_save = '...\\Fig 3\\'

# In[Fig3 a]
data = pd.read_csv(path+'Figure 3(a).csv')

fig,ax1 = plt.subplots(1,1,figsize=(4,3),dpi=400)
ax2 = ax1.twinx()
ax1.plot(data.iloc[:,1],color='#845EC2')
ax1.set_ylabel('Voltage (V)',color='#845EC2')
ax1.set_xlabel('Sample')
ax1.set_ylim([3.2,4.3])
ax2.plot(-1*(data.iloc[:,2]),'--',color='#FF6F91')
ax2.set_ylabel('Current (A)',color='#FF6F91')
ax2.set_ylim([10,250])
plt.tight_layout()
# plt.savefig(path_save+'Figure 3(a).pdf', format='pdf')
# plt.savefig(path_save+'Figure 3(a).png', format='png')
plt.show()

# In[Fig3 b]
data = pd.read_csv(path+'Figure 3(b).csv')

x_voltage = np.arange(3900,4051)
cap = (data.iloc[:300,1]).astype('float32')
palette = plt.get_cmap('Blues') # summer_r
data_seq = data.iloc[:,2:].values

plt.figure(figsize = (4,3),dpi=400)
for i in range(len(cap)):
    c = (cap[i]-np.min(cap))/(np.max(cap)-np.min(cap))*255
    plt.plot(x_voltage,data_seq[i],color = palette(int(c)))
plt.scatter(x=[3900]*len(cap),y=[0]*len(cap),c=cap,s=0.1,cmap='Blues')
plt.colorbar()
plt.xlabel('Voltage (mV)')
plt.ylabel('Increment capacity (Ah)')
plt.tight_layout()
# plt.savefig(path_save+'Figure 3(b).pdf', format='pdf')
# plt.savefig(path_save+'Figure 3(b).png', format='png')
plt.show()

# In[Fig3 c]
data = pd.read_csv(path+'Figure 3(c).csv')
x_voltage = np.arange(3900,4051)
fig,ax1 = plt.subplots(figsize = (4,3),dpi=400)
ax1.plot(x_voltage,data.iloc[:,1],color='green')
ax1.set_xlabel('Voltage (mV)')
ax1.set_ylabel('Increment capacity(Ah)',color='green')
ax2 = ax1.twinx()
ax2.plot(x_voltage,data.iloc[:,2],color='r')
ax2.set_ylabel('Temperature (   )',color='r')
plt.tight_layout()
# plt.savefig(path_save+'Figure 3(c).pdf', format='pdf')
# plt.savefig(path_save+'Figure 3(c).png', format='png')
plt.show()

# In[Fig3 d]
data = pd.read_csv(path+'Figure 3(d).csv')

fig,ax1 = plt.subplots(1,1,figsize=(4,3),dpi=400)
ax2 = ax1.twinx()
ax1.plot(data.iloc[:,1],color='#845EC2',label='Max cell voltage')
ax1.plot(data.iloc[:,2],'--',color='#D65DB1',label='Min cell voltage')
ax1.set_ylabel('Cell voltage (V)')
ax1.set_ylim([3.5,4.3])
ax1.set_xlabel('Sample')
ax1.legend(loc=2,fontsize=10)
ax2.plot(data.iloc[:,3],'-',color='#FF6F91',label='Vd')
ax2.set_ylabel('Cell voltage range (V)',color='#FF6F91')
ax2.set_ylim([0.01,0.141])
plt.tight_layout()
# plt.savefig(path_save+'Figure 3(d).pdf', format='pdf')
# plt.savefig(path_save+'Figure 3(d).png', format='png')
plt.show()

# In[Fig3 e]
data = pd.read_csv(path+'Figure 3(e).csv')

plt.figure(figsize = (4,3),dpi=400)
ax = sns.heatmap(data.iloc[:,1:],cmap = 'Greens')
plt.xticks([])
plt.xlabel('Cell number')
plt.yticks([])
plt.ylabel('Voltage (V)')
plt.tight_layout()
# plt.savefig(path_save+'Figure 3(e).pdf', format='pdf')
# plt.savefig(path_save+'Figure 3(e).png', format='png')
plt.show()

# In[Fig3 f]
data = pd.read_csv(path+'Figure 3(f).csv')

plt.figure(figsize = (4,3),dpi=400)
for i in range(1,16):
    sns.kdeplot(data.iloc[:,i], fill=True)
plt.xlabel('Point features (normalized)')
plt.tight_layout()
# plt.savefig(path_save+'Figure 3(f).pdf', format='pdf')
# plt.savefig(path_save+'Figure 3(f).png', format='png')
plt.show()




