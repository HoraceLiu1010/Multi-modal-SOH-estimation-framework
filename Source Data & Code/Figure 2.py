import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

def normal (x,mu,sigma):
    pdf = np.exp( -1*((x-mu)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi)*sigma)
    return pdf

path = '...\\Fig 2\\'
path_save = '...\\Fig 2\\'

# In[Fig2 a~d]
data = pd.read_csv(path+'Figure 2(a~d).csv')
xlabel = ['Mileage (km)','Duration (days)','Use intensity (km/day)','Average speed (km/h)']

for i in range(1,5):
    plt.figure(figsize=(3,2),dpi=400)
    sns.histplot(data = data.iloc[:,i], bins=20, kde=False,color='#22BABB')
    plt.ylabel('Count')
    plt.xlabel(xlabel[i-1])
    if i ==1:
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.gca().ticklabel_format(useMathText=True)
    plt.tight_layout()
    # plt.savefig(path_save+'Figure 2(a).pdf', format='pdf')
    # plt.savefig(path_save+'Figure 2(a).png', format='png')
    plt.show()
    
# In[Fig2 e~f]
data = pd.read_csv(path+'Figure 2(e~f).csv')

plt.figure(figsize=(3,2),dpi=400)
ax1 = sns.histplot(data = data.iloc[:,1], color='#22BABB',bins=50, kde=False)
plt.xlabel('Charge start SOC (%)')
ax2 = ax1.twinx()
(mu,sigma) = stats.norm.fit(data.iloc[:,1])
plt.axvline(mu,color='#FA7F08',linestyle='--',linewidth=3)
x = np.linspace(mu-6*sigma, mu+6*sigma,100)
y = normal(x,mu,sigma)
ax2.plot(x,y,color='#E53935',label='Charging start')
ax1.set_ylabel('Count')
ax2.set_ylabel('')
ax2.set_yticks([])
plt.xlim(0,100)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plt.gca().ticklabel_format(useMathText=True)
plt.tight_layout()
# plt.savefig(path+'Figure 2(e).png', format='png')
# plt.savefig(path+'Figure 2(e).pdf', format='pdf')
plt.show()

plt.figure(figsize=(3,2),dpi=400)
sns.histplot(data = data.iloc[:,2], color='#22BABB',bins=20, kde=False)
plt.xlabel('Charge end SOC (%)')
plt.ylabel('Count')
plt.xlim(0,100)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().ticklabel_format(useMathText=True)
plt.tight_layout()
# plt.savefig(path+'Figure 2(f).png', format='png')
# plt.savefig(path+'Figure 2(f).pdf', format='pdf')
plt.show()

# In[Fig2 g]
data1 = pd.read_csv(path+'Figure 2(g)_capacity.csv')
data2 = pd.read_csv(path+'Figure 2(g)_SOC.csv')

# plt.figure(figsize = (4,3),dpi=400)
fig,ax1 = plt.subplots(figsize = (6,2.2),dpi=400)
ax1.scatter( data2.iloc[:,1] ,data2.iloc[:,2],s=2,c='#22BABB')
ax1.set_ylabel('SOC (%)',color='#22BABB')
ax1.set_xlabel('Sample')

ax2 = ax1.twinx()
ax2.bar(data1.iloc[:,1],data1.iloc[:,2],width=10,color='#FA7F08')
ax2.set_ylabel('Partial charging capacity (Ah)',color='#FA7F08',fontsize=8)

plt.tight_layout()
# plt.savefig(path+'Figure 2(g).png', format='png')
# plt.savefig(path+'Figure 2(g).pdf', format='pdf')
plt.show()

# In[Fig2 h]
data = pd.read_csv(path+'Figure 2(h).csv')
soh = data.iloc[:,2].astype('float')
mile = data.iloc[:,1].astype('float')
poly = np.polyfit(mile,soh,3)
poly_para = np.poly1d(poly)
y = poly_para(mile)

plt.figure(figsize=(6,2.5),dpi=400)
plt.scatter(mile,soh,s=2,color='#22BABB')
plt.plot(mile,y,'--',c='#FA7F08',linewidth = 3,label='Cubic spline fitting')
plt.legend(loc=1)
plt.xlabel('Mileage (km)')
plt.ylabel('SOH')
plt.tight_layout()
# plt.savefig(path+'Figure 2(h).png', format='png')
# plt.savefig(path+'Figure 2(h).pdf', format='pdf')
plt.show()


plt.figure(figsize=(4,3),dpi=400)
plt.plot(mile,soh,'.',color='#22BABB')
plt.plot(mile,y,'--',c='#FA7F08',linewidth = 3,label='Cubic spline fitting')
plt.xlabel('Mileage (km)')
plt.ylabel('SOH')
plt.tight_layout()
# plt.savefig(path+'Figure 2(tmp).png', format='png')
# plt.savefig(path+'Figure 2(tmp).pdf', format='pdf')
plt.show()

# In[Fig2 i]
data = pd.read_csv(path+'Figure 2(i).csv')

plt.figure(figsize=(3,2.5),dpi=400)
for i in range(len(data)):
    tmp1 = [i,i]
    tmp2 = [data.iloc[i,1],data.iloc[i,2]]
    plt.plot(tmp1,tmp2,'-' ,color='#FA7F08')
    plt.plot(i,tmp2[0],'x' ,color='#22BABB')
    plt.plot(i,tmp2[1],'^', color='#22BABB')
    
plt.plot(tmp1,tmp2,'-' ,color='#FA7F08',label='SOC error')
plt.plot(i,tmp2[0],'x' ,color='#22BABB',label='BMS SOC')
plt.plot(i,tmp2[1],'^', color='#22BABB',label='OCV corected SOC')
plt.xlabel('Sample')
plt.ylabel('SOC (%)')
plt.legend(loc=4,fontsize = 8)
plt.tight_layout()
# plt.savefig(path+'Figure 2(i).png', format='png')
# plt.savefig(path+'Figure 2(i).pdf', format='pdf')
plt.show()

# In[Fig2 j]
data = pd.read_csv(path+'Figure 2(j).csv')
data_group = list(data.groupby(by='Sample'))

fig = plt.figure(figsize=(3.8,3.8),dpi=400)
ax = fig.add_subplot(111,projection='3d')
data_out = []
for i in range(len(data_group)):
    # i = 0
    tmp = data_group[i][1].iloc[:,1:].values
    ys = tmp[:,1]
    xs = tmp[:,0]
    zs = tmp[:,2]
    ax.plot(xs,ys,zs,linewidth=1)
    plt.tight_layout()
ax.view_init(elev=20,azim=-50)
ax.set_xlabel('Voltage (V)',labelpad=8)
ax.set_ylabel('Sample',labelpad=8)

ax.tick_params(axis='z',pad=1)
plt.tight_layout()
# plt.savefig(path+'Figure 2(j).png', format='png')
# plt.savefig(path+'Figure 2(j).pdf', format='pdf')
plt.show()




