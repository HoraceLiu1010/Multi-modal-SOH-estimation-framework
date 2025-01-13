import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

path = '...\\Fig 4\\'
path_save = '...\\Fig 4\\'

# In[Fig4 a]
data = pd.read_csv(path+'Figure 4(a).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('3 modalities')
plt.tight_layout()
# plt.savefig(path_save+'Figure 4(a).pdf', format='pdf')
# plt.savefig(path_save+'Figure 4(a).png', format='png')
plt.show()

# In[Fig4 b]
data = pd.read_csv(path+'Figure 4(b).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]
error = abs(soh-soh_est)

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('SVR')
plt.tight_layout()
# plt.savefig(path_save+'Figure 4(b).pdf', format='pdf')
# plt.savefig(path_save+'Figure 4(b).png', format='png')
plt.show()


# In[Fig4 c]
data = pd.read_csv(path+'Figure 4(c).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]
error = abs(soh-soh_est)

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('RFR')
plt.tight_layout()
# plt.savefig(path_save+'Figure 4(c).pdf', format='pdf')
# plt.savefig(path_save+'Figure 4(c).png', format='png')
plt.show()

# In[Fig4 d]
data = pd.read_csv(path+'Figure 4(d).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]
error = abs(soh-soh_est)

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('GPR')
plt.tight_layout()
# plt.savefig(path_save+'Figure 4(d).pdf', format='pdf')
# plt.savefig(path_save+'Figure 4(d).png', format='png')
plt.show()

# In[Fig4 e]
data = pd.read_csv(path+'Figure 4(e).csv')

plt.figure(figsize = (7,3),dpi=400)
sns.violinplot(x=data['model'], y=data['error'], width = 0.4, palette=sns.color_palette())
plt.xlabel('Models')
plt.ylabel('error')
plt.tight_layout()
# plt.savefig(path_save+'Figure 4(e).pdf', format='pdf')
# plt.savefig(path_save+'Figure 4(e).png', format='png')
plt.show()





