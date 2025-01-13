import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

path = '...\\Fig 5\\'
path_save = '...\\Fig 5\\'

# In[Fig5 a]
data = pd.read_csv(path+'Figure 5(a).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('3 modalities')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(a).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(a).png', format='png')
plt.show()

# In[Fig5 b]
data = pd.read_csv(path+'Figure 5(b).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('2 modalities')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(b).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(b).png', format='png')
plt.show()

# In[Fig5 c]
data = pd.read_csv(path+'Figure 5(c).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('1 modalities')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(c).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(c).png', format='png')
plt.show()

# In[Fig5 d]
data = pd.read_csv(path+'Figure 5(d).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('3 modalities with CNN')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(d).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(d).png', format='png')
plt.show()

# In[Fig5 e]
data = pd.read_csv(path+'Figure 5(e~g).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('3 modalities with RNN')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(e).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(e).png', format='png')
plt.show()

# In[Fig5 f]
data = pd.read_csv(path+'Figure 5(e~g).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,3]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('3 modalities with LSTM')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(f).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(f).png', format='png')
plt.show()

# In[Fig5 g]
data = pd.read_csv(path+'Figure 5(e~g).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,4]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('3 modalities with FNN')
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(g).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(g).png', format='png')
plt.show()


# In[Fig5 h]
data = pd.read_csv(path+'Figure 5(h).csv')

plt.figure(figsize = (7,3),dpi=400)
sns.violinplot(x=data['model'], y=data['error'], width = 0.8,palette=sns.color_palette())
plt.xlabel('Models')
plt.ylabel('error')
plt.xticks(rotation=15)
plt.tight_layout()
# plt.savefig(path_save+'Figure 5(h).pdf', format='pdf')
# plt.savefig(path_save+'Figure 5(h).png', format='png')
plt.show()






