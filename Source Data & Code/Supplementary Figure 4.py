import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

path = 'E:\\NC-矢量图\\Source data of pictures\\Supplementary Figure 4\\'
path_save = '...\\SUPP Figure 4\\'

# In[SUPP Fig4 a]
lr_e = -0.05
lr_e_warm = -0.06
lr_k = 0.00001
lr_b = 0.0001
epochs = 100
x_epoch = np.arange(1,epochs+1)
y_lr_decay = []
for i in range(len(x_epoch)):
    if i<1:
        y_lr_decay.append (i*lr_k+lr_b)
    else:
        y_lr_decay.append (y_lr_decay[i-1]*math.exp(lr_e))

y_lr_warm = []
for i in range(len(x_epoch)):
    if i<10:
        y_lr_warm.append (i*lr_k+0.00001)
    else:
        y_lr_warm.append (y_lr_warm[i-1]*math.exp(lr_e_warm))

plt.figure(figsize=(3.5,3),dpi=400)
plt.plot(x_epoch,y_lr_decay,label='Exponential decline')
plt.plot(x_epoch,y_lr_warm,label='Warm-up')
plt.axhline(0.00002,color='#845EC2',linestyle='-',linewidth=2,label='Fixed')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.gca().ticklabel_format(useMathText=True)
plt.legend(fontsize=10)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(a).png', format='png')
# plt.savefig(path+'SUPP Figure 4(a).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 a]
data = pd.read_csv(path+'SUPP Figure 4(b, f~h).csv')
title = ['Warm-up','Fixed','Exponential decline']
error_pd = pd.DataFrame()
for i in range(3):
    error = abs(data.iloc[:,i+2] - data.iloc[:,1])
    error_pd_1 = pd.DataFrame()
    error_pd_1['error'] = error
    error_pd_1['Model'] = title[i]
    error_pd = pd.concat([error_pd,error_pd_1])

plt.figure(figsize = (7,3),dpi=400)
sns.violinplot(x=error_pd['Model'], y=error_pd['error'], width = 0.3, palette=sns.color_palette())
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(b).png', format='png')
# plt.savefig(path+'SUPP Figure 4(b).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 c]
data = pd.read_csv(path+'SUPP Figure 4(c).csv')

loss = data.iloc[:,1]
val_loss = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(loss,label='Training loss')
plt.plot(val_loss,label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=10)
plt.title('Warm-up')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(c).png', format='png')
# plt.savefig(path+'SUPP Figure 4(c).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 d]
data = pd.read_csv(path+'SUPP Figure 4(d).csv')

loss = data.iloc[:,1]
val_loss = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(loss,label='Training loss')
plt.plot(val_loss,label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=10)
plt.title('Fixed')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(d).png', format='png')
# plt.savefig(path+'SUPP Figure 4(d).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 e]
data = pd.read_csv(path+'SUPP Figure 4(e).csv')

loss = data.iloc[:,1]
val_loss = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(loss,label='Training loss')
plt.plot(val_loss,label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize=10)
plt.title('Exponential decline')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(e).png', format='png')
# plt.savefig(path+'SUPP Figure 4(e).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 f]
data = pd.read_csv(path+'SUPP Figure 4(b, f~h).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,2]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('Warm-up')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(f).png', format='png')
# plt.savefig(path+'SUPP Figure 4(f).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 g]
data = pd.read_csv(path+'SUPP Figure 4(b, f~h).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,3]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('Fixed')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(g).png', format='png')
# plt.savefig(path+'SUPP Figure 4(g).pdf', format='pdf')
plt.show()

# In[SUPP Fig4 h]
data = pd.read_csv(path+'SUPP Figure 4(b, f~h).csv')

soh = data.iloc[:,1]
soh_est = data.iloc[:,3]

plt.figure(figsize = (3.5,3),dpi=400)
plt.plot(soh,soh)
plt.plot(soh,soh_est,'.')
plt.xlabel('SOH labels')
plt.ylabel('SOH estimations')
plt.title('Exponential decline')
plt.tight_layout()
# plt.savefig(path+'SUPP Figure 4(h).png', format='png')
# plt.savefig(path+'SUPP Figure 4(h).pdf', format='pdf')
plt.show()






