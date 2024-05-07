import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.models import Sequential,Model
from keras.layers import Dense, Input, concatenate
from sklearn import metrics
from tensorflow.keras import layers
import math
from sklearn.preprocessing import MinMaxScaler
tf.config.list_physical_devices('GPU')

from numpy.random import seed
seed(2024)
tf.random.set_seed(2024)

# In[import and split data]
path_feature = 'E:\\Multi-modal framework for SOH estimation\\Multi-modal_features\\' # Please replace the path here
data_point_pd = pd.read_csv(path_feature+'point_features.csv') 
data_seq_pd = pd.read_csv(path_feature+'151by2_V1.csv')
data_plane_pd = pd.read_csv(path_feature+'96by96_V1.csv')
useful = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]# 

col = data_point_pd.columns[useful]
data_point = data_point_pd.iloc[:,useful].values
scaler1 = MinMaxScaler()
data_point = scaler1.fit_transform(data_point)

data_seq = data_seq_pd.iloc[:,2:].values
scaler2 = MinMaxScaler()
data_seq = scaler2.fit_transform(data_seq).reshape(-1,151,2)

data_plane = data_plane_pd.iloc[:,1:].values
scaler3 = MinMaxScaler()
data_plane = scaler3.fit_transform(data_plane).reshape(-1,96,96)

Total_num = len(data_point)
Train_num = round(Total_num*0.8)
index = np.array(range(0,Total_num))
np.random.shuffle(index)
train_idx=index[:Train_num]
test_idx=index[Train_num:]

xt_point = data_point[train_idx]
xt_seq = data_seq[train_idx]
xt_plane = data_plane[train_idx]
xs_point = data_point[test_idx]
xs_seq = data_seq[test_idx]
xs_plane = data_plane[test_idx]

y = data_point_pd.iloc[:,1].values
y = y/np.max(y)
yt = y[train_idx]
ys = y[test_idx]

# In[Resnet 2D]
layer_dims = [2,2,2,2]
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Convolution2D(filter_num, kernel_size=(5,5), strides=stride, padding='same')#padding='same'表示自动适配使得输出的数据shape与输入的数据shape一样
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Convolution2D(filter_num, kernel_size=(5,5), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Convolution2D(filter_num, kernel_size=(1,1), strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output
def build_resblock(filter_num, blocks, stride=1):
    res_blocks = Sequential()
    res_blocks.add(BasicBlock(filter_num, stride))
    for _ in range(1, blocks):
        res_blocks.add(BasicBlock(filter_num, stride=1))
    return res_blocks
layer1 = build_resblock(16,  layer_dims[0])
layer2 = build_resblock(32, layer_dims[1], stride=2)
layer3 = build_resblock(32, layer_dims[2], stride=2)
layer4 = build_resblock(64, layer_dims[3], stride=2)

avgpool = layers.GlobalAveragePooling2D()
fc1 = layers.Dense(16,activation='relu')
fc2 = layers.Dense(8,activation='relu')
fc = layers.Dense(2)
plane_input = Input(shape=(96, 96, 1,))
x = layers.Convolution2D(16, kernel_size=(3,3), strides=1)(plane_input)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(x)
x = layer1(x)
x = layer2(x)
x = layer3(x)
x = layer4(x)
x = avgpool(x)
x = fc1(x)
x = fc2(x)
x = fc(x)

# In[Resnet 1D]
layer_dims_1D = [2,2,2,2]
class BasicBlock_1D(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock_1D, self).__init__()
        self.conv1 = layers.Convolution1D(filter_num, kernel_size=30, strides=stride, padding='same')#padding='same'表示自动适配使得输出的数据shape与输入的数据shape一样
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Convolution1D(filter_num, kernel_size=30, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Convolution1D(filter_num, kernel_size=1, strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output
def build_resblock_1D(filter_num, blocks, stride=1):
    res_blocks = Sequential()
    res_blocks.add(BasicBlock_1D(filter_num, stride))
    for _ in range(1, blocks):
        res_blocks.add(BasicBlock_1D(filter_num, stride=1))
    return res_blocks
layer1_1D = build_resblock_1D(16,  layer_dims_1D[0])
layer2_1D = build_resblock_1D(32, layer_dims_1D[1], stride=2)
layer3_1D = build_resblock_1D(64, layer_dims_1D[2], stride=2)
layer4_1D = build_resblock_1D(128, layer_dims_1D[3], stride=2)
avgpool_1D = layers.GlobalAveragePooling1D()
fc1_1D = layers.Dense(16,activation='relu')
fc2_1D = layers.Dense(8,activation='relu')
fc_1D = layers.Dense(2)
seq_input = Input(shape=(151, 2,))
x_1D = layers.Convolution1D(16, kernel_size=3, strides=1)(seq_input)
x_1D = layers.BatchNormalization()(x_1D)
x_1D = layers.Activation('relu')(x_1D)
x_1D = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x_1D)
x_1D = layer1_1D(x_1D)
x_1D = layer2_1D(x_1D)
x_1D = layer3_1D(x_1D)
x_1D = layer4_1D(x_1D)
x_1D = avgpool_1D(x_1D)
x_1D = fc1_1D(x_1D)
x_1D = fc2_1D(x_1D)
x_1D = fc_1D(x_1D)

# In[build model]
point_input = Input(shape=(xt_point.shape[1]))
model_point = BatchNormalization()(point_input)
model_point = Dense(8, activation='relu')(point_input)
dense_point = Dense(2, activation='relu')(model_point)

concat = concatenate([x,x_1D,dense_point])
concat_out = Dense(1, activation='sigmoid')(concat)

model = Model(inputs=[plane_input,seq_input,point_input],outputs=concat_out)
adam = Adam()
model.compile(loss='mean_absolute_percentage_error', optimizer=adam)
model.summary()


# In[warm up]
lr_e = -0.05
lr_k = 0.00001
lr_b = 0.00001
epochs = 100
def schedule(epoch, lr):
    if epoch<10:
        return epoch*lr_k+lr_b
    else:
        return lr*tf.math.exp(lr_e)
x_epoch = np.arange(1,epochs+1)
y_lr = []
for i in range(len(x_epoch)):
    if i<10:
        y_lr.append (i*lr_k+lr_b)
    else:
        y_lr.append (y_lr[i-1]*math.exp(lr_e))
plt.figure(figsize=(4,3),dpi=400)
plt.plot(x_epoch,y_lr)
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.show()

callback_list = [
    keras.callbacks.LearningRateScheduler(
        schedule,verbose=0
        )
    ]

# In[train]
history = model.fit([xt_plane, xt_seq, xt_point], yt, batch_size=16, epochs=epochs, validation_split=0.2, verbose=1, callbacks=callback_list )
loss = history.history

model_name = 'model_sample.tf'
# model.save(model_name)

# In[plot the training loss]
plt.figure(figsize=(5,3),dpi=400)
plt.plot(loss['loss'],label='loss') 
plt.plot(loss['val_loss'],label='val_loss')
plt.legend(loc=1)
plt.show()

# In[optional, load the pretrain model]
# pretrain_model = 'E:\\Multi-modal framework for SOH estimation\\dual_RN_v_8.tf'
# model = keras.models.load_model(pretrain_model)

# In[validation]
ys_est = model.predict([xs_plane, xs_seq, xs_point],verbose=1,batch_size=1)

test_mae = metrics.mean_absolute_error(ys,ys_est)
test_mape = metrics.mean_absolute_percentage_error(ys,ys_est)
test_rmse = (metrics.mean_squared_error(ys,ys_est))**0.5
test_max = np.max(abs(ys-ys_est))

mape_base = test_mape
print(test_mae,test_mape*100,test_rmse,test_max)

plt.figure(figsize=(5,3),dpi=400)
plt.plot(ys,'o') 
plt.plot(ys_est,'o')
plt.show()


