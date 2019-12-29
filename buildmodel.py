from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Flatten, UpSampling2D, BatchNormalization
from keras.layers.core import Lambda
from keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.metrics import categorical_accuracy
from keras import optimizers
from HTBloss_core import HTB_Loss
import keras.backend as K
import keras
import tensorflow as tf

def GCN(c, k, ip):
    G_L1 = Conv2D(c, (k, 1), padding='same')(ip)
    G_L2 = Conv2D(c, (1, k), padding='same')(G_L1)
    G_R1 = Conv2D(c, (1, k), padding='same')(ip)
    G_R2 = Conv2D(c, (k, 1), padding='same')(G_R1)
    return Add()([G_L2, G_R2])

def BR(c, k, ip):
    B_L1 = Conv2D(c, (k, 1), padding='same')(ip)
    B_L2 = BatchNormalization()(B_L1)
    B_L2 = LeakyReLU(alpha=0.15)(B_L2)
    B_L3 = Conv2D(c, (k, 1), padding='same')(B_L2)
    return Add()([B_L3, ip])

def Conv_block(inp, f1, f2):
    # Conv_block acts as Down Sampler
    x = Conv2D(f1, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Conv2D(f2, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x


# Input
ip = Input(shape=(294, 820, 3), name="X")

# A. Down Sampler
# Conv Block1
x1 = Conv_block(ip, 4, 10)

# Conv Layer2
x2 = Conv_block(x1, 8, 15)
# Conv Layer3
x3 = Conv_block(x2, 16, 20)
# Conv Layer4
x4 = Conv_block(x3, 32, 30)

# Conv Layer5
x5 = Conv2D(40, (3, 3), padding='same')(x4)
x5 = BatchNormalization()(x5)
x5 = LeakyReLU(alpha=0.3)(x5)
x5 = Conv2D(50, (3, 3), padding='same', activation='relu')(x5)

# B. Move Rightward
# 18 51 30
x5 = GCN(30, 3, x5)
x5 = BR(30, 3, x5)

# Deconv5
x5 = BatchNormalization()(x5)
# 18 51 30
x5 = LeakyReLU(alpha=0.3)(x5)
# print(x5)

# Add 4 and 5
x5 = Add()([x5, x4])
# 18 51 25
x5 = GCN(25, 3, x5)
x5 = BR(25, 3, x5)

# print(x5)
# Deconv4 -- UP Sampler
# BEFORE Transpose 18 21
x5 = Conv2DTranspose(20, (2, 2), strides=(2, 2), padding='valid')(x5)
# AFTER Transpose 37 42
x5 = BatchNormalization()(x5)
x5 = LeakyReLU(alpha=0.3)(x5)
# print(x5)

# Add 3 and 4
# x5: 37 102 20
# x3: 36 102 20
x5 = Add()([x5, x3])
x5 = GCN(25, 3, x5)
x5 = BR(25, 3, x5)

# Deconv3
x5 = Conv2DTranspose(15, (3, 3), strides=(2, 2), padding='valid')(x5)
x5 = BatchNormalization()(x5)
x5 = LeakyReLU(alpha=0.3)(x5)

# Add 3 and 4
# 72 204 15 + 73 205 15
x5 = Add()([x5, x2])
x5 = GCN(20, 3, x5)
x5 = BR(20, 3, x5)

# Deconv2
x5 = Conv2DTranspose(10, (3, 2), strides=(2, 2), padding='valid')(x5)
x5 = BatchNormalization()(x5)
x5 = LeakyReLU(alpha=0.3)(x5)

# Add 3 and 4
x5 = Add()([x5, x1])
x5 = GCN(15, 3, x5)
x5 = BR(15, 3, x5)

x5 = Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='valid')(x5)
x5 = BatchNormalization()(x5)
x5 = LeakyReLU(alpha=0.3)(x5)
x5 = GCN(3, 3, x5)
x5 = BR(3, 3, x5)


# Custom Metric
def HTBloss_metric(y_true, y_pred):
    loss = tf.numpy_function(HTB_Loss.calculate, [y_pred, y_true, lane_status],
                          tf.float32, name='HTBloss_metric')
    return loss

# Model Compile
# Uncomment below to support HTBloss Metric (Which do not support GPU)
# lane_status = Input(shape=(4, 1), name="LaneStatus")
# model = Model(inputs=[ip, lane_status], outputs=[x5])

# GPU-Support
model = Model(inputs=[ip], outputs=[x5])

model.summary()
Ndm = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

# model.compile(optimizer='Adam', loss=keras.losses.binary_crossentropy, metrics=[HTBloss_metric])
model.compile(optimizer='Adam', loss=keras.losses.binary_crossentropy)

json_string = model.to_json()
with open('GCN_model.json', 'w') as of:
    of.write(json_string)
