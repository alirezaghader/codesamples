import numpy as np
from sklearn.metrics import r2_score,mean_absolute_percentage_error,mean_absolute_error,mean_squared_error
import os
from  builtins import any as b_any
import time
import h5py
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import initializers, activations

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
import tensorflow.keras as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Dense, Dropout, Activation, Layer,Flatten, add, Add, multiply,GlobalAveragePooling2D, GlobalMaxPooling2D,\
    Reshape,Permute, Concatenate, Lambda,BatchNormalization,Dot,dot
from tensorflow.keras.models import load_model

florida_list_max=[11435, 15018 ,15983 ,19724 ,17601 ,12206,10038,9990,8223,6903,9284,9363]
#minesota_list_max=[11688, 12307, 11987, 12708, 11536, 10856,9164,8839,6533,5479,7191,8768]
#california_list_max=[10840 ,13407, 17985 ,16306, 14182, 12581,10334,8159, 7170 ,7257, 9106, 8498 ]
# farvardin california eslah konam
#hf_1 = h5py.File(f'E:/minnesota/data_minnesota_{name}_normalized_new_total.h5', 'r')
month=['farvardin','ordibehesht','khordad','tir','mordad','shahrivar']
name=month[0]
maximum=florida_list_max[0]
hf_1 = h5py.File(f'E:/florida/total_data_{name}_normalized_gldas_excluded_last_.h5', 'r')
Data = hf_1['evapotranspiration']
Data = (np.array(Data))
print(Data.shape)
#Data=np.concatenate((Data[:,:,:,:,:,:29],Data[:,:,:,:,:,30:35],Data[:,:,:,:,:,36:]),axis=-1)

#Data=np.concatenate((Data[:,:,:,:,:,:12],Data[:,:,:,:,:,15:]),axis=-1)
#Data[:,:,:,:,:,18:23],
#Data=np.concatenate((Data[:,:,:,:,:,2:14],np.expand_dims((Data[:,:,:,:,:,-1]),axis=-1) ),axis=-1)
#Data=np.concatenate((Data[:,:,:,:,:,:2],Data[:,:,:,:,:,14:18],Data[:,:,:,:,:,23:]),axis=-1)
#Data=np.concatenate((np.expand_dims((Data[:,:,:,:,:,14]),axis=-1),np.expand_dims((Data[:,:,:,:,:,23]),axis=-1),Data[:,:,:,:,:,18:23],np.expand_dims((Data[:,:,:,:,:,-1]),axis=-1)),axis=-1)

#print(Data.shape)

#Data=np.random.rand(16,4,100,32,32,41)

"""hf = h5py.File('D:/h5/data_florida_Prop.h5', 'r')

#hf = h5py.File('D:/h5/minnesota_type3.h5', 'r')
Data_20189 = hf['evapotranspiration']
Data_20189 = (np.array(Data_20189))
Data_20189=Data_20189[1:,2,:,:,:,:]

print(Data_20189.shape)"""
X_train = Data[:12, :, :, :, :, :-1]
Y_train = Data[:12, :, :, :, :, -1]
X_valid = Data[12:14, :, :, :, :, :-1]
Y_valid = Data[12:14, :, :, :, :, -1]
X_test = Data[14:, :, :, :, :, :-1]
Y_test = Data[14:, :, :, :, :, -1]


Y_train = np.reshape((Y_train),
                     (Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], Y_train.shape[3], Y_train.shape[4], 1))
Y_valid = np.reshape((Y_valid),
                     (Y_valid.shape[0], Y_valid.shape[1], Y_valid.shape[2], Y_valid.shape[3], Y_valid.shape[4], 1))
Y_test=np.reshape((Y_test),(Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],Y_test.shape[3],Y_test.shape[4],1))

X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1] * X_train.shape[2], X_train.shape[-3], X_train.shape[-2],
                          X_train.shape[-1])
X_valid = X_valid.reshape(X_valid.shape[0] * X_valid.shape[1] * X_valid.shape[2], X_valid.shape[-3], X_valid.shape[-2],
                          X_valid.shape[-1])
X_test = X_test.reshape(X_test.shape[0] * X_test.shape[1] * X_test.shape[2], X_test.shape[-3], X_test.shape[-2],
                        X_test.shape[-1])

'''fpar-lai-7-3-2-lst-3-5-19-1'''

#Y_test=np.reshape((Y_test),(Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],Y_test.shape[3],Y_test.shape[4],1))

"""X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1]*X_train.shape[2], X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
X_valid = X_valid.reshape(X_valid.shape[0]*X_valid.shape[1]*X_valid.shape[2], X_valid.shape[-3], X_valid.shape[-2], X_valid.shape[-1])
X_test = X_test.reshape(X_test.shape[0]*X_test.shape[1]*X_test.shape[2], X_test.shape[-3], X_test.shape[-2], X_test.shape[-1])"""

Y_test_mse_N, Y_test_mae_N,Y_test_mape_N, Y_valid_mse_N, Y_valid_mae_N,Y_valid_mape_N, Y_train_mse_N, Y_train_mae_N,Y_train_mape_N = [], [], [], [], [], [],[], [], [],
valllosss,combination=[],[]

lb=[1,1,1,.2,.00001,1,1,1]
ul=[4.99,4.99,2.99,.5,.005,3.99,4.99,4.99]

class Self_attention(Layer):
    def __init__(self,activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(Self_attention, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
    def build(self, input_shape,):
        #optimize second variable input_shape[-3]
        self.Wq = self.add_weight(shape=(input_shape[-1],input_shape[-2],input_shape[-3],),
                                        name='Wq',
                                        initializer=self.kernel_initializer,)
        self.Wk = self.add_weight(shape=(input_shape[-1], input_shape[-2], input_shape[-3],),
                                        name='Wk',
                                        initializer=self.kernel_initializer, )
        self.built = True

    def call(self, input,trainable=True):
        input = (tf.transpose(input, perm=[0,3,1, 2]))
        print(input.shape,'shapex')
        q = self.recurrent_activation(tf.matmul(input, self.Wq))
        K=self.activation(tf.matmul(input, self.Wk))
        q_k = self.recurrent_activation(tf.matmul(q, tf.transpose(K, perm=[0,1, 3, 2])))
        q_k = (k.softmax(q_k, axis=-1))
        x=tf.matmul(input, q_k)
        x = (tf.transpose(x, perm=[0,2, 3, 1]))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
              }
        base_config = super(Self_attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class calculat_K(Layer):
    def __init__(self, activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(calculat_K, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

    def build(self, input_shape, ):
        self.Wk = self.add_weight(shape=(input_shape[-1], input_shape[-2], input_shape[-3],),
                                  name='Wk',
                                  initializer=self.kernel_initializer, )

        self.built = True

    def call(self, x, trainable=True):
        x = (tf.transpose(x, perm=[0, 3, 1, 2]))
        K=self.activation(tf.matmul(x, self.Wk))
        K=tf.transpose(K, perm=[0,1, 3, 2])
        return K

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
        }
        base_config = super(calculat_K, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class calculate_Q(Layer):
    def __init__(self, activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(calculate_Q, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)

    def build(self, input_shape, ):
        self.Wq = self.add_weight(shape=(input_shape[-1], input_shape[-2], input_shape[-3],),
                                  name='Wq',
                                  initializer=self.kernel_initializer, )

        self.built = True

    def call(self, x, trainable=True):
        x = (tf.transpose(x, perm=[0, 3, 1, 2]))
        q = self.recurrent_activation(tf.matmul(x, self.Wq))
        #print(x.shape, 'shapex')
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3], input_shape[1], input_shape[2])

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'activation': activations.serialize(self.activation),
            'recurrent_activation': activations.serialize(self.recurrent_activation),
        }
        base_config = super(calculate_Q, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def self_attention(q, k,v ):
    v = Permute((3, 1, 2))(v)
    Q_K=tf.matmul(q,k)
    Q_K=Activation('sigmoid')(Q_K)
    Q_K=tf.keras.activations.softmax(Q_K,axis=-1)
    #badesh ya ghablesh? ghablesh
    #mean ba halate ghabl?
    #badesh ba yek activation?

    """Q_K=tf.reduce_mean(Q_K, axis=1,keepdims=True)
    Q_K = Activation('tanh')(Q_K)"""

    x = tf.matmul(v, Q_K)
    #x=K.layers.concatenate([x1, x2], axis=-3,)
    #print(x.shape,'x')
    #x=Permute((2,3,1))(x)
    # Transposed convolution parameters
    #x=K.layers.Conv2D(filters=x.shape[-1]/2,**params)(x)
    x = Permute((2, 3, 1))(x)
    return x


def fitness(X):
    print(X)
    combination.append([X])
    Y_valid = Data[12:14, :, :, :, :, -1]
    Y_valid = np.reshape((Y_valid),
                         (Y_valid.shape[0], Y_valid.shape[1], Y_valid.shape[2], Y_valid.shape[3], Y_valid.shape[4], 1))
    Y_valid = Y_valid.reshape(Y_valid.shape[0] * Y_valid.shape[1] * Y_valid.shape[2], Y_valid.shape[-3],
                              Y_valid.shape[-2], Y_valid.shape[-1])

    Y_train = Data[:12, :, :, :, :, -1]
    Y_train = np.reshape((Y_train),
                         (Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], Y_train.shape[3], Y_train.shape[4], 1))
    Y_train = Y_train.reshape(Y_train.shape[0] * Y_train.shape[1] * Y_train.shape[2], Y_train.shape[-3],
                              Y_train.shape[-2], Y_train.shape[-1])

    Y_test = Data[14:, :, :, :, :, -1]
    """Y_test = np.expand_dims(Y_test, axis=0)
    Y_test = np.expand_dims(Y_test, axis=0)"""
    Y_test = np.reshape((Y_test),
                        (Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], Y_test.shape[3], Y_test.shape[4], 1))
    Y_test = Y_test.reshape(Y_test.shape[0] * Y_test.shape[1] * Y_test.shape[2], Y_test.shape[-3], Y_test.shape[-2],
                            Y_test.shape[-1])
    """Y_test = np.reshape((Y_test),
                        (1,Y_test.shape[0], Y_test.shape[1], Y_test.shape[2], Y_test.shape[3], 1))
    Y_test = Y_test.reshape(Y_test.shape[0] * Y_test.shape[1] * Y_test.shape[2], Y_test.shape[-3], Y_test.shape[-2],
                            Y_test.shape[-1])"""
    input_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    num_chan_out = 1
    dropout=round(X[3],3)
    param_kernal=int(X[2])*2+1
    fms=int(X[0])*16
    print(fms,'fms')
    fms2=int(X[1])*16
    batch_size=128
    learning_rate=X[4]
    value=int(X[5])
    #ratio=int(X[6])
    #ratio2 = int(X[7])

    adam = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam", )
    inputs = K.layers.Input(input_shape, name="MRImages")

    # Convolution parameters
    params = dict(kernel_size=(param_kernal, param_kernal), activation="tanh",
              padding="same",)

# Transposed convolution parameters
    params_trans = dict(kernel_size=(2, 2), strides=(2, 2),
                    padding="same")
#pre
    encodeAAa = K.layers.Conv2D(name="encodeAAa", filters=fms*2, **params)(inputs)
    encodeAAb = K.layers.Conv2D(name="encodeAAb", filters=fms2, **params)(encodeAAa)
    #encodeAA=Self_attention()(encodeAA)
    #encodeAA=K.layers.BatchNormalization()(encodeAA)

    #A #32*32*16
    encodeAa = K.layers.Conv2D(name="encodeAa", filters=fms, **params)(encodeAAb)
    encodeAb = K.layers.Conv2D(name="encodeAb", filters=fms, **params)(encodeAa)
    encodeAb = K.layers.SpatialDropout2D(dropout)(encodeAb)

    # 16*16*16
    poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeAb)
    #B 16*16*32
    encodeBa = K.layers.Conv2D(name="encodeBa", filters=fms*2, **params)(poolA)
    encodeBb = K.layers.Conv2D(name="encodeBb", filters=fms*2, **params)(encodeBa)
    encodeBb = BatchNormalization() (encodeBb)
    encodeBb = Activation(activation='tanh') (encodeBb)

    # 8*8*32
    poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeBb)
    #C 8*8*64
    encodeCC = K.layers.Conv2D(name="encodeCa", filters=fms*4, **params)(poolB)
    encodeCb = K.layers.Conv2D(name="encodeCb", filters=fms*4, **params)(encodeCC)
    encodeCb = K.layers.SpatialDropout2D(dropout)(encodeCb)

    if value == 2:
        # bottle_neck
        # 4*4*64
        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeCb)
        # 4*4*128
        encodeDa = K.layers.Conv2D(name="encodeDa", filters=fms * 8, **params)(poolC)
        encodeDb = K.layers.Conv2D(name="encodeDb", filters=fms * 8, **params)(encodeDa)

        # 8*8*64
        upa = K.layers.Conv2DTranspose(name="transconvE", filters=fms * 4, **params_trans)(encodeDb)
        # encodeCbb = K.layers.Conv2D(name="encodeCbb", filters=fms * 8, **params)(encodeCb)
        K_encodeCb = calculat_K()(encodeCb)
        Q_upa = calculate_Q()(upa)
        layera = self_attention(k=K_encodeCb, q=Q_upa, v=upa)
        #layera = channel_attention(layera)
        concatE = K.layers.concatenate([encodeCb,layera ], axis=-1, name="concatE")
        #concatE = channel_attention(concatE,ratio=ratio*2)

        # 8*8*256
        decodeCa = K.layers.Conv2D(name="decodeCa", filters=fms * 4, **params)(concatE)
        decodefinal = K.layers.Conv2D(name="decodeCb", filters=fms * 4, **params)(decodeCa)
        # 16*16*256
        upb = K.layers.Conv2DTranspose(name="transconvC", filters=fms * 2, **params_trans)(decodefinal)
    elif value == 1:
        # C 4*4*64
        poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeCb)
        # C 4*4*128
        encodeDa = K.layers.Conv2D(name="encodeDa", filters=fms * 8, **params)(poolC)
        encodeDb = K.layers.Conv2D(name="encodeDb", filters=fms * 8, **params)(encodeDa)
        # C 2*2*128
        poolD = K.layers.MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeDb)
        # C 2*2*256
        encodeEa = K.layers.Conv2D(name="encodeEa", filters=fms * 16, **params)(poolD)
        encodeEb = K.layers.Conv2D(name="encodeEb", filters=fms * 16, **params)(encodeEa)
        encodeEb = BatchNormalization()(encodeEb)
        encodeEb = Activation(activation='tanh')(encodeEb)
        encodeEb = K.layers.SpatialDropout2D(dropout)(encodeEb)
        # C 4*4*128
        upaa = K.layers.Conv2DTranspose(name="transconvE", filters=fms * 8, **params_trans)(encodeEb)
        # encodeDcc = K.layers.Conv2D(name="encodeDcc", filters=fms * 8, **params)(encodeDb)
        K_encodeDc = calculat_K()(encodeDb)
        Q_upaa = calculate_Q()(upaa)
        layeraa = self_attention(k=K_encodeDc, q=Q_upaa, v=upaa)
        #layeraa = channel_attention(layeraa)

        #channel_layer= tf.transpose( tf.matmul(tf.transpose(layeraa, perm=[0,3,1, 2]),tf.transpose(channel, perm=[0,3,1, 2])),perm=[0,2, 3, 1])

        concatE = K.layers.concatenate([layeraa, encodeDb ,], axis=-1, name="concatE")
        #concatE = channel_attention(concatE,ratio=ratio)

        # C 4*4*128
        decodeDa = K.layers.Conv2D(name="decodeDa", filters=fms * 8, **params)(concatE)
        decodeDa = K.layers.Conv2D(name="decodeDb", filters=fms * 8, **params)(decodeDa)

        # C 8*8*64
        upaaa = K.layers.Conv2DTranspose(name="transconvEE", filters=fms * 4, **params_trans)(decodeDa)
        # decodeCbb = K.layers.Conv2D(name="decodeCbb", filters=fms * 4, **params)(encodeCb)
        K_encodeCb = calculat_K()(encodeCb)
        Q_upaaa = calculate_Q()(upaaa)
        layeraaa = self_attention(k=K_encodeCb, q=Q_upaaa, v=upaaa)
        #layeraaa = channel_attention(layeraaa)

        concatF = K.layers.concatenate([ encodeCb,layeraaa], axis=-1, name="concatF")
        #concatF = channel_attention(concatF,ratio=ratio*2)

        # C 8*8*64
        decodeCa = K.layers.Conv2D(name="decodeCaa", filters=fms * 4, **params)(concatF)
        decodefinal = K.layers.Conv2D(name="decodeCbbb", filters=fms * 4, **params)(decodeCa)
        # C 16*16*32
        upb = K.layers.Conv2DTranspose(name="transconvC", filters=fms * 2, **params_trans)(decodefinal)
    else:
        upb = K.layers.Conv2DTranspose(name="transconvC", filters=fms * 2, **params_trans)(encodeCb)
    # C 16*16*32
    # encodeBbb = K.layers.Conv2D(name="encodeBbb", filters=fms * 2, **params)(encodeBb)
    K_encodeBb = calculat_K()(encodeBb)
    Q_upb = calculate_Q()(upb)
    layerb = self_attention(k=K_encodeBb, q=Q_upb, v=upb, )
    #layerb = channel_attention(layerb)

    concatC = K.layers.concatenate([encodeBb,layerb,], axis=-1, name="concatC")
    #concatC = channel_attention(concatC,ratio=ratio2)

    # 16*16*128
    decodeBa = K.layers.Conv2D(name="decodeBa", filters=fms * 2, **params)(concatC)
    decodeBb = K.layers.Conv2D(name="decodeBb", filters=fms * 2, **params)(decodeBa)
    decodeBb = BatchNormalization()(decodeBb)
    decodeBb = Activation(activation='tanh')(decodeBb)
    decodeBb = K.layers.SpatialDropout2D(dropout)(decodeBb)

    # 32*32*16
    upc = K.layers.Conv2DTranspose(name="transconvB", filters=fms * 1, **params_trans)(decodeBb)
    # encodeAbb = K.layers.Conv2D(name="encodeAbb", filters=fms*1, **params)(encodeAb)
    K_encodeAb = calculat_K()(encodeAb)
    Q_upc = calculate_Q()(upc)
    layerbb = self_attention(k=K_encodeAb, q=Q_upc, v=upc, )
    #layerbb = channel_attention(layerbb)

    concatD = K.layers.concatenate([ upc,layerbb ], axis=-1, name="concatD")
    #concatD = channel_attention(concatD,ratio=ratio2*2)
    # 32*32*16
    convOuta = K.layers.Conv2D(name="convOuta", filters=fms, **params)(concatD)

    convOuta = K.layers.SpatialDropout2D(dropout)(convOuta)
    convOutb = K.layers.Conv2D(name="convOutb", filters=int(fms/4), **params)(convOuta)
    convOutc = K.layers.Conv2D(name="convOutd", filters=int(fms/8), **params)(convOutb)

    prediction = K.layers.Conv2D(name="PredictionMask",
                                 filters=num_chan_out, kernel_size=(1, 1),
                                 activation="linear")(convOutc)


    model = K.models.Model(inputs=[inputs], outputs=[prediction], name="2DUNet_Brats_Decathlon")
    model.compile(loss='mean_absolute_error', # one may use 'mean_absolute_error' as  mean_squared_error
                      optimizer=adam,
                      metrics=[tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.MeanAbsolutePercentageError()])

    #model.summary()
    """model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'E:/florida/saved models/saved_model_{name}_california_self_index_noise.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,)
    #baseline=0.08
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=200, verbose=2,
              callbacks=[model_checkpoint_callback,tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25,baseline=0.06,restore_best_weights=True)],validation_data=(X_valid,Y_valid))
"""

    saved_model = load_model(f'D:/h5/saved california/self/saved_model_{name}_california_self_index_dem.h5',
                             custom_objects={'calculat_K':calculat_K,'calculate_Q':calculate_Q,'self_attention':self_attention,})

    saved_model.compile(loss='mean_absolute_error', # one may use 'mean_absolute_error' as  mean_squared_error
                      optimizer=adam,
                      metrics=[ tf.keras.metrics.MeanSquaredError()])

    """score_test = saved_model.evaluate(X_test, Y_test, verbose=0,batch_size=batch_size)
    print(score_test,'= score_test')
    Y_test_mse_N.append(score_test[1])
    Y_test_mae_N.append(score_test[0])
    Y_test_mape_N.append(score_test[-1])
    score_train = saved_model.evaluate(X_train, Y_train, verbose=0,batch_size=batch_size)
    print(score_train, '= score_train')
    Y_train_mse_N.append(score_train[1])
    Y_train_mae_N.append(score_train[0])
    Y_train_mape_N.append(score_train[-1])
    score_valid = saved_model.evaluate(X_valid, Y_valid, verbose=0,batch_size=batch_size)
    print(score_valid, '= score_valid')
    Y_valid_mse_N.append(score_valid[1])
    Y_valid_mae_N.append(score_valid[0])
    Y_valid_mape_N.append(score_valid[-1])
    valloss = score_valid[0]
    valllosss.append(valloss)"""
    """list_valid = []
    preds_val = saved_model.predict(X_valid, batch_size=128)
    # print(preds.shape, Y_test.shape)
    Y_valid = np.reshape((Y_valid), (Y_valid.shape[0], Y_valid.shape[1] * Y_valid.shape[2]))
    preds_val = np.reshape((preds_val), (preds_val.shape[0], preds_val.shape[1] * preds_val.shape[2]))
    preds_val = preds_val * 8223
    Y_valid = Y_valid * 8223
    for i in range(Y_valid.shape[0]):
        for j in range(Y_valid.shape[1]):
            if Y_valid[i, j] != 0 and preds_val[i, j] != 0:
                list_valid.append((np.absolute((Y_valid[i, j] - preds_val[i, j]) / Y_valid[i, j])).item())
    print(sum(list_valid) / len(list_valid))

    list_train = []
    preds_train = saved_model.predict(X_train, batch_size=128)
    # print(preds.shape, Y_test.shape)
    Y_train = np.reshape((Y_train), (Y_train.shape[0], Y_train.shape[1] * Y_train.shape[2]))
    preds_train = np.reshape((preds_train), (preds_train.shape[0], preds_train.shape[1] * preds_train.shape[2]))
    preds_train = preds_train * 15018
    Y_train = Y_train * 15018
    for i in range(Y_train.shape[0]):
        for j in range(Y_train.shape[1]):
            if Y_train[i, j] != 0 and preds_train[i, j] != 0:
                list_train.append((np.absolute((Y_train[i, j] - preds_train[i, j]) / Y_train[i, j])).item())
    print(sum(list_train) / len(list_train))"""

    preds = saved_model.predict(X_test, batch_size=128)
    print(preds.shape, Y_test.shape)
    Y_test = np.reshape((Y_test), (Y_test.shape[0], Y_test.shape[1] * Y_test.shape[2]))
    preds = np.reshape((preds), (preds.shape[0], preds.shape[1] * preds.shape[2]))
    preds = preds * maximum
    Y_test = Y_test * maximum
    """with h5py.File(f'E:/florida/saved data/Y_test_california_{name}_self_index_noise.h5', 'w') as hf:
        hf.create_dataset("Y_test",  data=Y_test,dtype='float32',compression='gzip')
    with h5py.File(f'E:/florida/saved data/preds_california_{name}_self_index_noise.h5', 'w') as hf:
        hf.create_dataset("preds", data=preds, dtype='float32', compression='gzip')"""
    # 11435 15018 15983 19724 17601 12206
    # 9990

    list = []
    for i in range(Y_test.shape[0]):
        for j in range(Y_test.shape[1]):
            if Y_test[i, j] != 0 and preds[i, j] != 0:
                list.append((np.absolute((Y_test[i, j] - preds[i, j]) / Y_test[i, j])).item())
    print(sum(list) / len(list))

    print(r2_score(Y_test, preds),
          mean_absolute_error(Y_test, preds)/1000, ((mean_squared_error(Y_test, preds)) ** .5)/1000,
          mean_absolute_percentage_error(Y_test, preds))
    print(np.average(Y_test), np.max(Y_test), np.min(Y_test))
    """for i in range(11):
        print(i+1)
        list = []
        c = np.where(Data_20189 == i)
        #print(c, )
        Y_testt = np.take(Y_test, c)

        # Y_testt = np.reshape(Y_testt, Y_testt.shape[1] * 2)

        predss = np.take(preds, c)
        print(Y_testt.shape)
        print(predss.shape)
        # predss = np.reshape(predss, predss.shape[1] * 2)
        for i in range(Y_testt.shape[0]):
            for j in range(Y_testt.shape[1]):
                if Y_testt[i, j] != 0 and predss[i, j] != 0:
                    list.append((np.absolute((Y_testt[i, j] - predss[i, j]) / Y_testt[i, j])).item())
        print(sum(list) / len(list))


        average = np.average(Y_testt)
        r_2 = 1 - np.sum((predss - Y_testt) ** 2) / np.sum((average - Y_testt) ** 2)

        print(r_2,)
        print(mean_absolute_error(Y_testt, predss))
        print((mean_squared_error(Y_testt, predss)) ** .5)
        #print( mean_absolute_percentage_error(Y_testt, predss))
        print(np.max(Y_testt))
        print(np.min(Y_testt))
        print(np.average(Y_testt))"""

    del model
    tensorflow.keras.backend.clear_session()
    k.clear_session()
    tf.compat.v1.reset_default_graph()
    #return valloss

Leader_scores=[]
Leader_positions=[]
import numpy as np
from numpy.random import uniform, randint, choice
import random
import math

class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
import time

Leader_scores=[]
Leader_positions=[]

def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500

    # initialize the location and Energy of the rabbit
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("inf")  # change this to -inf for maximization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Initialize the locations of Harris' hawks
    X = np.asarray(
        [x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)

    ############################
    s = solution()

    print('HHO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundries

            X[i, :] = np.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = objf(X[i, :])

            # Update the location of Rabbit
            if fitness < Rabbit_Energy:  # Change this to > for maximization problem
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2 * (1 - (t / Max_iter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (
                E0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[i, :] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i, :]
                    )

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random.random()  # probablity of each event

                if (
                    r >= 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (6) in paper
                    X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :]
                    )

                if (
                    r >= 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    Jump_strength = 2 * (
                        1 - random.random()
                    )  # random jump strength of the rabbit
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                    r < 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X[i, :])
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()
                if (
                    r < 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X.mean(0)
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X.mean(0))
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()

        convergence_curve[t] = Rabbit_Energy
        if t % 1 == 0:
            print(["At iteration "+ str(t)+ " the best fitness is "+ str(Rabbit_Energy)])
            print(["At iteration " + str(t) + " the best set is " + str(Rabbit_Location)])
            Leader_scores.append(Rabbit_Energy)
            Leader_positions.append(Rabbit_Location)
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location

    return s
def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u= 0.01*np.random.randn(dim)*sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v),(1/beta))
    step = np.divide(u,zz)
    return step

for i in range(1):
    #HHO(objf=fitness, lb=lb, ub=ul, dim=8, SearchAgents_no=5, Max_iter=5)
    fitness(X=[1.00000000e+00, 1.00000000e+00 ,1.00000000e+00, 2.00000000e-01,2.08761243e-04, 1.62669585e+00])
    print('Y_test_mse_N,Y_test_mae_N ', Y_test_mse_N, Y_test_mae_N,Y_test_mape_N)
    print('Y_valid_mse_N,Y_valid_mae_N ', Y_valid_mse_N, Y_valid_mae_N,Y_valid_mape_N)
    print('Y_train_mse_N,Y_train_mae_N ', Y_train_mse_N, Y_train_mae_N,Y_train_mape_N )
    print('Leader_scores: ', Leader_scores)
    print('Leader_positions: ', Leader_positions)


    """val, idx = min((val, idx) for (idx, val) in enumerate(valllosss))
    print('val,idx ', val, idx)
    print(Y_test_mse_N[idx], Y_test_mae_N[idx], Y_train_mse_N[idx],
          Y_train_mae_N[idx], Y_valid_mse_N[idx], Y_valid_mae_N[idx], )
    print('valllosss: ', valllosss)
    with open('california_farvardin_attention_self_attention_new', "w") as text_file:
        print(combination,
              file=text_file)

    Y_test_mse_N, Y_test_mae_N = [], []
    Y_valid_mse_N, Y_valid_mae_N = [], [],
    Y_train_mse_N, Y_train_mae_N = [], []
    Leader_scores = []
    Leader_positions = []
    valllosss, combination = [], []"""
