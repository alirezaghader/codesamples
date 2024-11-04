from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import activations
from keras import initializers
from keras.layers import RNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional,ConvLSTM2D,Layer
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping,History
from sklearn.utils import check_array
from keras.optimizers import Adam
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy import savetxt


class LSTMCelll(Layer):
    def __init__(self, units,
                 unit1=22,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 attention1=True,
                 **kwargs):
        super(LSTMCelll, self).__init__(**kwargs)
        self.units = units
        self.unit1=unit1
        self.attention1=attention1
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.state_size = (self.units, self.units)
    def build(self, input_shape,):
        input_dim = input_shape[-1]
        self.kernel=self.add_weight(shape=[input_dim, self.units*4 ],
                                      name='kernel',
                                      initializer=self.kernel_initializer)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer)
        self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,)
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]
        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None


        if self.attention1:
            self.Wq_k = self.add_weight(shape=(self.units,self.unit1 * 2,),
                                        name='Wq_k',
                                        initializer=self.kernel_initializer,)
            self.Wq = self.Wq_k[:, :self.unit1]
            self.Wk = self.Wq_k[:, self.unit1:]
            #bias had did not prvide a good performance
            '''if self.use_bias:
                self.bq_k = self.add_weight(shape=[self.unit1*2],
                                        name='bq_k',
                                        initializer=self.bias_initializer)
                self.bq = self.bq_k[ :self.unit1]
                self.bk = self.bq_k[ self.unit1:]'''
        self.built = True
    def attention (self,x):
        if self.attention1:
            q =self.recurrent_activation(K.dot(x, self.Wq))
            k =self.activation(K.dot(x,self.Wk))
            q_k = self.recurrent_activation(K.dot(q,tf.transpose(k)))
            q_k=(K.softmax(q_k, axis=-1))
            x=(x[:,None]*tf.transpose(q_k)[:,:,None])
            return tf.reduce_sum(x, axis=0,keepdims=False)

    def call(self, inputs, states,trainable=True):

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        x_i = K.dot(inputs, self.kernel_i)
        x_f = K.dot(inputs, self.kernel_f)
        x_c = K.dot(inputs, self.kernel_c)
        x_o = K.dot(inputs, self.kernel_o)
        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias_i)
            x_f = K.bias_add(x_f, self.bias_f)
            x_c = K.bias_add(x_c, self.bias_c)
            x_o = K.bias_add(x_o, self.bias_o)

        d=self.activation(x_c + K.dot(h_tm1 , self.recurrent_kernel_c))
        if self.attention1:
            d=self.attention(d)
        c = self.recurrent_activation(x_f + K.dot(h_tm1 , self.recurrent_kernel_f)) *(c_tm1) + \
            self.recurrent_activation(x_i + K.dot(h_tm1 , self.recurrent_kernel_i)) * d
        #o = self.recurrent_activation(x_o + K.dot(h_tm1 , self.recurrent_kernel_o))
        h = self.recurrent_activation(x_o + K.dot(h_tm1 , self.recurrent_kernel_o)) * (self.activation(c))
        return h, [h, c]

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.units)
    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'attention1':self.attention1,
                  'unit1':self.unit1,}
        base_config = super(LSTMCelll, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

warnings.filterwarnings('ignore')



model = Sequential()

#unit1= usually between 18 to 34
model.add(RNN(LSTMCelll(units=128,unit1=32,attention1=True),input_shape=(lookback, 2)))
model.add(Dense(1))
adam = Adam(lr=.01, decay=.00001)
model.compile(optimizer=adam, loss='mse',metrics=['mae', 'mape'])
checkpointer = ModelCheckpoint(filepath='lstm_first day.h5', verbose=0, save_best_only=True)
batchsize=256
history = History()

#dar epoch bala javabe khub midahad. hamchenin daraye yek jahesh dar val_loss hast va bad az jahesh khata mojadad kahesh miyabad.
blackbox = model.fit(X_train, Y_train,
                    epochs=600,
                    batch_size=batchsize,
                    validation_data=(X_valid,Y_valid),
                    callbacks=[EarlyStopping(monitor='val_loss',patience=70,
                    min_delta=10**(-8)),checkpointer,history], verbose=1, shuffle=False)

print(blackbox.history['val_loss'][-1],blackbox.history['loss'][-1])
model2 = load_model('boulder_lstm_first day.h5',custom_objects={'LSTMCelll':LSTMCelll})
scorep=model2.evaluate(X_valid,Y_valid,batch_size=batchsize)
print(scorep)

#bach_size baraye predict ba batchsize model.fit bayesti barabar bashad.
Y_predict=model2.predict(X_test,batch_size=batchsize)