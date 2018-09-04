# -*- coding: utf-8 -*-
"""
Created on Wed Mar  14 9:39 2018
derived class from keras.callbacks
    class LR_change_batch
    class droprate_change_batch
    class val_test
derived class layer
    AdaptedDilatedConV1D
function
    confusion matrix
    combined dilated convolution

@author: a273
"""

import random
import numpy as np
import math
import keras
from keras import backend as K
import keras.layers as KL
from keras.engine.topology import Layer
from keras.utils import conv_utils
from collections import OrderedDict

# derived class from keras.callbacks
class LR_change_batch(keras.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an batch index as input
            (integer, indexed from 0) and returns a preodic
            learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LR_change_batch, self).__init__()
        self.schedule = schedule

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.schedule(batch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        
class droprate_change_batch(keras.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an batch index as input
            (integer, indexed from 0) and returns a new
            drop_out_rate for layer 'change_drop' rate as output (float).
    """

    def on_batch_begin(self, batch, logs=None):
        drop_rate = random.random()*.8 +.1
        drop_layer = self.model.get_layer('change_drop')
        config = drop_layer.get_config()
        config['rate'] = drop_rate
        drop_layer = KL.Dropout.from_config(config)

class val_test(keras.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and predict the test data
    """
    def __init__(self, test_data, test_label):
        self.test_data = test_data
        self.test_label = test_label
        self.test_acc = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs={}):
        test_re = self.model.evaluate(self.test_data, self.test_label, verbose = 0)
        self.test_acc.append(test_re[1])
        self.test_loss.append(test_re[0])
        print('test_result',test_re)

class save_log(keras.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and predict the test data
    """
    def __init__(self):
        self.history = OrderedDict()
        self.history['acc'] = []
        self.history['loss'] = []
        self.history['test_acc'] = []
        self.history['test_loss'] = []
        self.history['val_acc'] = []
        self.history['val_loss'] = []

    def on_epoch_end(self, epoch, logs={}):
        self.history['acc'].append(logs.get('acc'))
        self.history['loss'].append(logs.get('loss'))
        self.history['val_acc'].append(logs.get('val_acc'))
        self.history['val_loss'].append(logs.get('val_loss'))
        # self.history['test_acc'] = self.test_acc
        # self.history['test_loss'] = self.test_loss
        # self.history['test_acc'].append(logs.get('test_acc'))
        # self.history['test_loss'].append(logs.get('test_loss'))
        save_dict(self.history,'temp')

# derived class layer
class reAC1D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 activation = None, 
                 ac_list = [1],
                 use_bias = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
        self.activation = KL.activations.get(activation)
        self.ac_list = ac_list
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        
        super(reAC1D, self).__init__(**kwargs)

    def build(self, input_shape):
        # print('ac_list',self.ac_list)
        # print('output',input_shape)
        if self.dim_ordering == 'th':
            channel_axis = 1
        elif self.dim_ordering == 'tf':
            channel_axis = -1
            
        input_dim = input_shape[channel_axis]
        # print('1',input_dim)
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.filters*len(self.ac_list))

    def get_config(self):
        config = {'ac_list': self.ac_list}
        base_config = super(reAC1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        input_shape = K.shape(inputs)
        outputs = []
        for i in range (len(self.ac_list)):
            # print('i',i,'input_shape',K.int_shape(inputs),'k_shape',K.int_shape(self.kernel))
            temp_outputs = K.conv1d(
                                inputs,
                                self.kernel,
                                strides= 1,
                                padding = 'same',
                                data_format = 'channels_last',
                                dilation_rate=self.ac_list[i])
            if self.use_bias:
                temp_outputs = K.bias_add(
                temp_outputs,
                self.bias,
                data_format='channels_last')
            if self.activation is not None:
                temp_outputs = self.activation(temp_outputs)
                
            outputs.append(temp_outputs)
        out = K.concatenate(outputs)
        # print('output',K.int_shape(out))
        out = K.reshape(out,[input_shape[0],input_shape[1],self.filters*len(self.ac_list)])
        return out

class reAC2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 activation = None, 
                 ac_list = [1],
                 use_bias = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.activation = KL.activations.get(activation)
        self.ac_list = ac_list
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        
        super(reAC2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # print('ac_list',self.ac_list)
        # print('output',input_shape)
        if self.dim_ordering == 'th':
            channel_axis = 1
        elif self.dim_ordering == 'tf':
            channel_axis = -1
            
        input_dim = input_shape[channel_axis]
        # print('1',input_dim)
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.filters*len(self.ac_list))

    def get_config(self):
        config = {'ac_list': self.ac_list}
        base_config = super(reAC2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        input_shape = K.shape(inputs)
        outputs = []
        for i in range (len(self.ac_list)):
            # print('i',i,'input_shape',K.int_shape(inputs),'k_shape',K.int_shape(self.kernel))
            if isinstance(self.ac_list[i], int):
                temp_dilation_rate = list(self.ac_list[i])
            elif isinstance(self.ac_list[i], tuple):
                temp_dilation_rate = self.ac_list[i]
            elif isinstance(self.ac_list[i], list):
                temp_dilation_rate = self.ac_list[i]
            temp_outputs = K.conv2d(
                                inputs,
                                self.kernel,
                                strides= (1,1),
                                padding = 'same',
                                data_format = 'channels_last',
                                dilation_rate=temp_dilation_rate)
            if self.use_bias:
                temp_outputs = K.bias_add(
                temp_outputs,
                self.bias,
                data_format='channels_last')
            if self.activation is not None:
                temp_outputs = self.activation(temp_outputs)
                
            outputs.append(temp_outputs)
        out = K.concatenate(outputs,axis=3)
        # print('output',K.int_shape(out))
        out = K.reshape(out,[input_shape[0],input_shape[1],input_shape[2],self.filters*len(self.ac_list)])
        return out

class shuffle_pool(Layer):
    def __init__(self, shuffle_list, **kwargs):
        self.shuffle_list = shuffle_list
        super(shuffle_pool, self).__init__(**kwargs)

    def call(self, x):
        tensor_shape = K.int_shape(x)
        input_shape = K.shape(x)
        shuffle_axis = len(tensor_shape) - 2
        outputs = []
        for i in self.shuffle_list:
            if shuffle_axis == 2:
                temp_x = K.reshape(x,[input_shape[0],input_shape[1],int(tensor_shape[2]/i),int(i)])
                decom_x_1 = temp_x[:,:,0:K.int_shape(temp_x)[2]:2,:]
                decom_x_1 = K.reshape(decom_x_1,[input_shape[0],input_shape[1],int(tensor_shape[2]/2),1])
                decom_x_2 = temp_x[:,:,1:K.int_shape(temp_x)[2]:2,:]
                decom_x_2 = K.reshape(decom_x_2,[input_shape[0],input_shape[1],int(tensor_shape[2]/2),1])
                decom_x = K.concatenate((decom_x_1,decom_x_2),axis=3)
            elif shuffle_axis == 1:
                temp_x = K.reshape(x,[input_shape[0],int(tensor_shape[1]/i),int(i)])
                decom_x_1 = temp_x[:,0:K.int_shape(temp_x)[1]:2,:]
                decom_x_1 = K.reshape(decom_x_1,[input_shape[0],int(tensor_shape[1]/2),1])
                decom_x_2 = temp_x[:,1:K.int_shape(temp_x)[1]:2,:]
                decom_x_2 = K.reshape(decom_x_2,[input_shape[0],int(tensor_shape[1]/2),1])
                decom_x = K.concatenate((decom_x_1,decom_x_2),axis=2)
            outputs.append(decom_x)
        out = K.concatenate(outputs,axis=shuffle_axis+1)
        out = K.reshape(out,[input_shape[0],input_shape[1],int(tensor_shape[2]/2),2*len(self.shuffle_list)])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],int(input_shape[2]/2),2*len(self.shuffle_list))

class gather_loss(Layer):
    def __init__(self, loss_weight=0, **kwargs):
        self.loss_weight = loss_weight
        super(gather_loss, self).__init__(**kwargs)

    def call(self, x):
        tensor_shape = K.int_shape(x)
        input_shape = K.shape(x)
        var_x = K.var(x,axis=0)
        loss = K.mean(var_x)
        loss = loss * self.loss_weight
        self.add_loss(loss)
        return x

    def set_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'loss_weight': self.loss_weight}
        base_config = super(gather_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class classfy_loss(Layer):
    def __init__(self, loss_weight=0, **kwargs):
        self.loss_weight = loss_weight
        super(classfy_loss, self).__init__(**kwargs)

    def call(self, inputs):
        tensor_shape = K.int_shape(inputs[0])
        input_shape = K.shape(inputs[0])
        x = inputs[0]
        x_label = inputs[1]
        
        # x = K.print_tensor(x,'x')
        x_transpose = K.transpose(x)
        # x_label = K.print_tensor(x_label,'x_label')
        class_sum = K.dot(x_transpose,x_label)
        class_sum = K.transpose(class_sum)
        # class_sum = K.print_tensor(class_sum*1e20,'class_sum')
        label_sum = K.sum(x_label,axis=0)
        # label_sum = K.print_tensor(label_sum,'label_sum')
        label_sum = K.reshape(label_sum,(K.int_shape(x_label)[1],1))
        label_sum = K.repeat_elements(label_sum,K.int_shape(class_sum)[1],axis=1)
        class_ave = class_sum/label_sum
        
        # class_dis = K.expand_dims(class_ave)
        # class_dis = K.repeat_elements(class_dis,K.int_shape(class_ave)[0],axis=2)
        # class_dis = class_dis - K.permute_dimensions(class_dis,(2,1,0))
        # class_dis = K.sqrt(K.sum(K.square(class_dis),axis=1))
        
        # class_dis = K.mean(class_dis)

        class_dis = K.sum(K.square(class_ave[1] - class_ave[2])) \
                    + K.sum(K.square(class_ave[0] - class_ave[1])) \
                    + K.sum(K.square(class_ave[0] - class_ave[2]))

        # x_sub = x - K.dot(x_label,class_ave)
        # x_sqr = K.square(x_sub)
        # dense_sum = K.dot(K.transpose(x_sqr),x_label)
        # dense_ave = K.transpose(dense_sum)/label_sum
        # dense_dis = K.sqrt(K.sum(dense_ave,axis=1))
        # # dense_dis = K.print_tensor(dense_dis,'dense_dis')
        # # class_dis = K.print_tensor(class_dis,'class_dis')

        # loss_all = dense_dis/class_dis
        # loss_all = K.print_tensor(loss_all,'loss_all')
        loss = K.exp(-class_dis)
        if self.loss_weight == 0:
            loss = 0
        else:
            loss = loss * self.loss_weight
        self.add_loss(loss)
        return x

    def set_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'loss_weight': self.loss_weight}
        base_config = super(classfy_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class adawindow(Layer):
#     def __init__(self, compress_size, activation=None, **kwargs):
#         self.compress_size = compress_size
#         self.activation = KL.activations.get(activation)
#         super(adawindow, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel1 = self.add_weight(name='kernel', 
#                                       shape=(input_shape[2], self.compress_size),
#                                       initializer='uniform',
#                                       trainable=True)
#         self.kernel2 = self.add_weight(name='kernel', 
#                                       shape=(self.compress_size, input_shape[2]),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(adawindow, self).build(input_shape)  # Be sure to call this somewhere!

#     def call(self, x):
#         compress = K.dot(x, self.kernel1)
#         if self.activation is not None:
#             compress = self.activation(compress)
#         window = K.dot(compress, self.kernel2)
#         window = K.sigmoid(window)
#         return  x*window


# class AdaptedDilatedConV1D(KL.Conv1D):

#     def __init__(self, **kwargs):
#         super(AdaptedDilatedConV1D, self).__init__(**kwargs)

#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.dilation = self.add_weight(shape=(1,),
#                                         initializer=keras.initializers.Ones(),
#                                         name='dilation')
#         super(AdaptedDilatedConV1D, self).build(input_shape)  # Be sure to call this somewhere!

#     def call(self, x):
#         return K.dot(x, self.kernel)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

# regulizer
def std_reg(weight_matrix,loss_weight=0.01):
    return loss_weight * K.std(weight_matrix)
# loss_function
def classify_loss(y_pre,y_tru):
    x_transpose = np.transpose(y_pre)
    class_sum = np.dot(x_transpose,y_tru)
    class_sum = np.transpose(class_sum)
    label_sum = np.sum(y_tru,axis=0)
    label_sum = np.reshape(label_sum,(np.shape(y_tru)[1],1))
    label_sum = np.repeat(label_sum,np.shape(class_sum)[1],axis=1)
    class_ave = class_sum/label_sum
    
    class_dis = np.reshape(class_ave,np.shape(class_ave)+(1,))
    class_dis = np.repeat(class_dis,np.shape(class_ave)[0],axis=2)
    class_dis = class_dis - np.transpose(class_dis,(2,1,0))
    class_dis = np.sqrt(np.sum(np.square(class_dis),axis=1))
    
    class_dis = np.mean(class_dis,axis=0)

    x_sub = y_pre - np.dot(y_tru,class_ave)
    x_sqr = np.square(x_sub)
    dense_sum = np.dot(np.transpose(x_sqr),y_tru)
    dense_ave = np.transpose(dense_sum)/label_sum
    dense_dis = np.sqrt(np.sum(dense_ave,axis=1))

    loss_all = dense_dis/class_dis
    loss = np.exp(-np.sum(loss_all))
    return loss

def mmd_loss(y_pre,y_tru):
    mean_y_pre = K.mean(y_pre,axis=0)
    mean_y_tru = K.mean(y_tru,axis=0)
    return K.mean(K.square(mean_y_pre-mean_y_tru))
# function
def Confusion_Matrix(y_tru, y_pre):
    for i in range(y_pre.shape[0]):
        j = np.argmax(y_pre[i,:])
        y_pre[i,:] = np.zeros((1,y_pre.shape[1]))
        y_pre[i, j] = 1
    return np.dot(y_tru.transpose(),y_pre)

def Combined_Dilated_ConV(x, filters, kernel_size, _name, d_rate, activation, is_bn = False):
    outs = []
    for i in range(1, d_rate):
        temp_output = KL.Conv1D(filters, kernel_size = kernel_size, padding = 'same', strides = 1, dilation_rate = i)(x)
        if is_bn:
            temp_output = KL.BatchNormalization(axis = 2)(temp_output)
        temp_output = KL.Activation(activation)(temp_output)
        outs.append(temp_output)

    out = KL.concatenate(outs,axis = 2,name = _name)
    return out

def save_dict(_dict, name, ps = '', save_path = './result/'):
    if '.txt' not in name:
        name = name + '.txt'
    f = open(save_path + name,'w')
    f.writelines(['ps:', ps, '\n'])
    for x in _dict.keys():
        f.writelines([x,'\t'])
    max_len = 0
    for x in _dict.values():
        if len(x) > max_len:
            max_len = len(x)
    for x in _dict.values():
        while(len(x)) < max_len:
            x.append('')
    for i in range(max_len):
        f.write('\n')
        for x in _dict.keys():
            f.writelines([str(round(_dict[x][i],4)),'\t'])
    f.writelines(['\n','end'])
    f.close()
    
def make_fft(data, axis=-1):
    length = np.shape(data)[axis]
    NFFT = 2**math.ceil(math.log2(length)) 
    re_fft_data = np.fft.fft(data, NFFT,axis=axis) 
    return abs(re_fft_data)

def shuffle_data(data, label):
    assert data.shape[0] == label.shape[0]
    index = [x for x in range(data.shape[0])]
    random.shuffle(index)
    r_data = data[index,]
    r_label = label[index,]
    return r_data,r_label
    
def normalize_data(data,select='std'):
    for i in range(data.shape[0]):
        if select == 'fft':
            data[i,] = data[i,] / np.max(data[i,])
        else:
            data[i,] = data[i,] - np.mean(data[i,])
            if select == 'min-max':
                data[i,] = data[i,] / max(np.max(data[i,]),abs(np.min(data[i,])))
            elif select == 'std':
                data[i,] = data[i,] / np.std(data[i,])
            else:
                raise ValueError
    return data

def to_onehot(label,num_class=None):
    if num_class == None:
        r_label = np.zeros((label.shape[0], np.max(label)+1))
    else:
        r_label = np.zeros((label.shape[0],num_class))
    for i, x in enumerate(label):
        r_label[i, int(x)] = 1
    return r_label

def acc(y_tru,y_pre):
    y_tru = np.argmax(y_tru,axis=1)
    y_pre = np.argmax(y_pre,axis=1)
    return np.sum(np.equal(y_tru,y_pre))/y_tru.shape[0]
