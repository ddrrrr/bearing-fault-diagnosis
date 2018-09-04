import NET
import utils
from dataset import DataSet
from usedataset import UseDataSet
import keras
import keras.layers as KL
import keras.backend as K
from keras.models import Model
import numpy as np
import pickle
import scipy.io as sio 
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score
from collections import OrderedDict
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

class GAN_1D():
    def __init__(self):
        self.len_segment = 2048
        self.len_data = 1
        self.time_shift = 1
        self.clip_value = 0.01

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    def confidence_loss(self,y_true,y_pred):
        return K.sum(K.square(K.max(y_true,axis=0)-K.max(y_pred,axis=0)))

    def Conv_M(self,x,filters,activation='linear'):
        y1 = KL.Conv2D(filters[0],(1,1))(x)
        y1 = KL.BatchNormalization()(y1)
        y1 = KL.Activation(activation)(y1)
        y1 = KL.Conv2D(filters[1],(1,3),padding='same')(y1)
        y1 = KL.Conv2D(filters[2],(1,3),padding='same')(y1)
        y1 = KL.BatchNormalization()(y1)
        y1 = KL.Activation(activation)(y1)

        y2 = KL.Conv2D(filters[3],(1,1))(x)
        y2 = KL.BatchNormalization()(y2)
        y2 = KL.Activation(activation)(y2)
        y2 = KL.Conv2D(filters[4],(1,3),dilation_rate=(1,2),padding='same')(y2)
        y2 = KL.Conv2D(filters[5],(1,3),dilation_rate=(1,2),padding='same')(y2)
        y2 = KL.BatchNormalization()(y2)
        y2 = KL.Activation(activation)(y2)

        y3 = KL.Conv2D(filters[6],(1,1))(x)
        y3 = KL.BatchNormalization()(y3)
        y3 = KL.Activation(activation)(y3)
        y3 = KL.Conv2DTranspose(filters[7],(1,3),padding='same')(y3)
        y3 = KL.Conv2DTranspose(filters[7],(1,3),padding='same')(y3)
        y3 = KL.BatchNormalization()(y3)
        y3 = KL.Activation(activation)(y3)

        out = KL.Concatenate()([y1,y2,y3])
        return out
    
    def build_M(self):
        inp = KL.Input(shape=(self.len_data,self.len_segment,1))
        x = inp
        
        x = KL.Reshape((self.len_segment,1))(x)
        x = KL.Conv1D(128,33,strides=1,padding='same',name='conv1')(x)
        x = KL.BatchNormalization(name='bn1')(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.MaxPool1D(32)(x)
        x = KL.Conv1D(128,17,padding='same',name='conv2')(x)
        x = KL.BatchNormalization(name='bn2')(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.MaxPool1D(16)(x)

        x = KL.Flatten()(x)
        out = x
        # out_deconv = y
        return Model([inp], [out])



    def build_C(self):
        inp = KL.Input(shape=(512,))
        x = inp
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(256)(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(3,activation='softmax')(x)
        out = x
        return Model(inp,out)

    def build_D(self):
        inp = KL.Input(shape=(512,))
        x = inp
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(512)(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(128)(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(2,activation='linear')(x)
        out = x
        return Model(inp,out)
        
    def train_s1(self,epochs,batch_size=128):
        # load dataset
        source_data = DataSet()
        source_data.name = 'cwru_data_12k'
        # source_data.name = 'ours_data'
        source_data = source_data.load()
        usedataset = UseDataSet(source_data)
        usedataset.make_data(
                            fix_factor = {'K000':0,'KA07':1,'KA14':2,'KA21':3,'KI07':4,'KI14':5,'KI21':6,'KO07':7,'KO14':8,'KO21':9},
                            # fix_factor = ['K001','K002','KA01','KA02','KI01','KI02'],
                            train_data_factor = ['0hp'],
                            # train_data_factor = ['25'],
                            test_data_factor = ['1hp'],
                            # test_data_factor = ['48'],
                            time_shift = self.time_shift,
                            len_segment = self.len_segment)
        # data preprocess
        train_data, train_label = usedataset.gen_data4epoch(self.len_data, 'train', 10000)
        test_data, test_label = usedataset.gen_data4epoch(self.len_data,'test', 3000)
        # train_data = np.abs(np.fft.fft(train_data,axis=2))
        # test_data = np.abs(np.fft.fft(test_data,axis=2))
        train_data = utils.normalize_data(train_data,'min-max')
        test_data = utils.normalize_data(test_data,'min-max')

        ms = self.build_M()
        c = self.build_C()
        inp = KL.Input(shape=(self.len_data,self.len_segment,1))
        x = ms(inp)
        out = c(x)
        ms_p_c = Model(inp,out)
        ms_p_c.compile(optimizer = keras.optimizers.Adam(2e-4),loss='categorical_crossentropy',metrics=['acc'])
        ms_p_c.summary()
        early_stop = keras.callbacks.EarlyStopping(patience=50)
        val_test = utils.val_test(test_data,test_label)
        save_best_model = keras.callbacks.ModelCheckpoint('./net_weights/MS_P_C.hdf5',save_best_only=True)

        temp_his = ms_p_c.fit(train_data,train_label,epochs=epochs,batch_size=batch_size,validation_split=0.1,callbacks=[val_test,early_stop,save_best_model])
        ms_p_c.load_weights('./net_weights/MS_P_C.hdf5')
        ms.save_weights('./net_weights/MS10.hdf5')
        c.save_weights('./net_weights/C10.hdf5')

        train_data, train_label = usedataset.gen_data4epoch(self.len_data, 'train', 1000)
        test_data, test_label = usedataset.gen_data4epoch(self.len_data,'test', 1000)
        train_data = utils.normalize_data(train_data,'min-max')
        test_data = utils.normalize_data(test_data,'min-max')
        train_fea = ms.predict(train_data)
        test_fea = ms.predict(test_data)
        sio.savemat('cnn_fea_pca.mat',{'train_fea':train_fea,'train_label':train_label,'test_fea':test_fea,'test_label':test_label})

    def train_s2(self,train_factor,test_factor,epochs,epoch_size,batch_size=128,save_interval=10):
        # load dataset
        source_data = DataSet()
        # source_data.name = 'cwru_data_12k'
        source_data.name = 'ours_data'
        source_data = source_data.load()
        usedataset = UseDataSet(source_data)
        usedataset.make_data(
                            # fix_factor = {'K000':0,'KA07':1,'KA14':2,'KA21':3,'KI07':4,'KI14':5,'KI21':6,'KO07':7,'KO14':8,'KO21':9},
                            # fix_factor = ['K001','K002','KA01','KA02','KI01','KI02'],
                            fix_factor = [],
                            train_data_factor = [train_factor],
                            # train_data_factor = ['25'],
                            test_data_factor = [test_factor],
                            # test_data_factor = ['48'],
                            time_shift = self.time_shift,
                            len_segment = self.len_segment)
        # data preprocess
        train_data, train_label = usedataset.gen_data4epoch(self.len_data, 'train', epoch_size)
        test_data, test_label = usedataset.gen_data4epoch(self.len_data,'test', epoch_size)
        # train_data = np.abs(np.fft.fft(train_data,axis=2))
        # test_data = np.abs(np.fft.fft(test_data,axis=2))
        # test_data_0,test_label_0 = usedataset.gen_data4label(self.len_data,'test',0,epoch_size)
        train_data = utils.normalize_data(train_data,'min-max')
        test_data = utils.normalize_data(test_data,'min-max')
        # test_data_0 = utils.normalize_data(test_data_0,'min-max')

        ms = self.build_M()
        # ms.load_weights('./net_weights/MS10.hdf5')
        ms.compile(optimizer=keras.optimizers.Adam(),loss='mse')
        ms.summary()
        c = self.build_C()
        # c.load_weights('./net_weights/C10.hdf5')
        c.compile(optimizer=keras.optimizers.Adam(5e-4),loss='categorical_crossentropy',metrics=['acc'])
        d = self.build_D()
        # d.load_weights('./net_weights/best_v.hdf5')
        d.compile(optimizer=keras.optimizers.Adam(5e-4),loss='mse',metrics=['acc'])

        inp = KL.Input(shape=(self.len_data,self.len_segment,1))
        fea = ms(inp)
        valid = d(fea)
        classify = c(fea)
        # for l in ms.layers:
        #     if l.name in ['conv1','conv2','conv3']:
        #         l.trainable = False
        d.trainable = False
        mt_p_c_d = Model(inp,[classify,valid])
        mt_p_c_d.compile(optimizer=keras.optimizers.Adam(5e-4),loss=['categorical_crossentropy','mse'],loss_weights=[1,1],metrics=['acc'])
        mt_p_c_d.summary()
        self.c_m_acc = 0
        record = OrderedDict({'Dloss':[],'Dacc':[],'Gloss':[],'Gacc':[],'Cacc':[]})
        epoch_record = OrderedDict({'Closs':[],'Cacc':[],'Sloss':[],'Sacc':[]})
        for i in range(epochs):

            for j in range(int(epoch_size/batch_size/8)):
                
                for k in range(2):
                    temp_idx = np.random.randint(0,epoch_size,batch_size)
                    d_fea = ms.predict(np.concatenate([train_data[temp_idx,],test_data[temp_idx,]],axis=0))
                    d_l = keras.utils.to_categorical(np.array(([1]*batch_size+[0]*batch_size)))
                    # d_l = np.array(([1]*batch_size+[-1]*batch_size))
                    # d_loss1 = d.train_on_batch(d_fea[0:batch_size,],d_l[0:batch_size,])
                    # d_loss2 = d.train_on_batch(d_fea[batch_size:-1,],d_l[batch_size:-1,])
                    # d_loss = 0.5 * np.add(d_loss1, d_loss2)
                    d_loss = d.train_on_batch(d_fea,d_l)
                    # for l in d.layers:
                    #     weights = l.get_weights()
                    #     weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    #     l.set_weights(weights)

                    
                
                sample_weights = [np.array(([1] * batch_size + [0] * batch_size)),np.ones((batch_size * 2,))]
                for k in range(1):
                    temp_idx = np.random.randint(0,epoch_size,batch_size)
                    g_data = np.concatenate([train_data[temp_idx,],test_data[temp_idx,]],axis=0)
                    g_label = np.concatenate([train_label[temp_idx,],test_label[temp_idx,]],axis=0)
                    g_valid = keras.utils.to_categorical(np.array(([0]*batch_size+[1]*batch_size)))
                    # g_valid = np.array(([-1]*batch_size+[1]*batch_size))
                    g_loss = mt_p_c_d.train_on_batch(g_data,[g_label,g_valid],sample_weight=sample_weights)
                    # g_data2 = np.concatenate([train_data[temp_idx,],test_data_0[temp_idx,]],axis=0)
                    # g_label2 = np.concatenate([train_label[temp_idx,],test_label_0[temp_idx,]],axis=0)
                    # g_loss2 = mt_p_c_d.train_on_batch(g_data2,[g_label2,g_valid])

                if j % 20 == 0:
                    print ("%d epoch %d batch [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%] [classify acc.: %.2f%%]" % (i,j, d_loss[0], 100*d_loss[1], g_loss[2],100*g_loss[4],100*g_loss[3]))
                # record['Dloss'].append(round(d_loss[0],2))
                # record['Dacc'].append(round(100*d_loss[1],2))
                # record['Gloss'].append(round(g_loss[0],2))
                # record['Gacc'].append(round(100*g_loss[4],2))
                # record['Cacc'].append(round(100*g_loss[3],2))
                # utils.save_dict(record,'gan_record')

            train_fea = ms.predict(train_data)
            train_eva = c.evaluate(train_fea,train_label)
            print(train_eva)
            test_fea = ms.predict(test_data)
            test_eva = c.evaluate(test_fea,test_label)
            print(test_eva)
            epoch_record['Closs'].append(round(test_eva[0],6))
            epoch_record['Cacc'].append(round(test_eva[1],6))
            epoch_record['Sloss'].append(round(train_eva[0],6))
            epoch_record['Sacc'].append(round(train_eva[1],6))
            utils.save_dict(epoch_record,'epoch_record')
            if test_eva[1] >= self.c_m_acc:
                self.c_m_acc = test_eva[1]
                ms.save_weights('./net_weights/best_mt.hdf5')
                d.save_weights('./net_weights/best_v.hdf5')
                c.save_weights('./net_weights/best_c.hdf5')

            if (i+1)%5 == 0:
                K.set_value(mt_p_c_d.optimizer.lr,K.get_value(mt_p_c_d.optimizer.lr)*0.8)
                K.set_value(d.optimizer.lr,K.get_value(d.optimizer.lr)*0.8)
                # mt_p_c_d.compile(optimizer=keras.optimizers.Adam(5e-*0.9**(i/5)),loss=['categorical_crossentropy','categorical_crossentropy'],loss_weights=[1,1],metrics=['acc'])
                # d.compile(optimizer=keras.optimizers.Adam(5e-*0.9**(i/7)),loss='categorical_crossentropy',metrics=['acc'])

        ms.save_weights('./net_weights/last_mt.hdf5')
        d.save_weights('./net_weights/last_v.hdf5')
        c.save_weights('./net_weights/last_c.hdf5')

    def test(self,train_factor,test_factor):
        source_data = DataSet()
        # source_data.name = 'cwru_data_12k'
        source_data.name = 'ours_data'
        source_data = source_data.load()
        usedataset = UseDataSet(source_data)
        usedataset.make_data(
                            # fix_factor = {'K000':0,'KA07':1,'KA14':2,'KA21':3,'KI07':4,'KI14':5,'KI21':6,'KO07':7,'KO14':8,'KO21':9},
                            # fix_factor = ['K001','K002','KA01','KA02','KI01','KI02'],
                            fix_factor = [],
                            train_data_factor = [train_factor],
                            # train_data_factor = ['25'],
                            test_data_factor = [test_factor],
                            # test_data_factor = ['48'],
                            time_shift = self.time_shift,
                            len_segment = self.len_segment)
        # data preprocess
        train_data, train_label = usedataset.gen_data4epoch(self.len_data, 'train', 1000)
        test_data, test_label = usedataset.gen_data4epoch(self.len_data,'test', 1000)
        # train_data = np.abs(np.fft.fft(train_data,axis=2))
        # test_data = np.abs(np.fft.fft(test_data,axis=2))
        # test_data_0,test_label_0 = usedataset.gen_data4label(self.len_data,'test',0,epoch_size)
        train_data = utils.normalize_data(train_data,'min-max')
        test_data = utils.normalize_data(test_data,'min-max')

        ms = self.build_M()
        ms.load_weights('./net_weights/best_mt.hdf5')
        ms.compile(optimizer=keras.optimizers.Adam(),loss='mse')
        c = self.build_C()
        c.load_weights('./net_weights/best_c.hdf5')
        c.compile(optimizer=keras.optimizers.Adam(),loss='categorical_crossentropy')

        train_fea = ms.predict(train_data)
        test_fea = ms.predict(test_data)
        test_pre = c.predict(test_fea)
        sio.savemat('gan_fea_pca4test.mat',{'train_fea':train_fea,'train_label':train_label,'test_fea':test_fea,'test_label':test_label,'test_pre':test_pre})
                

    def save_net(self,net,name):
        time_str = time.strftime('%H_%M_%S',time.localtime(time.time()))
        net.save_weights('./net_weights/'+name+'_'+time_str+'.hdf5')


if __name__ == '__main__':
    acgan = GAN_1D()
    # acgan.train_s1(50)
    acgan.train_s2('48','64',120,10000,batch_size=8,save_interval=10)
    acgan.test('48','64')
    print(acgan.c_m_acc)