# -*- coding: utf-8 -*-
"""
Created on Fri Mar  10 16:36 2018
class dataset
    attribute:
        data

@author: a273
TODO
    multiple labels
"""

import os
import operator
import random
import pywt
import datetime
import scipy.io as sio
import pickle as pickle
import numpy as np
from scipy import interpolate
from functools import reduce

class DataSet():
    '''
        make
            L get_data
                L self.name = germen_data or cwru_data or ours_data
            L process
                L slice
                L label (one-hot)
                L operation = raw or fft or wpd
                L normalize
                L delete fault data
                L make information
            L save
        get
            L load
        use
            L divide into train and test set
    '''
    def __init__(self):
        self.data = []
        self.name = ''
        self.source_path = ''
        self.save_path = '../data/'
        self.number_data = 0
        self.save_his = {}

    def _get_file_data(self, file_name):
        if self.name == 'germen_data':
            # source_path = 'E:/cyh/temp/德data/DATASET/'
            a = sio.loadmat(self.source_path + file_name)
            file_name = file_name.replace('.mat', '')
            a = a[file_name]
            a = a['Y']
            for _ in range(3):
                a = a[0]
            file_data = a[6]
            '''
                0 for force 16005
                1 for phase_current_1 256070
                2 for phase_current_2 256070
                3 for speed 16005
                4 for temp_2_bearing_module 5
                5 for torque 16005
                6 for vibration signal 256070
            '''
            file_data = file_data[2]
            file_data = file_data[0]
            dict_file_data = {}
            dict_file_data['name'] = file_name
            dict_file_data['speed'] = file_name[0:3]
            dict_file_data['bearing'] = file_name[12:16]
            dict_file_data['sample_rate'] = '64'
            dict_file_data['load'] = file_name[4:11]
            dict_file_data['data'] = file_data
            dict_file_data['length'] = np.shape(file_data)[0]
            '''
                TODO
                make a dictionary
            '''
        elif self.name == 'cwru_data':
            file_data = sio.loadmat(self.source_path + file_name)
            dict_file_data = []
            for key in file_data:
                dict_file_data_temp = {}
                if len(file_data[key])>=1000:
                    file_data_temp=file_data[key]
                    file_data_temp = np.reshape(file_data_temp,(-1,))
                    dict_file_data_temp['name'] = file_name+'_'+key
                    dict_file_data_temp['speed'] = 'H30'
                    dict_file_data_temp['bearing'] = file_name[0:4]
                    dict_file_data_temp['sample_rate'] = '48'
                    dict_file_data_temp['load'] = file_name[5:6]+'hp'
                    dict_file_data_temp['data'] = file_data_temp
                    dict_file_data_temp['length'] = np.shape(file_data_temp)[0]
                    dict_file_data.append(dict_file_data_temp)
        elif self.name == 'cwru_data_12k':
            file_data = sio.loadmat(self.source_path + file_name)
            dict_file_data = []
            for key in file_data:
                dict_file_data_temp = {}
                if len(file_data[key])>=1000:
                    file_data_temp=file_data[key]
                    file_data_temp = np.reshape(file_data_temp,(-1,))
                    dict_file_data_temp['name'] = file_name+'_'+key
                    dict_file_data_temp['speed'] = 'H30'
                    dict_file_data_temp['bearing'] = file_name[0:4]
                    dict_file_data_temp['sample_rate'] = '12'
                    dict_file_data_temp['load'] = file_name[5:6]+'hp'
                    dict_file_data_temp['data'] = file_data_temp
                    dict_file_data_temp['length'] = np.shape(file_data_temp)[0]
                    dict_file_data.append(dict_file_data_temp)
        elif self.name == 'ours_data':
            file_data = np.load(self.source_path + file_name)
            dict_file_data = {}
            dict_file_data['name'] = file_name
            dict_file_data['speed'] = file_name[0:3]
            dict_file_data['bearing'] = file_name[4:8]
            dict_file_data['sample_rate'] = file_name[9:11]
            dict_file_data['load'] = ''
            dict_file_data['data'] = file_data
            dict_file_data['length'] = np.shape(file_data)[0]
        else:
            pass

        return dict_file_data

    def process(self):
        if self.name == 'germen_data':
            self.source_path = 'E:/cyh/temp/德data/dataset/'
            file_names = os.listdir(self.source_path)
            file_names.sort()
            for name in file_names:
                if '_9' in name:
                    self.data.append(self._get_file_data(name))
            self.number_data = len(self.data)
        elif self.name == 'cwru_data':
            self.source_path = 'E:/cyh/data_sum/temp/cwru_data/48k/'
            file_names = os.listdir(self.source_path)
            file_names.sort()
            for name in file_names:
                re_dict = self._get_file_data(name)
                for x in re_dict:
                    self.data.append(x)
            self.number_data = len(self.data)
        elif self.name == 'cwru_data_12k':
            self.source_path = 'E:/cyh/data_sum/temp/cwru_data/12k/'
            file_names = os.listdir(self.source_path)
            file_names.sort()
            for name in file_names:
                re_dict = self._get_file_data(name)
                for x in re_dict:
                    self.data.append(x)
            self.number_data = len(self.data)
        elif self.name == 'ours_data':
            self.source_path = '../data/new_data/'
            file_names = os.listdir(self.source_path)
            file_names.sort()
            for name in file_names:
                self.data.append(self._get_file_data(name))
            self.number_data = len(self.data)
        else:
            print('self.name name error!')

    def save(self):
        assert self.name != ''
        assert self.save_path != ''
        # save_class = {'data':self.data,
        #               'name':self.name,
        #               'source_path':self.source_path,
        #               'save_path':self.save_path,
        #               'number_data':self.number_data}
        pickle.dump(self, open(self.save_path + 'DataSet_' +
                                     self.name + '.pkl', 'wb'), True)
        print('dataset ', self.name, ' has benn saved\n')

    def load(self):
        assert self.name != ''
        assert self.save_path != ''
        full_name = self.save_path + 'DataSet_' + self.name + '.pkl'
        load_class = pickle.load(open(full_name, 'rb'))
        assert load_class.name == self.name
        assert load_class.save_path == self.save_path
        # self.data = load_class['data']
        # self.source_path = load_class['source_path']
        # self.number_data = load_class['number_data']
        
        print('dataset ', self.name, ' has been load')
        return load_class

    def gen_data(self,
                 name,
                 fix_factor,
                 train_data_factor,
                 test_data_factor,
                 len_segment,
                 len_shift,
                 label_select='onehot',
                 is_save=False):
        if name != '':
            self.name = name
        para_dict = {
                    'fix_factor':fix_factor,
                    'train_data_factor':train_data_factor,
                    'test_data_factor':test_data_factor,
                    'len_segment':len_segment,
                    'len_shift':len_shift
                }

        # if has the same list dataset
        assert self.name != ''
        assert self.save_path != ''
        if ('DataSet_' + self.name + '.pkl') in os.listdir(self.save_path):
            self = self.load()
        else:
            self.process()
            self.save()

        # if the same factor of use dataset exist in save_his
        is_gen_use_dataset = True

        for key in self.save_his:
            if operator.eq(self.save_his[key], para_dict):
                # if this file exist in save_path, load it
                # else delete the delete the history and regeneral the dataset
                if os.path.exists(self.save_path + 'DataSet_' + key + '.pkl'):
                    use_dataset = pickle.load(open(self.save_path + 'DataSet_' + key + '.pkl', 'rb'))
                    is_gen_use_dataset = False
                else:
                    self.save_his.pop(key)
                    
                break

        if is_gen_use_dataset:
            train_data_list = []
            train_label_list = []
            test_data_list = []
            test_label_list = []
            # modified in 16 Mar
            # if train_data_factor and test_data_factor are dict, e.g. {'K001':0, 'KA01':1, 'KI01':2}
            # elif train_data_factor and test_data_factor are list, then K->0, KA->1, KI->2
            if isinstance(train_data_factor, dict):
                for x in self.data:
                    # if x is what we need
                    # modified in 17 Mar
                    # if fix_factor is empty, then all of data is used
                    if len(fix_factor) == 0 or reduce(lambda a, b: a&b,
                            [_ in list(x.values()) for _ in fix_factor]):
                        # find out x in train_data or test_data
                        is_train_data = [_ in list(x.values()) for _ in list(train_data_factor.keys())]
                        is_test_data = [_ in list(x.values()) for _ in list(test_data_factor.keys())]
                        if reduce(lambda a, b: a|b, is_train_data):
                            train_data_list.append(x)
                            train_label_list.append(list(train_data_factor.values())[is_train_data.index(True)])
                        elif reduce(lambda a, b: a|b, is_test_data):
                            test_data_list.append(x)
                            test_label_list.append(list(test_data_factor.values())[is_test_data.index(True)])
                        else:
                            pass
            elif isinstance(train_data_factor, list):
                for x in self.data:
                    # if x is what we need
                    if len(fix_factor) == 0 or reduce(lambda a, b: a&b,
                            [_ in list(x.values()) for _ in fix_factor]):
                        # temp_label, K->0, KA->1, KI->2
                        if 'K0' in x['bearing']:
                            temp_label = 0
                        elif 'KA' in x['bearing']:
                            temp_label = 1
                        elif 'KI' in x['bearing']:
                            temp_label = 2
                        else:
                            raise Exception('wrong name in dataset.data[\'bearing\']')
                        # find out x in train data or test data
                        if reduce(lambda a, b: a|b, [_ in list(x.values()) for _ in train_data_factor]):
                            train_data_list.append(x)
                            train_label_list.append(temp_label)
                        elif reduce(lambda a, b: a|b, [_ in list(x.values()) for _ in test_data_factor]):
                            test_data_list.append(x)
                            test_label_list.append(temp_label)
                        else:
                            pass
            else:
                raise ValueError('train_data_factor is neither dict nor list!!!')

            train_data, train_label = self._divided(train_data_list,
                                                    train_label_list,
                                                    len_segment,
                                                    len_shift)
            test_data, test_label = self._divided(test_data_list,
                                                test_label_list,
                                                len_segment,
                                                len_shift)

            train_data, train_label = self._normalize(train_data, train_label)
            test_data, test_label = self._normalize(test_data, test_label)

            train_label = self._changelabel(train_label, label_select)
            if test_data != []:
                test_label = self._changelabel(test_label, label_select)

            train_data, train_label = self._shuffle(train_data, train_label)

            use_dataset = {
                'train_data':train_data,
                'train_label':train_label,
                'test_data':test_data,
                'test_label':test_label
            }

            if is_save:
                save_dict_key = self.name + datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")
                
                self.save_his[save_dict_key] = para_dict
                self.save()
                pickle.dump(use_dataset, open(self.save_path + 'DataSet_' +
                                              save_dict_key + '.pkl', 'wb'), True)
        return use_dataset

    def _divided(self, _data, _label, len_segment, len_shift):
        r_label = []
        r_data = []
        for j, x in enumerate(_data):
            num_segment = (x['length'] - len_segment)//len_shift
            for i in range(num_segment):
                r_data.append(x['data'][i * len_shift : i * len_shift + len_segment])
                r_label.append(_label[j])
        return np.array(r_data), r_label

    def _normalize(self, _data, _label):
        delete_list = []
        for i in range(_data.shape[0]):
            if np.max(_data[i,]) > np.min(_data[i,]):
                _data[i,] = _data[i,] - np.mean(_data[i,])
                _data[i,] = _data[i,] / max(abs(np.min(_data[i,])), abs(np.max(_data[i,])))
                # _data[i, ] = _data[i, ] * (_up - _down) + _down
            else:
                delete_list.append(i)

            if len(delete_list):
                _data = np.delete(_data, delete_list, axis=0)
                del _label[delete_list]
                print(len(delete_list),' data have been deleted!')
                # _label = np.delete(_label, delete_list, axis=0)

        return _data, _label

    def _changelabel(self, _label, _select):
        if _select == 'onehot':
            assert isinstance(_label, list)
            r_label = np.zeros((len(_label), max(_label)+1))
            for i, x in enumerate(_label):
                r_label[i, x] = 1
        elif _select == 'scalar':
            assert isinstance(_label, np.ndarray)
            r_label = np.argmax(_label, axis=1)
        else:
            print(type(_label))
            raise NameError
        return r_label

    def _shuffle(self, _data, _label):
        _index = [i for i in range(_data.shape[0])]
        random.shuffle(_index)
        return _data[_index,], _label[_index]


if __name__ == '__main__':
    data_set = DataSet()
    data_set.name = 'cwru_data_12k'
    data_set.process()
    data_set.save()
    # source_data = DataSet()
    # source_data.name = 'cwru_data'
    # source_data.name = 'ours_data'
    # source_data = source_data.load()
    # print('ddd')