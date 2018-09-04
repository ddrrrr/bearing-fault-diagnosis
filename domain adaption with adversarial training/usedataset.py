"""
Created on Mon Apr  9 15:54 2018
class usedataset

@author: a273
TODO
"""

import random
from dataset import DataSet
import scipy.io as sio
import pickle as pickle
import numpy as np
from functools import reduce

class UseDataSet():
    def __init__(self, source_data):
        
        self.source_data = source_data

    def make_data(self,fix_factor,train_data_factor,test_data_factor,time_shift,len_segment,label_type='onehot'):
        self.train_data_list = []
        self.train_data_start_list = []
        self.train_label_list = []
        self.test_data_list = []
        self.test_data_start_list =[]
        self.test_label_list = []
        self.len_segment = len_segment
        # modified in 16 Mar
        # if train_data_factor and test_data_factor are dict, e.g. {'K001':0, 'KA01':1, 'KI01':2}
        # elif train_data_factor and test_data_factor are list, then K->0, KA->1, KI->2
        if len(fix_factor)>0 and isinstance(fix_factor,dict):
            for x in self.source_data.data:
                is_fix_data = [temp for temp in fix_factor.keys() if temp in x.values()]
                if len(is_fix_data)>0:
                    temp_label = fix_factor.get(is_fix_data[0])
                    if len([temp for temp in train_data_factor if temp in x.values()])>0:
                        self.train_data_list.append(x)
                        self.train_label_list.append(temp_label)
                    if len([temp for temp in test_data_factor if temp in x.values()])>0:
                        self.test_data_list.append(x)
                        self.test_label_list.append(temp_label)
                    else:
                        pass
        else:
            if isinstance(train_data_factor, dict):
                for x in self.source_data.data:
                    # if x is what we need
                    # modified in 17 Mar
                    # if fix_factor is empty, then all of data is used
                    if len(fix_factor) == 0 or len([temp for temp in fix_factor if temp in x.values()])>0:
                        # find out x in train_data or test_data
                        is_train_data = [temp for temp in train_data_factor.keys() if temp in x.values()]
                        is_test_data = [temp for temp in test_data_factor.keys() if temp in x.values()]
                        if len(is_train_data)>0:
                            self.train_data_list.append(x)
                            self.train_label_list.append(train_data_factor.get(is_train_data[0]))
                        if len(is_test_data)>0:
                            self.test_data_list.append(x)
                            self.test_label_list.append(test_data_factor.get(is_test_data[0]))
                        else:
                            pass
            elif isinstance(train_data_factor, list):
                for x in self.source_data.data:
                    # if x is what we need
                    if len(fix_factor) == 0 or len([temp for temp in fix_factor if temp in x.values()])>0:
                        # temp_label, K->0, KA->1, KI->2
                        if 'K0' in x['bearing']:
                            temp_label = 0
                        elif 'KA' in x['bearing']:
                            temp_label = 1
                        elif 'KI' in x['bearing']:
                            temp_label = 2
                        else:
                            pass
                        # find out x in train data or test data
                        if len([temp for temp in train_data_factor if temp in x.values()])>0:
                            self.train_data_list.append(x)
                            self.train_label_list.append(temp_label)
                        if len([temp for temp in test_data_factor if temp in x.values()])>0:
                            self.test_data_list.append(x)
                            self.test_label_list.append(temp_label)
                        else:
                            pass
            else:
                raise ValueError('train_data_factor is neither dict nor list!!!')
                
        # devide
        self.train_data_start_list, self.train_label = self.devide(self.train_data_list, self.train_label_list, len_segment = len_segment, time_shift = time_shift)
        self.test_data_start_list, self.test_label = self.devide(self.test_data_list, self.test_label_list, len_segment = len_segment, time_shift = time_shift)
        # change label
        self.train_label = self.change_label(self.train_label, label_type)
        self.test_label = self.change_label(self.test_label, label_type)
        self.train_label = np.array(self.train_label)
        self.test_label = np.array(self.test_label)


    def devide(self, _data_list, _label_list, len_segment, time_shift):
        start_point_list = []
        r_label = []
        # r_data = []
        for j, x in enumerate(_data_list):
            if x['sample_rate'] == '64':
                len_shift = round(64 * time_shift)
            elif x['sample_rate'] == '48':
                len_shift = round(48 * time_shift)
            elif x['sample_rate'] == '25':
                len_shift = round(25.6 * time_shift)
            elif x['sample_rate'] == '12':
                len_shift = round(12 * time_shift)
            else:
                raise ValueError('this data', x['name'], 'has other sample rate', x['sample_rate'])
            num_segment = (x['length'] - len_segment)//len_shift
            start_point_list.append([i for i in range(0,num_segment*len_shift,len_shift)])
            # temp_data_list = []
            # for i in range(num_segment):
            #     temp_data = (x['data'][i * len_shift : i * len_shift + len_segment])
            #     if max(temp_data) > min(temp_data):
            #         temp_data_list.append(temp_data)
            # r_data.append(np.array(temp_data_list))
            r_label.append(_label_list[j])
        return start_point_list, r_label
    
    def change_label(self, _label, _select):
        if _select == 'onehot':
            assert isinstance(_label, list)
            r_label = np.zeros((len(_label), max(_label)+1))
            for i, x in enumerate(_label):
                r_label[i, x] = 1
        else:
            r_label = np.transpose(np.array(_label))
        return r_label

    def gen_sample_data(self,_data,_len_segment,_start_list):
        r_data = []
        for x in _start_list:
            # gen_augment_data
            x = x + np.random.normal(0, 3)
            x = max(round(x),0)
            temp_data = _data['data'][x : x + _len_segment]
            if max(temp_data) <= min(temp_data):
                return -1
            r_data.append(temp_data)
        return np.array(r_data)

    def gen_data4batch(self, len_data, type_data, batchsize):
        if type_data == 'train':
            data = self.train_data_list
            start_list = self.train_data_start_list
            label = self.train_label
        elif type_data == 'test':
            data = self.test_data_list
            start_list = self.test_data_start_list
            label = self.test_label
        else:
            raise ValueError
        while True:
            batch_data = []
            batch_label = []
            seq_list = [random.randint(0, len(data)-1) for i in range(batchsize)]
            for x in seq_list:
                temp_data = data[x]
                count = -1
                while count == -1:
                    start_row = random.randint(0,len(start_list[x])-len_data-1)
                    temp_start_list = start_list[x][start_row:start_row+len_data]
                    data_from_gen_sample_data = self.gen_sample_data(temp_data,self.len_segment,temp_start_list)
                    if isinstance(data_from_gen_sample_data,np.ndarray):
                        count = 1
                batch_data.append(data_from_gen_sample_data)
                batch_label.append(label[x,:])

            batch_data = np.array(batch_data)
            batch_data = batch_data[:,:,:,np.newaxis]
            yield batch_data, np.array(batch_label)


    def gen_data4epoch(self, len_data, type_data, num_data):
        if type_data == 'train':
            data = self.train_data_list
            start_list = self.train_data_start_list
            label = self.train_label
        elif type_data == 'test':
            data = self.test_data_list
            start_list = self.test_data_start_list
            label = self.test_label
        else:
            raise ValueError
        batch_data = []
        batch_label = []
        seq_list = []
        for i in range(label.shape[1]):
            j = 0
            while j < num_data/(label.shape[1]):
                idx = random.randint(0,len(data)-1)
                if np.argmax(label[idx,]) == i:
                    seq_list.append(idx)
                    j = j + 1

        random.shuffle(seq_list)
        # seq_list = [random.randint(0, len(data)-1) for i in range(num_data)]
        for x in seq_list:
            temp_data = data[x]
            count = -1
            while count == -1:
                start_row = random.randint(0,len(start_list[x])-len_data-1)
                temp_start_list = start_list[x][start_row:start_row+len_data]
                data_from_gen_sample_data = self.gen_sample_data(temp_data,self.len_segment,temp_start_list)
                if isinstance(data_from_gen_sample_data,np.ndarray):
                    count = 1
            batch_data.append(data_from_gen_sample_data)
            batch_label.append(label[x,])

        batch_data = np.array(batch_data)
        batch_data = batch_data[:,:,:,np.newaxis]
        return batch_data, np.array(batch_label)

    def gen_data4label(self, len_data, type_data, label_data, num_data):
        if type_data == 'train':
            data = self.train_data_list
            start_list = self.train_data_start_list
            label = self.train_label
        elif type_data == 'test':
            data = self.test_data_list
            start_list = self.test_data_start_list
            label = self.test_label
        else:
            raise ValueError
        data_list = []
        for i in range(label.shape[0]):
            if np.argmax(label[i]) == label_data:
                data_list.append(i)
        seq_list = [random.randint(0, len(data_list)-1) for i in range(num_data)]
        r_data = []
        r_label = []
        for x in seq_list:
            temp_data = data[data_list[x]]
            count = -1
            while count == -1:
                start_row = random.randint(0,len(start_list[data_list[x]])-len_data-1)
                temp_start_list = start_list[data_list[x]][start_row:start_row+len_data]
                data_from_gen_sample_data = self.gen_sample_data(temp_data,self.len_segment,temp_start_list)
                if isinstance(data_from_gen_sample_data,np.ndarray):
                    count = 1
            r_data.append(self.gen_sample_data(temp_data,self.len_segment,temp_start_list))
            r_label.append(label[data_list[x],:])

        r_data = np.array(r_data)
        r_data = r_data[:,:,:,np.newaxis]
        return r_data, np.array(r_label)
            

if __name__ == '__main__':
    source_data = DataSet()
    source_data.name = 'ours_data'
    source_data = source_data.load()
    usedataset = UseDataSet(source_data)
    usedataset.make_data(fix_factor = [],
                        train_data_factor = ['H15'],
                        test_data_factor = ['H17'],
                        time_shift = 2.5,
                        len_segment = 196)