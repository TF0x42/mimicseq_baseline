from torch.utils.data import Dataset
import pandas as pd
import multiprocessing
from datetime import timedelta
import numpy as np
import torch
import bisect

'''
Build custom Dataset class
'''
class MedicalDataset(Dataset):
    def __init__(self, split_type="1day", version='train', num_labels='c10', path = '/home/tobi/cooperation_projects/irregular_time_series/data/big_data', min_num_events=10, days_distance_prediction=1, num_samples=10000):
        self.num_samples = num_samples
        self.split_type = split_type
        train_val_test = [0.7, 0.1, 0.2] # 49, 7, 14
        self.num_labels=num_labels
        self.sample_ids=[0]*73  # max is 513741
        self.version = version
        self.delta = timedelta(days=days_distance_prediction)
        self.delta2 = timedelta(days=2*days_distance_prediction)
        self.eventtypes = pd.read_parquet(path+'/eventtypes.parquet')
        self.eventtypes.head(100).to_csv("eventtyper_tmp.csv")
        self.min_num_events = min_num_events
        if version=='train':
            self.train_data = []
            for i in range(73):
                print(f"loading instance i={i}")
                train_data = pd.read_parquet(path + '/train-0000000000' + str(i).zfill(2) + '.parquet')
                self.train_data.append(train_data)
            for k in range(len(self.train_data)):
                self.sample_ids[k]=int(self.train_data[k]['sample_id'].max())
        if version=='val':
            pass
        if version=='test':
            self.test_data = pd.read_parquet(path+'/test.parquet')
            #print(set(self.test_data['sample_id'].to_numpy().tolist()))

    def __len__(self):
        if self.version=='train':
            return 500000
            return self.num_samples  #513741
        # if self.version=='val':
        #     return 100
        if self.version=='test':
            return 10000
            return int(self.num_samples/10)
        
    def __getitem__(self, idx):
        ## Split Type: 1 day - 1 day
        if self.split_type=='1day':
            if self.version == 'train':
                # return np.zeros(88000), np.zeros(10)
                i = 0
                while idx > self.sample_ids[i]:
                    i+=1
                #print(idx)
                train = np.zeros(88000)
                label_mapping = {
                    'event_id': 88000,
                    'c10': 10,
                    'c100': 100,
                    'c1000': 1000,
                    'c10000': 10000,
                }
                label2 = np.zeros(label_mapping.get(self.num_labels, 0))
                tmp =[]
                filtered_df = self.train_data[i][self.train_data[i]['sample_id'] == idx]
                #print(filtered_df)
                tmp.append([idx])
                data_instance = []
                label = []
                for l in range(len(filtered_df)):
                    if filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta:
                        if self.num_labels=='event_id':
                            data_instance.append(filtered_df['event_id'].iloc[l])
                        else: 
                            data_instance.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                    elif filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] > self.delta and filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta2:
                        if self.num_labels=='event_id':
                            label.append(filtered_df[self.num_labels].iloc[l])
                        else:
                            try:
                                label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            except:
                                print("some problem")
                tmp.append(data_instance)
                tmp.append(label)
                for a in tmp[1]:
                    train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train, label2 #.astype(float)
            if self.version == 'test':
                train = np.zeros(88000)
                label_mapping = {
                    'event_id': 88000,
                    'c10': 10,
                    'c100': 100,
                    'c1000': 1000,
                    'c10000': 10000,
                }
                label2 = np.zeros(label_mapping.get(self.num_labels, 0))
                tmp =[]
                filtered_df = self.test_data[self.test_data['sample_id'] == idx]
                tmp.append([idx])
                data_instance = []
                label = []
                for l in range(len(filtered_df)):
                    if filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta:
                        if self.num_labels=='event_id':
                            data_instance.append(filtered_df['event_id'].iloc[l])
                        else: 
                            data_instance.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            #print(filtered_df['eventtime'].iloc[l])
                    elif filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] > self.delta and filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta2:
                        if self.num_labels=='event_id':
                            label.append(filtered_df[self.num_labels].iloc[l])
                        else:
                            try:
                                label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            #print(filtered_df['eventtime'].iloc[l])
                            except:
                                print("some problem")
                tmp.append(data_instance)
                tmp.append(label)
                for a in tmp[1]:
                    train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train.astype(float), label2.astype(float)














        ## Split Type: everything but last day - last day
        if self.split_type=='everything_but_last_day':
            if self.version == 'train':
                i = 0
                while idx > self.sample_ids[i]:
                    i+=1
                train = np.zeros(88000)
                label_mapping = {
                    'event_id': 88000,
                    'c10': 10,
                    'c100': 100,
                    'c1000': 1000,
                    'c10000': 10000,
                }
                label2 = np.zeros(label_mapping.get(self.num_labels, 0))
                tmp =[]
                filtered_df = self.train_data[i][self.train_data[i]['sample_id'] == idx]
                tmp.append([idx])
                data_instance = []
                label = []
                for l in range(len(filtered_df)):
                    if filtered_df['eventtime'].iloc[len(filtered_df)-1] - filtered_df['eventtime'].iloc[l] > self.delta:
                        if self.num_labels=='event_id':
                            data_instance.append(filtered_df['event_id'].iloc[l])
                        else: 
                            data_instance.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                    else:
                        if self.num_labels=='event_id':
                            label.append(filtered_df[self.num_labels].iloc[l])
                        else:
                            try:
                                label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            except:
                                print("some problem")
                tmp.append(data_instance)
                tmp.append(label)
                for a in tmp[1]:
                    train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train.astype(float), label2.astype(float)
            if self.version == 'test':
                train = np.zeros(88000)
                label_mapping = {
                    'event_id': 88000,
                    'c10': 10,
                    'c100': 100,
                    'c1000': 1000,
                    'c10000': 10000,
                }
                label2 = np.zeros(label_mapping.get(self.num_labels, 0))
                tmp =[]
                #print(self.test_data)
                filtered_df = self.test_data[self.test_data['sample_id'] == idx]
                tmp.append([idx])
                data_instance = []
                label = []
                for l in range(len(filtered_df)):
                    if filtered_df['eventtime'].iloc[len(filtered_df)-1] - filtered_df['eventtime'].iloc[l] > self.delta:
                        if self.num_labels=='event_id':
                            data_instance.append(filtered_df['event_id'].iloc[l])
                        else: 
                            data_instance.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                    else:
                        if self.num_labels=='event_id':
                            label.append(filtered_df[self.num_labels].iloc[l])
                        else:
                            label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                tmp.append(data_instance)
                tmp.append(label)
                for a in tmp[1]:
                    train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train.astype(float), label2.astype(float)

    
# md = MedicalDataset(split_type='1day', version='train', num_labels='c10')
# print(md[138724])
# for i in range(5000):
#     print(i+269*511)
#     print(md[i+269*511])
            
#md = MedicalDataset(split_type='1day', version='test', num_labels='c10')