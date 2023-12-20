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
    def __init__(self, version='train', num_labels='c10', path = '/home/tobi/cooperation_projects/irregular_time_series/data/big_data', min_num_events=10, days_distance_prediction=1, num_processes=73):
        train_val_test = [0.7, 0.1, 0.2] # 49, 7, 14
        self.num_labels=num_labels
        self.sample_ids=[0]*73
        self.version = version
        self.delta = timedelta(days=days_distance_prediction)
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


    def __len__(self):
        if self.version=='train':
            return 1000  #513741
        # if self.version=='val':
        #     return 100
        if self.version=='test':
            return 100
        

    def __getitem__(self, idx):
        # if idx%20==0: # val, 10%= 10/100
        #     pass 
        # elif idx%4==0: # test, 10%=20/100
        #     pass
        # else: # train, 70% = 70/100
        #     pass 
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
                        label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])

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
            print("hello")
            print(filtered_df)
            print("wtf")
            
            #print(filtered_df)
            #print(filtered_df)
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


    # def __getitem__(self, idx):
    #     # Use binary search for efficiency
    #     i = bisect.bisect_right(self.sample_ids, idx) - 1

    #     # Initialize arrays
    #     train = np.zeros(88000)
    #     label2 = np.zeros(88000)

    #     # Filter the dataframe
    #     filtered_df = self.train_data[i][self.train_data[i]['sample_id'] == idx]

    #     # Compute the time difference
    #     time_diff = filtered_df['eventtime'].iloc[-1] - filtered_df['eventtime']

    #     # Use boolean indexing for efficiency
    #     data_instance = filtered_df.loc[time_diff > self.delta, 'event_id']
    #     label = filtered_df.loc[time_diff <= self.delta, 'event_id']

    #     # Set corresponding indices to 1
    #     train[data_instance] = 1
    #     label2[label] = 1

    #     return train.astype(float), label2.astype(float)
    








md = MedicalDataset(version='test', num_labels='c10')
print(md[1])