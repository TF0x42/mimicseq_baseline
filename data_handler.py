from torch.utils.data import Dataset
import pandas as pd
from datetime import timedelta
import numpy as np
import os



def loading_bar(iteration, total, bar_length=50):
    progress = ((iteration+1) / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    percent = round(progress * 100, 2)
    if progress ==1:
        print(f'[{arrow + spaces}] {percent}% Complete', end='\n')
    else:
        print(f'[{arrow + spaces}] {percent}% Complete', end='\r')


class MedicalDataset(Dataset):
    def __init__(self, split_type="1day", version='train', num_labels='c10', path = os.getcwd()+'/data', days_distance_prediction=1, include_intensities=False, skip=False):
        self.skip=skip
        self.include_intensities = include_intensities
        self.split_type = split_type
        self.num_labels=num_labels
        self.sample_ids=[0]*73  # max is 513741
        self.version = version
        self.delta = timedelta(days=days_distance_prediction)
        self.delta2 = timedelta(days=2*days_distance_prediction)
        self.eventtypes = pd.read_parquet(path+'/eventtypes.parquet')
        #self.eventtypes.head(100).to_csv("eventtyper_tmp.csv")
        if version=='train':
            self.train_data = []
            print("Loading dataset...")
            for i in range(73):
                loading_bar(i, 73)
                train_data = pd.read_parquet(path + '/train-0000000000' + str(i).zfill(2) + '.parquet')
                self.train_data.append(train_data)
            print("Dataset loaded.")
            for k in range(len(self.train_data)):
                self.sample_ids[k]=int(self.train_data[k]['sample_id'].max())
        if version=='val':
            pass
        if version=='test':
            self.test_data = pd.read_parquet(path+'/test.parquet')

    def __len__(self):
        if self.version=='train':
            if self.skip:
                return 513741 -100000
            else:
                return 513741
        if self.version=='test':
            if self.skip:
                return 10000 - 1000
            else:
                return 10000

    def __getitem__(self, idx):
        ## Split Type: 1 day - 1 day
        if self.split_type=='1day':
            if self.version == 'train':
                if self.skip:
                    idx = idx +100000
                else:
                    idx = idx
                i = 0
                while idx > self.sample_ids[i]:
                    i+=1
                train = np.zeros(87899)
                label_mapping = {
                    'event_id': 87899,
                    'c10': 10,
                    'c100': 100,
                    'c1000': 1000,
                    'c10000': 10000,
                }
                label2 = np.zeros(label_mapping.get(self.num_labels, 0))
                tmp =[]
                filtered_df = self.train_data[i][self.train_data[i]['sample_id'] == idx]
                #filtered_df.to_csv("tmp.csv")
                tmp.append([idx])
                data_instance = []
                label = []
                intensities = []
                for l in range(len(filtered_df)):
                    if filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta:
                        if self.num_labels=='event_id':
                            data_instance.append(filtered_df['event_id'].iloc[l])
                        else: 
                            data_instance.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            if not np.isnan(filtered_df['intensity'].iloc[l]):
                                intensities.append(filtered_df['intensity'].iloc[l])
                            else: 
                                intensities.append(1)
                    elif filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] > self.delta and filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta2:
                        if self.num_labels=='event_id':
                            label.append(filtered_df[self.num_labels].iloc[l])
                        else:
                            try:
                                label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            except:
                                pass #two instances in the dataset generate a problem for some reason
                tmp.append(data_instance)
                tmp.append(label)
                for a, b in zip(tmp[1], intensities):
                    if self.include_intensities:
                        train[a]=b
                    else: 
                        train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train, label2 #.astype(float)
            if self.version == 'test':
                if self.skip:
                    idx = idx +1000
                else:
                    idx = idx
                train = np.zeros(87899)
                label_mapping = {
                    'event_id': 87899,
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
                intensities=[]
                self.longenough=False
                for l in range(len(filtered_df)):
                    if filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta:
                        if self.num_labels=='event_id':
                            data_instance.append(filtered_df['event_id'].iloc[l])
                        else: 
                            data_instance.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            if not np.isnan(filtered_df['intensity'].iloc[l]):
                                intensities.append(filtered_df['intensity'].iloc[l])
                            else: 
                                intensities.append(1)
                    elif filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] > self.delta and filtered_df['eventtime'].iloc[l] - filtered_df['eventtime'].iloc[0] < self.delta2:
                        if self.num_labels=='event_id':
                            label.append(filtered_df[self.num_labels].iloc[l])
                        else:
                            try:
                                label.append(self.eventtypes[self.num_labels].iloc[filtered_df['event_id'].iloc[l]])
                            except:
                                pass #two instances in the dataset generate a problem for some reason
                tmp.append(data_instance)
                tmp.append(label)
                for a, b in zip(tmp[1], intensities):
                    if self.include_intensities:
                        train[a]=b
                    else: 
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
                train = np.zeros(87899)
                label_mapping = {
                    'event_id': 87899,
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
                                pass #two instances in the dataset generate a problem for some reason
                tmp.append(data_instance)
                tmp.append(label)
                for a in tmp[1]:
                    if self.include_intensities:
                        train[a]=b
                    else: 
                        train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train.astype(float), label2.astype(float)
            if self.version == 'test':
                train = np.zeros(87899)
                label_mapping = {
                    'event_id': 87899,
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
                                pass #two instances in the dataset generate a problem for some reason
                tmp.append(data_instance)
                tmp.append(label)
                for a in tmp[1]:
                    if self.include_intensities:
                        train[a]=b
                    else: 
                        train[a]=1
                for a in tmp[2]:
                    label2[a]=1
                return train.astype(float), label2.astype(float)
