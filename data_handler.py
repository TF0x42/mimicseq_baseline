from torch.utils.data import Dataset
import pandas as pd
import multiprocessing
from datetime import timedelta
import numpy as np
import torch

'''
Build custom Dataset class
'''
class MedicalDataset(Dataset):
    def __init__(self, version='train', path = '/home/tobi/cooperation_projects/irregular_time_series/data/big_data', min_num_events=10, days_distance_prediction=1, num_processes=73):
        self.sample_ids=[0]*73
        self.version = version
        self.delta = timedelta(days=days_distance_prediction)
        self.eventtypes = pd.read_parquet(path+'/eventtypes.parquet')
        #print(self.eventtypes)
        self.min_num_events = min_num_events
        if version=='train':
            #self.process_data(3, path)
            #self.multiprocessed_data_loading(num_processes, path)
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

    # def process_data(self, i, path):
    #     print(str(i).zfill(2))
    #     train_data = pd.read_parquet(path + '/train-0000000000' + str(i).zfill(2) + '.parquet')
    #     sample_id_min = int(train_data['sample_id'].min())
    #     sample_id_max = int(train_data['sample_id'].max())
    #     data = []
    #     #self.sample_ids[i]=sample_id_max
    #     return train_data
    #     # for k in range(sample_id_max):
    #     #     tmp = []
    #     #     print(f"Process {i}, k={k}, total={sample_id_max}")
    #     #     filtered_df = train_data[train_data['sample_id'] == sample_id_min + k]
    #     #     tmp.append([k])
    #     #     data_instance = []
    #     #     label = []
    #     #     for l in range(len(filtered_df)):
    #     #         if filtered_df['eventtime'].iloc[len(filtered_df)-1] - filtered_df['eventtime'].iloc[l] > self.delta:
    #     #             data_instance.append(filtered_df['event_id'].iloc[l])
    #     #         else:
    #     #             label.append(filtered_df['event_id'].iloc[l])
    #     #     tmp.append(data_instance)
    #     #     tmp.append(label)
    #     #     data.append(tmp)
    #     # return data

    # def multiprocessed_data_loading(self, num_processes, path):
    #     pool = multiprocessing.Pool(processes=num_processes)
    #     args = [(i, path) for i in range(73)]
    #     results = pool.starmap(self.process_data, args)
    #     self.train_data = [item for sublist in results for item in sublist]
    #     pool.close()
    #     pool.join()

    def __len__(self):
        if self.version=='train':
            return 513741
        if self.version=='val':
            pass
        if self.version=='test':
            return
        

    def __getitem__(self, idx):
        i = 0
        #print("idx: ", idx)
        #print(self.sample_ids)
        while idx > self.sample_ids[i]:
            i+=1
        #print("i: ", i)
        train = np.zeros(88000)
        label2 = np.zeros(88000)

        tmp =[]
        filtered_df = self.train_data[i][self.train_data[i]['sample_id'] == idx]
        tmp.append([idx])
        data_instance = []
        label = []
        for l in range(len(filtered_df)):
            if filtered_df['eventtime'].iloc[len(filtered_df)-1] - filtered_df['eventtime'].iloc[l] > self.delta:
                data_instance.append(filtered_df['event_id'].iloc[l])
            else:
                label.append(filtered_df['event_id'].iloc[l])
        tmp.append(data_instance)
        tmp.append(label)
        #print(tmp)
        for a in tmp[1]:
            # print(a)
            # print(f"len ={len(train)}")
            train[a]=1
        for a in tmp[2]:
            # print(a)
            # print(f"len ={len(label2)}")
            label2[a]=1
        return train.astype(float), label2.astype(float)


# md = MedicalDataset(version='train')

# print(md[51374])
# print("----")
# print(md[392309])
# # print("----")
# # print(md[513740])
# print(len(md))