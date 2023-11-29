'''
data imports, transformation, train-test-split, shifting
'''

import torch
from torch.utils.data import Dataset
import pandas as pd
import pyarrow.parquet as pq



class DataHandler():
    def __init__(self):
        pass

    def load_data(self):
        pass

    def train_test_split(self):
        pass

    def test_data(self):

        # # Reading a Parquet file
        eventtypes = pd.read_parquet('data/eventtypes.parquet')
        test_view = pd.read_parquet('data/test_view.parquet')
        train_view = pd.read_parquet('data/train_view.parquet')
        # # Viewing the data
        print(eventtypes)
        print(eventtypes.iloc[18901])
        print(eventtypes.iloc[1184])
        print(eventtypes.iloc[52132])
        print(eventtypes.iloc[34122])
        print("-----")
        print(train_view)
        unique_values = train_view["hadm_id"].unique()
        for un in unique_values.tolist():
            print(un)
        print(len(unique_values))

        print(train_view[train_view["hadm_id"]==23697777])
        # import numpy as np
        # event_embeddings_gpt3 = np.load("data/short.npy")
        # print(len(event_embeddings_gpt3))
        # print(len(event_embeddings_gpt3[0]))
        # print(event_embeddings_gpt3)
        # print("-----------------------------")
        # print(test_view)




# data_handler = DataHandler()
# data_handler.test_data()


'''
Build custom Dataset class
'''
class MedicalDataset(Dataset):
    def __init__(self, path = '/home/tobi/cooperation_projects/irregular_time_series/data/big_data'):
        eventtypes = pd.read_parquet(path+'/eventtypes.parquet')
        test_view = pd.read_parquet(path+'/test.parquet')
        self.train_data=[]
        patients = []
        current_sample_id = 0
        self.sample_ids =[]
        for i in range(73):
            if i<10:
                print(i)
                self.train_data.append(pd.read_parquet(path+'/train-00000000000'+str(i)+'.parquet'))
                self.sample_ids.append(int(self.train_data[i]['sample_id'].max()))
                for k in range(len(self.train_data[i])):
                    print(self.train_data[i]['sample_id'][k])
                    print(self.train_data[i].iloc[k])
                    print()
                #filtered_df = train_data[i][train_data[i]['sample_id'] == 441854]
                #print(filtered_df)
            else:
                print(i)
                self.train_data.append(pd.read_parquet(path+'/train-0000000000'+str(i)+'.parquet'))
                self.sample_ids.append(int(self.train_data[i]['sample_id'].max()))
                #filtered_df = train_data[i][train_data[i]['sample_id'] == 441854]
                #print(filtered_df)




    def __len__(self):
        # Return the number of samples in your dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Logic to retrieve a single sample from your dataset
        # Apply any transforms if they exist
        i = 0
        print("idx: ", idx)
        print(self.sample_ids)
        while idx > self.sample_ids[i]:
            i+=1
        print("i: ", i)
        print(len(self.train_data))
        return self.train_data[i][self.train_data[i]['sample_id'] == idx]

    
md = MedicalDataset()

print(md[513741])
