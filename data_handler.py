'''
data imports, transformation, train-test-split, shifting
'''


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
        #print(eventtypes)
        print(eventtypes.iloc[18901])
        print(eventtypes.iloc[1184])
        print(eventtypes.iloc[52132])
        print(eventtypes.iloc[34122])

        #print(train_view)
        print(train_view[train_view["hadm_id"]==23697777])
        # import numpy as np
        # event_embeddings_gpt3 = np.load("data/short.npy")
        # print(len(event_embeddings_gpt3))
        # print(len(event_embeddings_gpt3[0]))
        # print(event_embeddings_gpt3)
        # print("-----------------------------")
        # print(test_view)




data_handler = DataHandler()
data_handler.test_data()