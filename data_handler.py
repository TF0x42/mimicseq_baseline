from torch.utils.data import Dataset
import pandas as pd
import multiprocessing
from datetime import timedelta

'''
Build custom Dataset class
'''
class MedicalDataset(Dataset):
    def __init__(self, path = '/home/tobi/cooperation_projects/irregular_time_series/data/big_data', min_num_events=10, days_distance_prediction=1, num_processes=73):
        self.delta = timedelta(days=days_distance_prediction)
        self.eventtypes = pd.read_parquet(path+'/eventtypes.parquet')
        test_view = pd.read_parquet(path+'/test.parquet')
        self.min_num_events = min_num_events
        self.process_data(3, path)
        #self.multiprocessed_data_loading(num_processes, path)



    def process_data(self, i, path):
        print(str(i).zfill(2))
        train_data = pd.read_parquet(path + '/train-0000000000' + str(i).zfill(2) + '.parquet')
        sample_id_min = int(train_data['sample_id'].min())
        sample_id_max = int(train_data['sample_id'].max())
        data = []
        for k in range(sample_id_max):
            tmp = []
            print(f"Process {i}, k={k}, total={sample_id_max}")
            filtered_df = train_data[train_data['sample_id'] == sample_id_min + k]
            tmp.append([k])
            data_instance = []
            label = []
            for l in range(len(filtered_df)):
                if filtered_df['eventtime'].iloc[len(filtered_df)-1] - filtered_df['eventtime'].iloc[l] > self.delta:
                    data_instance.append(filtered_df['event_id'].iloc[l])
                else:
                    label.append(filtered_df['event_id'].iloc[l])
            tmp.append(data_instance)
            tmp.append(label)
            data.append(tmp)
        return data

    def multiprocessed_data_loading(self, num_processes, path):
        pool = multiprocessing.Pool(processes=num_processes)
        args = [(i, path) for i in range(73)]
        results = pool.starmap(self.process_data, args)
        self.train_data = [item for sublist in results for item in sublist]
        pool.close()
        pool.join()







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
