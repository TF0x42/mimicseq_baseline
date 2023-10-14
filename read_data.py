import pandas as pd
import pyarrow.parquet as pq

# Reading a Parquet file
eventtypes = pd.read_parquet('data/eventtypes.parquet')
test_view = pd.read_parquet('data/test_view.parquet')
train_view = pd.read_parquet('data/train_view.parquet')

# Viewing the data
print(eventtypes.head(10))