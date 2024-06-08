# mimicseq_baseline

##  Data
Put the data files downloaded from mimicseq into the data folder.

## usage
###  2 x 1000 MLP for first day - second day prediction, entire dataset:
python main.py --clustering=c10 --model_name=2x1000_c10_entire.sav --num_layers=2 --hidden_layer_size=1000
python main.py --clustering=c100 --model_name=2x1000_c100_entire.sav --num_layers=2 --hidden_layer_size=1000
python main.py --clustering=c1000 --model_name=2x1000_c1000_entire.sav --num_layers=2 --hidden_layer_size=1000
python main.py --clustering=c10000 --model_name=2x1000_c10000_entire.sav --num_layers=2 --hidden_layer_size=1000

###  2 x 1000 MLP for first day - second day prediction, skip first 100.000 train and 1.000 test samples:
python main.py --clustering=c10 --model_name=2x1000_c10_skip.sav --num_layers=2 --hidden_layer_size=1000 --skip=True
python main.py --clustering=c100 --model_name=2x1000_c100_skip.sav --num_layers=2 --hidden_layer_size=1000 --skip=True
python main.py --clustering=c1000 --model_name=2x1000_c1000_skip.sav --num_layers=2 --hidden_layer_size=1000 --skip=True
python main.py --clustering=c10000 --model_name=2x1000_c10000_skip.sav --num_layers=2 --hidden_layer_size=1000 --skip=True

###  3 x 5000 MLP for first day - second day prediction, entire dataset:
python main.py --clustering=c10000 --model_name=3x5000_c1000_entire.sav --num_layers=3 --hidden_layer_size=5000
python main.py --clustering=c10000 --model_name=3x5000_c10000_skipsav --num_layers=3 --hidden_layer_size=5000 --skip=True



## Note
The dataloader for the test dataset is slower for some batches, because the patients are roughly ordered to length of stay, therefore
more data is loaded into a pandas dataframe, before it is split into first - scecond day. This could be further optimized in the future
for this particular prediction task. However, loading the data in this way allows for end-of-stay prediction.











