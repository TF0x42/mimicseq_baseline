import argparse
import sys
import torch
from mlp_prediction import train_model, test_model

def get_args():
    parser = argparse.ArgumentParser('Interface for the event prediction task')
    parser.add_argument('--clustering', type=str, help='clustering to be used', default='c10',
                        choices=["c10", "c100", "c1000", "c10000"])
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--model_name', type=str, default='mlp_prediction', help='choose between the 2x1000 MLP and the 3x5000 MLP')
    parser.add_argument('--use_gpu', type=bool, default=False, choices=[False, True], help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--hidden_layer_size', type=int, default=1000, help='dimension of the hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--skip', type=bool, default=False, choices=[False, True], help='skip first 100k train, 1k test samples or not')
    parser.add_argument('--split_type', type=str, default='1day', choices=['1day', 'everything_but_last_day'], help='choose to predict the next day or the last day')
    parser.add_argument('--include_intensities', type=bool, default=False, choices=[False, True], help='include intensities in encoding or not')

    try:
        args = parser.parse_args()
        if args.use_gpu:
            args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
        else:
            args.device = 'cpu'
    except:
        parser.print_help()
        sys.exit()
    return args

if __name__ == "__main__":
    args = get_args()
    print(args)
    train_model(args)
    test_model(args)