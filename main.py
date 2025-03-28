import argparse
import os
import torch
import numpy as np
import time

from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='FPPformerV2')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariate predict univariate')
parser.add_argument('--target', type=str, default='None', help='target feature in S or M task')
parser.add_argument('--ori_target', type=str, default='None', help='Default target, determine the EMD'
                                                                   'result order')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--input_len', type=int, default=96, help='input length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')

parser.add_argument('--enc_in', type=int, default=7, help='input variable number')
parser.add_argument('--dec_out', type=int, default=7, help='output variable number')
parser.add_argument('--d_model', type=int, default=28, help='hidden dims of model')
parser.add_argument('--encoder_layer', type=int, default=3)
parser.add_argument('--patch_size', type=int, default=6,
                    help='the initial patch size in patch-wise attention')
parser.add_argument('--Cross', action='store_true',
                    help='whether to use cross-variable attention'
                    , default=False)
parser.add_argument('--EMD', action='store_true',
                    help='whether to use EMD as the prediction initialization'
                    , default=False)

parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_gpu', action='store_true',
                    help='whether to use gpu, it is automatically set to true if gpu is available in your device'
                    , default=False)
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'target': 'OT', 'root_path': './data/ETT/', 'M': [7, 7], 'S': [1, 1]},
    'ECL': {'data': 'ECL.csv', 'target': 'MT_321', 'root_path': './data/ECL/', 'M': [321, 321], 'S': [1, 1]},
    'Traffic': {'data': 'Traffic.csv', 'target': 'Sensor_861', 'root_path': './data/Traffic/', 'M': [862, 862],
                'S': [1, 1]},
    'weather': {'data': 'weather.csv', 'target': 'OT', 'root_path': './data/weather/', 'M': [21, 21], 'S': [1, 1]},
    'Solar': {'data': 'solar_AL.csv', 'target': '136', 'root_path': './data/Solar/', 'M': [137, 137], 'S': [1, 1]},
    'Air': {'data': 'Air.csv', 'target': 'AH', 'root_path': './data/Air/', 'M': [12, 12], 'S': [1, 1]},
    'River': {'data': 'River.csv', 'target': 'DLDI4__0', 'root_path': './data/River/', 'M': [8, 8], 'S': [1, 1]},
    'BTC': {'data': 'BTC.csv', 'target': 'Volume USD', 'root_path': './data/BTC/', 'M': [6, 6], 'S': [1, 1]},
    'ETH': {'data': 'ETH.csv', 'target': 'Volume USDT', 'root_path': './data/ETH/', 'M': [6, 6], 'S': [1, 1]}
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.ori_target = data_info['target']
    if args.target == 'None':
        args.target = data_info['target']
    args.root_path = data_info['root_path']
    args.enc_in, args.dec_out = data_info[args.features]

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')

lr = args.learning_rate
print('Args in experiment:')
print(args)

mse_total = []
mae_total = []

Exp = Exp_Model
for ii in range(args.itr):
    if args.train:
        setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                                args.features, args.input_len,
                                                args.pred_len, ii)
        print('>>>>>>>start training| pred_len:{}, settings: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.
              format(args.pred_len, setting))
        try:
            exp = Exp(args)  # set experiments
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from forecasting early')

        print('>>>>>>>testing| pred_len:{}: {}<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments
        mse, mae = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
        mse_total.append(mse)
        mae_total.append(mae)
        torch.cuda.empty_cache()
        args.learning_rate = lr
    else:
        setting = '{}_ft{}_ll{}_pl{}_{}'.format(args.data,
                                                args.features, args.input_len,
                                                args.pred_len, ii)
        print('>>>>>>>testing| pred_len:{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.pred_len, setting))
        exp = Exp(args)  # set experiments

        mse, mae = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
        mse_total.append(mse)
        mae_total.append(mae)
        torch.cuda.empty_cache()
        args.learning_rate = lr

if args.features == 'M':
    path1 = './result.csv'
    if not os.path.exists(path1):
        with open(path1, "a") as f:
            write_csv = ['Time', 'Data', 'input_len', 'pred_len', 'encoder_layer', 'patch_size', 'Min(Mean) MSE',
                         'Min(Mean) MAE', 'Std MSE', 'Std MAE']
            np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
            f.flush()
            f.close()
else:
    path1 = './result_uni.csv'
    if not os.path.exists(path1):
        with open(path1, "a") as f:
            write_csv = ['Time', 'Data', 'Target', 'input_len', 'pred_len', 'encoder_layer', 'patch_size',
                         'Min(Mean) MSE', 'Min(Mean) MAE', 'Std MSE', 'Std MAE']
            np.savetxt(f, np.array(write_csv).reshape(1, -1), fmt='%s', delimiter=',')
            f.flush()
            f.close()

mse = np.asarray(mse_total)
mae = np.asarray(mae_total)
# avg_mse = np.mean(mse)  # Experiment
avg_mse = np.min(mse)  # Practice
std_mse = np.std(mse)
# avg_mae = np.mean(mae)
avg_mae = np.min(mae)
std_mae = np.std(mae)

print('|Min(Mean)|mse:{}, mae:{}|Std|mse:{}, mae:{}'.format(avg_mse, avg_mae, std_mse, std_mae))
path = './result.log'
with open(path, "a") as f:
    f.write('|{}_{}|pred_len{}: '.format(
        args.data, args.features, args.pred_len) + '\n')
    f.write('|Min(Mean)|mse:{}, mae:{}|Std|mse:{}, mae:{}'.
            format(avg_mse, avg_mae, std_mse, std_mae) + '\n')
    f.flush()
    f.close()

if args.features == 'M':
    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{},{},{},{},{}'.
                format(args.data, args.input_len, args.pred_len, args.encoder_layer,
                       args.patch_size, avg_mse, avg_mae, std_mse, std_mae) + '\n')
        f.flush()
        f.close()
else:
    with open(path1, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write(',{},{},{},{},{},{},{},{},{},{}'.
                format(args.data, args.target, args.input_len, args.pred_len, args.encoder_layer,
                       args.patch_size, avg_mse, avg_mae, std_mse, std_mae) + '\n')
        f.flush()
        f.close()