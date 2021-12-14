import os
import numpy as np
import pandas as pd
import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import SGD, Adam

from TGNet_NYC_v8 import *
from utils_v8 import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Taxi demand prediction.')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='nyc_taxi')
    # parser.add_argument('--stepsize', type=int, default=1000)
    # parser.add_argument('--sampling_rate', type=float, default=0.08) #0.1)
    # parser.add_argument('--sampling_time', type=int, default=10)
    parser.add_argument('--inp_ch', type=int, default=20)#12)#10)
    parser.add_argument('--out_ch', type=int, default=1)
    # parser.add_argument('--hours2look', type=int, default=4)
    # parser.add_argument('--c2c_window', type=int, default=25)

    # Model parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--f_temporal', type=int, default=64)
    parser.add_argument('--f_gnblock', type=int, default=128)
    parser.add_argument('--reg', type=float, default=1e-4)#2e-5)
    # parser.add_argument('--reg', type=float, default=1e-20)#2e-5)
    parser.add_argument('--p_dropout', type=float, default=0.05)#0.1)#0.05)#0.5)

    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-4)#5e-5)#0.01)#1e-4)
    parser.add_argument('--decay', type=float, default=1e-5)  # 안씀
    parser.add_argument('--batch_size', type=int, default=150) #256)
    parser.add_argument('--epochs', type=int, default=100)#60) #200)

    # Result parameters
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--model_dir', type=str, default='./saved_models/')
    parser.add_argument('--model_name', type=str, default='accvnet1088')
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--traindata', action='store_true')
    # parser.add_argument('--testdata', action='store_true')
    # parser.add_argument('--gen_dat', action='store_true', help='use pre-generated datasets (per each day).')
    parser.add_argument('--traindataname', type=str, default='in_and_out3_100%_1000m_5min_train_2015-01-01_2015-02-09.npy')   # (sequence, longitude#, latitude#, feature depth)  # predict the next or next-next one
    #parser.add_argument('--testdataname', type=str, default='in_and_out3_100%_1000m_5min_test_2015-02-09_2015-03-01.npy')       # (sequence, longitude#, latitude#, feature depth)
    parser.add_argument('--testdataname', type=str, default='in_and_out3_100%_1000m_5min_test2_2015-03-01_2015-04-01.npy')       # (sequence, longitude#, latitude#, feature depth)
    parser.add_argument('--time_step', type=int, default=5)  # 5 min
    parser.add_argument('--pred_ahead', type=int, default=2)  # 2 * args.time_step
    parser.add_argument('--train_stdate', type=str, default="2015-01-01")
    parser.add_argument('--test_stdate', type=str, default="2015-02-09")
    #parser.add_argument('--test_stdate', type=str, default="2015-03-01")
    parser.add_argument('--sampling_rate', type=str, default='100%') #0.1)
    args = parser.parse_args()

    args.traindataname = f'in_and_out3_{args.sampling_rate}_1000m_5min_train_2015-01-01_2015-02-09.npy'
    if not args.test:
        args.testdataname = f'in_and_out3_{args.sampling_rate}_1000m_5min_test_2015-02-09_2015-03-01.npy'
        args.test_stdate = "2015-02-09"
    else:
        args.testdataname = f'in_and_out3_{args.sampling_rate}_1000m_5min_test2_2015-02-09_2015-03-01.npy'
        args.test_stdate = "2015-03-01"

    print(args.traindataname, args.testdataname, args.test_stdate)

    # Set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Set input channels
    # args.lag = int((1/args.sampling_time)*60*args.hours2look)

    # flag_sampling = False
    # if args.sampling_rate < 1.0:
    #     flag_sampling = True
    # if args.dataset == 'nyc_taxi':
    #     args.data_dir = f"./datasets/{args.dataset}/"
    #     args.model_name = f"{args.dataset}_{args.model_name}_sampled_{int(args.sampling_rate*100)}%_{args.stepsize}m_{args.sampling_time}min" if flag_sampling \
    #         else f"{args.dataset}_{args.stepsize}m_{args.sampling_time}min"
    # args.data_dir = f"./presampled_datasets/"

    args.data_dir = f"./"
    args.traindata = np.load(os.path.join(args.data_dir, args.traindataname))   # (?, 10, 20, 10+1)
    args.testdata = np.load(os.path.join(args.data_dir, args.testdataname))   # (?, 10, 20, 10+1)
    args.lat_grid = args.testdata.shape[1]
    args.long_grid = args.testdata.shape[2]

    args.model_name = f"{args.dataset}_presampled_{args.model_name}"

    args.model_input_shape = (args.lat_grid + 2, args.long_grid, args.inp_ch)   # 2 is padding
    args.temporal_shape = 57

    args.train_batches = int(np.floor(args.traindata.shape[0]/args.batch_size))
    args.test_batches = int(np.floor(args.testdata.shape[0]/args.batch_size))
    print("test batches:", args.test_batches)

    # Initialize model
    # if args.traindata or args.testdata:
    #     pass
    # else:
    model = AccVNet_v8(args)
    # model = TGNet_v7(input_shape, dinput_shape, call2call_shape, temporal_shape, vtot_feat_shape, args)
    # model = C2CNet_v5(input_shape, dinput_shape, call2call_shape, temporal_shape, vtot_feat_shape, args)

    # if args.gen_dat:
    #     df = None
    # else:
    #     # Load data
    #     print('Loading data...')
    #     raw_df_2015_01 = pd.read_csv('./datasets/nyc_taxi/yellow_tripdata_2015-01.csv',
    #                                  parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    #     raw_df_2016_01 = pd.read_csv('./datasets/nyc_taxi/yellow_tripdata_2016-01.csv',
    #                                  parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    #     raw_df = pd.concat((raw_df_2015_01, raw_df_2016_01), sort=True)
    #     df = preprocess(dataframe=raw_df, sampling_rate=args.sampling_rate)
    #     df = df.sort_values(by=['tpep_pickup_datetime'])
    #     df = df.reset_index(drop=True)
    #
    # print('Generating latitude & longitude grids...')
    # lat_grid, long_grid = generate_lat_long_grids(long_lb=-74.01905,
    #                                               lat_lb=40.711133,
    #                                               long_ub=-73.942279,
    #                                               lat_ub=40.805323)
    # for lat1, long1, lat2, long2 in zip(lat_grid[:-1], long_grid[:-1], lat_grid[1:], long_grid[1:]):
    #     print(f"* Haversine (sphere): {get_distance(lat1, long1, lat2, long2, mode='haversine')} km")
    #     print(f"* Vincenty (ellipse): {get_distance(lat1, long1, lat2, long2, mode='Vincenty')}\n")

    # Train/test
    # if args.traindata:
    #     print('\n* Train dataset generation')
    #     traindata(df=df, lat_grid=lat_grid, long_grid=long_grid, args=args)
    # elif args.testdata:
    #     print('\n* Test dataset generation')
    #     testdata(df=df, lat_grid=lat_grid, long_grid=long_grid, args=args)
    # elif args.test:
    if args.test:
        print('\n* Test mode')
        model.test(args=args)
    else:
        print('\n* Train mode')
        model.train(args=args)

    print('* Program ends\n')
