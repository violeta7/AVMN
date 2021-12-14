import os
import numpy as np
import pandas as pd
import random

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import SGD, Adam

# from TGNet_NYC_v6 import TGNet_v5, C2CNet_v5
# from utils_gen_dat2 import *
from utils_gen_dat3 import *

def traindata(df, lat_grid, long_grid, args):
    gen = data_wrapper(df,
                       lat_grid=lat_grid, long_grid=long_grid,
                       args=args,
                       mode='train',
                       datagen=True)
    next(gen)
    return

def testdata(df, lat_grid, long_grid, args):
    gen_v = data_wrapper(df,
                       lat_grid=lat_grid, long_grid=long_grid,
                       args=args,
                       mode='test',
                       datagen=True)
    next(gen_v)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Taxi demand prediction.')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='nyc_taxi')
    parser.add_argument('--stepsize', type=int, default=1000)
    parser.add_argument('--sampling_rate', type=float, default=1.0) #1.0) #0.1)
    parser.add_argument('--sampling_time', type=int, default=10)
    parser.add_argument('--hours2look', type=int, default=0) #4)
    parser.add_argument('--c2c_window', type=int, default=25)

    # Model parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--f_temporal', type=int, default=64)
    parser.add_argument('--f_gnblock', type=int, default=128)
    parser.add_argument('--reg', type=float, default=1e-4)#2e-5)
    parser.add_argument('--p_dropout', type=float, default=0.5)

    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=150) #256)
    parser.add_argument('--epochs', type=int, default=200)

    # Result parameters
    parser.add_argument('--output_dir', type=str, default='./outputs/')
    parser.add_argument('--model_dir', type=str, default='./saved_models/')
    parser.add_argument('--model_name', type=str, default='tgnet01')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--traindata', action='store_true')
    parser.add_argument('--testdata', action='store_true')
    parser.add_argument('--start_date', type=str, default=None)
    parser.add_argument('--end_date', type=str, default=None)
    args = parser.parse_args()

    # Set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Set input channels
    args.lag = int((1/args.sampling_time)*60*args.hours2look)

    #
    flag_sampling = False
    if args.sampling_rate < 1.0:
        flag_sampling = True
    if args.dataset == 'nyc_taxi':
        args.data_dir = f"./datasets/{args.dataset}/"
        args.model_name = f"{args.dataset}_{args.model_name}_sampled_{int(args.sampling_rate*100)}%_{args.stepsize}m_{args.sampling_time}min" if flag_sampling \
            else f"{args.dataset}_{args.stepsize}m_{args.sampling_time}min"

        # input_shape = (10, 20, args.lag)
        # dinput_shape = (10, 20, 2*args.lag)
        # call2call_shape = (10, 20, 5)
        # Is any bettery way to define shapes?
        input_shape = (12, 20, args.lag)
        dinput_shape = (12, 20, 2*args.lag)
        call2call_shape = (12, 20, args.c2c_window) #180) #args.lag*10) #C_INP_WINDOW
        vtot_feat_shape = (args.lag)
        temporal_shape = ((int((1/args.sampling_time)*60*24)*7), )
    else:
        raise IOError(repr("You must specify --dataset !"))
    if args.sampling_rate == 0.1:
        args.train_batch_size = 861615 #859937
        args.test_batch_size = 757269 #756401
    elif args.sampling_rate == 0.05:
        args.train_batch_size = 484486
        args.test_batch_size = 419537
    elif args.sampling_rate == 0.03:
        args.train_batch_size = 306299 #334207
        args.test_batch_size = 262700 #284270
    elif args.sampling_rate == 0.01:
        args.train_batch_size = 110334 #106631
        args.test_batch_size = 93942 #92106
    else:  # 100% / 10
        args.train_batch_size = 90000000
        args.test_batch_size = 90000000
    # args.train_batch_size = 107470 * args.sampling_rate
    # args.test_batch_size  = 10740080 * args.sampling_rate

    # Initialize model
    if args.traindata or args.testdata:
        pass
    else:
        # model = TGNet_v5(input_shape, dinput_shape, call2call_shape, temporal_shape, vtot_feat_shape, args)
        # model = C2CNet_v5(input_shape, dinput_shape, call2call_shape, temporal_shape, vtot_feat_shape, args)
        pass

    if args.traindata and os.path.exists(f"timeq_{int(args.sampling_rate*100)}%_train.pkl"):
        print("pickled data exists.")
        df = None
    elif False and args.testdata and os.path.exists(f"timeq_{int(args.sampling_rate*100)}%_test.pkl"):
        print("pickled data exists.")
        df = None
    else:
        # Load data
        print('Loading data...')
        raw_df_2015_01 = pd.read_csv('./datasets/nyc_taxi/yellow_tripdata_2015-01.csv',
                                     parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        # raw_df_2016_01 = pd.read_csv('./datasets/nyc_taxi/yellow_tripdata_2016-01.csv',
        raw_df_2015_02 = pd.read_csv('./datasets/nyc_taxi/yellow_tripdata_2015-02.csv',
                                     parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        raw_df_2015_03 = pd.read_csv('./datasets/nyc_taxi/yellow_tripdata_2015-03.csv',
                                     parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        raw_df = pd.concat((raw_df_2015_01, raw_df_2015_02, raw_df_2015_03), sort=True)
        df = preprocess(dataframe=raw_df, sampling_rate=args.sampling_rate)
        df = df.sort_values(by=['tpep_pickup_datetime'])
        df = df.reset_index(drop=True)

    print('Generating latitude & longitude grids...')
    lat_grid, long_grid = generate_lat_long_grids(long_lb=-74.01905,
                                                  lat_lb=40.711133,
                                                  long_ub=-73.942279,
                                                  lat_ub=40.805323)
    for lat1, long1, lat2, long2 in zip(lat_grid[:-1], long_grid[:-1], lat_grid[1:], long_grid[1:]):
        print(f"* Haversine (sphere): {get_distance(lat1, long1, lat2, long2, mode='haversine')} km")
        print(f"* Vincenty (ellipse): {get_distance(lat1, long1, lat2, long2, mode='Vincenty')}\n")

    # Train/test
    if args.traindata:
        print('\n* Train dataset generation')
        traindata(df=df, lat_grid=lat_grid, long_grid=long_grid, args=args)
    elif args.testdata:
        print('\n* Test dataset generation')
        testdata(df=df, lat_grid=lat_grid, long_grid=long_grid, args=args)
    elif args.test:
        print('\n* Test mode')
        # model.test(df=df, lat_grid=lat_grid, long_grid=long_grid, args=args)
    else:
        print('\n* Train mode')
        # model.train(df=df, lat_grid=lat_grid, long_grid=long_grid, args=args)

    print('* Program ends\n')
