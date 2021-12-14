import os
import pickle
import time

import numpy as np
import pandas as pd
import random

from bisect import bisect
from geopy.distance import distance as dist_Vincenty
from math import radians, cos, sin, asin, sqrt

import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow as tf

# week of the day --> onehot function
# st: starting times (in 10min), pandas.datetime
# index: dataset index
def onehotweek(st, index, to_int=False):
    global time_slice
    a_day = 6*24
    a_week = a_day*7
    weekday = st.dayofweek
    hour = st.hour
    minute = st.minute
    time_slice = 5  # 5 min --> this should be identical with time slice of the input volume matrix.

    #print(weekday, hour, minute, index)
    index_timeslot = int((weekday*a_day + hour*6 + minute//time_slice) + index) % a_week

    if to_int:
        return index_timeslot

    onehot = np.zeros([a_week], dtype=np.uint8)
    onehot[index_timeslot] = 1   # index should be int.

    return np.array(onehot)

def temporal_encoding(dt):
    '''
    * dt: datetime
    * mat_tod (dim. 48):
        00:00:00 - 00:29:59 --> idx 0
        00:30:00 - 00:59:59 --> idx 1
        01:00:00 - 01:29:59 --> idx 2
        ...
        23:30:00 - 23:59:59 --> idx 47
    * mat_dow (dim. 7):
        Mon. --> idx 0
        Tue. --> idx 1
        Wed. --> idx 2
        ...
        Sun. --> idx 6
    * mat_holiday (dim. 1):
        dt is a holiday --> 1
        else --> 0
    * mat_holiday_soon (dim. 1):
        dt is one day before a holiday --> 1
        else --> 0
    '''

    # Time of day (tod) and day of week (dow)
    hour, minute = dt.hour, dt.minute
    idx_tod = 2*hour%48 + round((minute+1)/60)
    idx_dow = dt.dayofweek
    mat_tod = np.zeros(48)
    mat_dow = np.zeros(7)
    mat_tod[idx_tod] = 1.
    mat_dow[idx_dow] = 1.

    # Official holidays in 2015 {month: day} (https://www.timeanddate.com/holidays/us/2015?hol=1)
    holiday_dict = {1: [1, 19], 2: [16], 5: [25], 7: [3, 4], 9: [7], 10: [12], 11: [11, 26], 12: [25]}
    month, day = dt.month, dt.day
    if month in holiday_dict:
        if day in holiday_dict.get(month):
            mat_holiday = np.ones(1)
        else:
            mat_holiday = np.zeros(1)
        if (dt + pd.Timedelta(days=1)).day in holiday_dict.get(month):
            mat_holiday_soon = np.ones(1)
        else:
            mat_holiday_soon = np.zeros(1)

    return np.concatenate((mat_tod, mat_dow, mat_holiday, mat_holiday_soon))


def temporal_encoding2(index, args, mode='train'):
    '''
    * dt: datetime
    * mat_tod (dim. 48):
        00:00:00 - 00:29:59 --> idx 0
        00:30:00 - 00:59:59 --> idx 1
        01:00:00 - 01:29:59 --> idx 2
        ...
        23:30:00 - 23:59:59 --> idx 47
    * mat_dow (dim. 7):
        Mon. --> idx 0
        Tue. --> idx 1
        Wed. --> idx 2
        ...
        Sun. --> idx 6
    * mat_holiday (dim. 1):
        dt is a holiday --> 1
        else --> 0
    * mat_holiday_soon (dim. 1):
        dt is one day before a holiday --> 1
        else --> 0
    '''

    if mode == 'train':
        st = pd.to_datetime(args.train_stdate)
    elif mode == 'test':
        st = pd.to_datetime(args.test_stdate)
    dt = st + pd.Timedelta("1 min") * args.time_step * (index + args.pred_ahead)
    # print(dt)

    # Time of day (tod) and day of week (dow)
    hour, minute = dt.hour, dt.minute
    idx_tod = 2*hour%48 + round((minute+1)/60)
    idx_dow = dt.dayofweek
    mat_tod = np.zeros(48)
    mat_dow = np.zeros(7)
    mat_tod[idx_tod] = 1.
    mat_dow[idx_dow] = 1.

    # Official holidays in 2015 {month: day} (https://www.timeanddate.com/holidays/us/2015?hol=1)
    holiday_dict = {1: [1, 19], 2: [16], 5: [25], 7: [3, 4], 9: [7], 10: [12], 11: [11, 26], 12: [25]}
    month, day = dt.month, dt.day
    if month in holiday_dict:
        if day in holiday_dict.get(month):
            mat_holiday = np.ones(1)
        else:
            mat_holiday = np.zeros(1)
        if (dt + pd.Timedelta(days=1)).day in holiday_dict.get(month):
            mat_holiday_soon = np.ones(1)
        else:
            mat_holiday_soon = np.zeros(1)
    else:
        mat_holiday = np.zeros(1)
        mat_holiday_soon = np.zeros(1)
        
    return np.concatenate((mat_tod, mat_dow, mat_holiday, mat_holiday_soon))


def data_wrapper(args, mode='train'):
    v_inp = np.zeros([args.batch_size, args.lat_grid, args.long_grid, args.inp_ch], dtype=np.float32)  # batch, time_window * neighbors
    v_inp_pad = np.zeros([args.batch_size, 1, args.long_grid, args.inp_ch], dtype=np.float32)
    t_inp = np.zeros([args.batch_size, args.temporal_shape], dtype=np.float32)
    v_lab = np.zeros([args.batch_size, args.lat_grid, args.long_grid, 1],
                       dtype=np.float32)  # a value in TARGET_CELL after 10~30min
    v_lab_pad = np.zeros([args.batch_size, 1, args.long_grid, 1], dtype=np.float32)
    # gen = data_loader(df, lat_grid, long_grid, args, mode=mode)

    # destination input
    d_inp = np.zeros([args.batch_size, args.lat_grid, args.long_grid, args.inp_ch], dtype=np.float32)  # batch, time_window * neighbors
    # coord input
    coord_inp_padded = np.zeros([args.batch_size, args.lat_grid, args.long_grid, 2], dtype=np.float32)
    # coord_mat_pad = np.zeros([1, args.long_grid, 2], dtype=np.float32)
    coord_mat = np.zeros([args.lat_grid, args.long_grid, 2], dtype=np.float32)
    for i in range(args.lat_grid):
        for j in range(args.long_grid):
            coord_mat[i,j] = np.array([i,j], dtype=np.float32)
    coord_mat_padded = coord_mat
    # coord_mat_padded = [coord_mat_pad, coord_mat, coord_mat_pad]
    # coord_mat_padded = np.concatenate(coord_mat_padded, axis=0)
    for b in range(args.batch_size):
        coord_inp_padded[b] = coord_mat_padded

    if mode == 'train':
        dat = args.traindata
        rand_i = list(range(dat.shape[0]))
        random.shuffle(rand_i)
    elif mode == 'test':
        dat = args.testdata
        rand_i = list(range(dat.shape[0]))

    i = 0
    while True:
        for b in range(args.batch_size):
            v_inp[b] = dat[rand_i[i+b],:,:,0:args.inp_ch]
            d_inp[b] = dat[rand_i[i+b],:,:,args.inp_ch:2*args.inp_ch]
            # v_lab[b] = dat[rand_i[i+b],:,:,args.inp_ch:args.inp_ch+args.out_ch]
            v_lab[b] = dat[rand_i[i+b],:,:,2*args.inp_ch:2*args.inp_ch+args.out_ch]

            # t_inp[b] = onehotweek(crnt, 0)
            # t_inp[b] = temporal_encoding(crnt)
            t_inp[b] = temporal_encoding2(rand_i[i+b], args, mode=mode)

        # v_inp = (v_inp - v_stat[2]) / (v_stat[3] - v_stat[2])
        # v_lab = (v_lab - v_stat[2]) / (v_stat[3] - v_stat[2])

        # v_inp_padded = [v_inp_pad, v_inp, v_inp_pad]
        # v_inp_padded = np.concatenate(v_inp_padded, axis=1)
        v_inp_padded = v_inp
        # v_lab_padded = [v_lab_pad, v_lab, v_lab_pad]
        # v_lab_padded = np.concatenate(v_lab_padded, axis=1)
        v_lab_padded = v_lab

        # destination input
        # d_inp_padded = [v_inp_pad, d_inp, v_inp_pad]
        # d_inp_padded = np.concatenate(v_inp_padded, axis=1)
        d_inp_padded = d_inp

        list_inputs = [v_inp_padded, d_inp_padded, t_inp, coord_inp_padded]
        # list_outputs = [v_lab_padded, t_inp]

        # print(v_inp_padded.shape, d_inp_padded.shape, t_inp.shape, coord_inp_padded.shape, v_lab_padded.shape)
        yield list_inputs, v_lab_padded
        # yield v_inp_padded, list_outputs

        i += args.batch_size
        if i + args.batch_size >= len(rand_i):
            i = 0
            if mode == 'train':
                random.shuffle(rand_i)


############################################################################
### Metric
############################################################################

def rmse_vol(y_true, y_pred):
    y_true = y_true[:,:,:,0]# * (v_stat[3] - v_stat[2]) + v_stat[2]
    y_pred = y_pred[:,:,:,0]# * (v_stat[3] - v_stat[2]) + v_stat[2]
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

# only with normalization (do not allow minus values)
def smape_vol(y_true, y_pred):
    y_true = y_true[:,:,:,0]# * (v_stat[3] - v_stat[2]) + v_stat[2]
    y_pred = y_pred[:,:,:,0]# * (v_stat[3] - v_stat[2]) + v_stat[2]
    smape = K.mean(K.abs(y_true - y_pred) / (K.abs(y_true)+K.abs(y_pred)+K.epsilon()))
    return smape
