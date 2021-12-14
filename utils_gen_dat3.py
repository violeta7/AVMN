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

def preprocess(dataframe, sampling_rate, mode='train'):
    # Remove outliers
    long_lb, lat_lb, long_ub, lat_ub = -74.25909, 40.477399, -73.700181, 40.916178
    dataframe = dataframe[((dataframe['passenger_count'] > 0) &
                           (dataframe['passenger_count'] < 9))]
    dataframe = dataframe[((dataframe['pickup_latitude'] > lat_lb) &
                           (dataframe['pickup_latitude'] < lat_ub) &
                           (dataframe['pickup_longitude'] > long_lb) &
                           (dataframe['pickup_longitude'] < long_ub))]
    mu_trip_dist, std_trip_dist = dataframe['trip_distance'].mean(), dataframe['trip_distance'].std()
    dataframe = dataframe[((dataframe['trip_distance'] > mu_trip_dist - std_trip_dist) &
                           (dataframe['trip_distance'] < mu_trip_dist + std_trip_dist))]
    dataframe = dataframe.sort_values(by=['tpep_pickup_datetime']).reset_index(drop=True)

    # # Sampling
    # sample_i, sample_f = dataframe.iloc[0], dataframe.iloc[-1]
    # samples_split = dataframe[dataframe['tpep_pickup_datetime'] == pd.to_datetime('2016-01-01 00:00:00')] #pd.to_datetime('2016-01-10 00:00:00')]
    # idx_sample_split = random.randint(0, len(samples_split)-1)
    # sample_split = samples_split.iloc[idx_sample_split]
    # dataframe_sampled = dataframe.sample(frac=sampling_rate,
    #                                      replace=False,
    #                                      random_state=int(sampling_rate*100)).sort_values(by=['tpep_pickup_datetime'])
    # dataframe_sampled.iloc[0] = sample_i
    # dataframe_sampled.iloc[-2] = sample_split
    # dataframe_sampled.iloc[-1] = sample_f
    # dataframe_sampled = dataframe_sampled.sort_values(by=['tpep_pickup_datetime']).reset_index(drop=True).drop_duplicates()
    # return dataframe_sampled
    return dataframe


def generate_lat_long_grids(long_lb, lat_lb, long_ub, lat_ub, H_grid=10, W_grid=20):
    lat_grid = np.arange(lat_lb, lat_ub, (lat_ub - lat_lb)/(H_grid + 1))
    long_grid = np.arange(long_lb, long_ub, (long_ub - long_lb)/(W_grid + 1))
    if len(lat_grid) != (H_grid + 1):
        lat_grid = lat_grid[:H_grid+1]
    if len(long_grid) != (W_grid + 1):
        long_grid = long_grid[:W_grid+1]
    return lat_grid, long_grid


def get_distance(lat1, long1, lat2, long2, mode='Vincenty'):
    if mode == 'Vincenty':
        coord1, coord2 = (lat1, long1), (lat2, long2)
        return dist_Vincenty(coord1, coord2)    # km
    elif mode == 'haversine':
        # The math module contains a function named
        # radians which converts from degrees to radians.
        lat1 = radians(lat1)
        long1 = radians(long1)
        lat2 = radians(lat2)
        long2 = radians(long2)

        # Haversine formula
        dlong = long2 - long1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlong/2)**2
        c = 2*asin(sqrt(a))

        # Radius of earth in kilometers. Use 3956 for miles
        r = 6371

        # calculate the result
        return c*r


def gps_to_index(x, y, lat_grid, long_grid):
    i_lat = bisect(lat_grid, x)
    if i_lat == 0:
        i_lat = -1
    elif i_lat == len(lat_grid):
        i_lat = -1

    i_long = bisect(long_grid, y)
    if i_long == 0:
        i_long = -1
    elif i_long == len(long_grid):
        i_long = -1

    return i_lat-1, i_long-1

# 1%
# v_stat = (0.1554840313341598, 0.45280235129964386, 0, 8)   # mean, std, min, max
# dv_stat = (0.13275451385048814, 0.4115845530817363, 0 ,9) # mean, std, min, max
# tot_stat = (30.237357945247552, 10.813915013833853, 0, 55)   # mean, std, min, max
# 3%
# v_stat = (0.47519417606453485, 0.9772860954599156, 0.0, 16.0)   # mean, std, min, max
# dv_stat = (0.40232150782798287, 0.8674650639904204, 0.0, 19.0) # mean, std, min, max
# tot_stat = (131.5, 7.22072634081149, 117.0, 143.0)   # mean, std, min, max
# 3%(random)
# v_stat = (0.4306592414601419, 0.9081865598942919, 0.0, 14.0)   # mean, std, min, max
# dv_stat = (0.36320408235816054, 0.8057430683048686, 0.0, 15.0) # mean, std, min, max
# tot_stat = (115.44444444444444, 9.934352421003172, 97.0, 132.0)   # mean, std, min, max
# 10%
# v_stat = (1.1379913660200127, 1.9784357619643569, 0.0, 26.0)   # mean, std, min, max
# dv_stat = (0.9995196807416563, 1.7711651217561748, 0.0, 30.0) # mean, std, min, max
# tot_stat = (302.3888888888889, 12.837180760453125, 271.0, 327.0)   # mean, std, min, max
# 100%

def data_loader(df, lat_grid, long_grid, args, mode='train', datagen=False):
    global start_time, end_time, v_stat, dv_stat, tot_stat
    hours2look=args.hours2look
    sampling_time=args.sampling_time
    sampling_rate=args.sampling_rate

    if mode == 'train':
        start_time = pd.Timestamp('2015-01-01 00:00:00')
        end_time = pd.Timestamp('2015-02-09 00:00:00')
    elif mode == 'test':
        start_time = pd.Timestamp('2015-02-09 00:00:00')
        end_time = pd.Timestamp('2015-03-01 00:00:00')
        # start_time = pd.Timestamp('2015-03-01 00:00:00')
        # end_time = pd.Timestamp('2015-04-01 00:00:00')

    if df is not None:
        df = df.loc[df.tpep_pickup_datetime >= start_time][df.tpep_pickup_datetime < end_time]
        print("df len:", len(df))
        print("df range:", np.amin(df.tpep_pickup_datetime), "~", np.amax(df.tpep_pickup_datetime))

    # # specific location
    # # df_row.pickup_latitude: 40.7453839090909 ~ 40.75394663636362
    # # df_row.pickup_longitude: -73.98249238095232 ~ -73.97883661904756
    # df = df.loc[df.pickup_latitude >= 40.7453839090909][df.pickup_latitude < 40.75394663636362]
    # df = df.loc[df.pickup_longitude >= -73.98249238095232][df.pickup_longitude < -73.97883661904756]
    # plt.hist(df.tpep_pickup_datetime, bins=144)
    # sys.exit()

    b = 0
    cnt = None
    dataset = []
    stored = False
#     if os.path.exists('./dataset_' + mode + '.pkl'):
    # savefilename = f"{args.dataset}_{args.model_name}_sampled_{int(args.sampling_rate*100)}%_{args.stepsize}m_{args.sampling_time}min" if flag_sampling \
    #     else f"{args.dataset}_{args.stepsize}m_{args.sampling_time}min_{mode}.pkl"

    if args.start_date is not None and args.end_date is not None:
        savefilename = f"./datasets_expscale_both/{args.dataset}_expscale_{int(args.sampling_rate*100)}%_{args.stepsize}m_{args.sampling_time}min_{mode}2_{args.start_date}_{args.end_date}.pkl"
    else:
        savefilename = f"./datasets_expscale_both/{args.dataset}_expscale_{int(args.sampling_rate*100)}%_{args.stepsize}m_{args.sampling_time}min_{mode}.pkl"

    if os.path.exists(savefilename):
        print(f"{savefilename} DOES exist.")
        if datagen:
            return   # break the loop
        dataset = pickle.load(open(f"{savefilename}", 'rb'))
        stored = True
    elif False and os.path.exists(f"timeq_{int(args.sampling_rate*100)}%_{mode}.pkl"):
        now = time.time()

        timeq = pickle.load(open(f"timeq_{int(args.sampling_rate*100)}%_{mode}.pkl", 'rb'))
        timeidx = pickle.load(open(f"timeidx_{int(args.sampling_rate*100)}%_{mode}.pkl", 'rb'))
        dtimeq = pickle.load(open(f"dtimeq_{int(args.sampling_rate*100)}%_{mode}.pkl", 'rb'))
        dtimeidx = pickle.load(open(f"dtimeidx_{int(args.sampling_rate*100)}%_{mode}.pkl", 'rb'))
        dprtq = pickle.load(open(f"dprtq_{int(args.sampling_rate*100)}%_{mode}.pkl", 'rb'))
        cid_to_time = pickle.load(open(f"cid_to_time_{int(args.sampling_rate*100)}%_{mode}.pkl", 'rb'))

        print("sampled data loaded.")

        V_INP_WINDOW = args.lag
        # D_INP_WINDOW = 2 * V_INP_WINDOW  # why???
        # C_INP_WINDOW = args.c2c_window #25 #180 #25 #V_INP_WINDOW * 10  # 8%,10% --> 25 / 20% --> 50 / 100% --> 180
        DAY100 = 86400
        # vmat_inp = np.zeros([len(lat_grid)-1, len(long_grid)-1, V_INP_WINDOW], dtype=np.float32)
        # dvmat_inp = np.zeros([len(lat_grid)-1, len(long_grid)-1, D_INP_WINDOW], dtype=np.float32)
        # vtot_inp = np.zeros([V_INP_WINDOW], dtype=np.float32)
        # cmat_inp = np.log10(DAY100)*np.ones([len(lat_grid)-1, len(long_grid)-1, C_INP_WINDOW], dtype=np.float32)
        vmat_lab = np.zeros([len(lat_grid)-1, len(long_grid)-1, 1], dtype=np.float32)
        # cmat_lab = np.log10(DAY100)*np.ones([len(lat_grid)-1, len(long_grid)-1, 1], dtype=np.float32)
        dvmat_lab = np.zeros([len(lat_grid)-1, len(long_grid)-1, 1], dtype=np.float32)
    else:
        stored = False
        now = time.time()

        dprtq = dict()
        cid_to_time = dict()
        for i in range(len(lat_grid) - 1):
            for j in range(len(long_grid) - 1):
                dprtq[i, j] = []

        timeq, dtimeq = [], []
        for row_idx, df_row in df.iterrows():
            # random sampling
            if random.random() > sampling_rate:
                continue
            # uniform sampling
            # sampling_interval = int(1/sampling_rate)
            # if row_idx % sampling_interval != 0:
            #     continue

            t0 = df_row.tpep_pickup_datetime
            te = df_row.tpep_dropoff_datetime
            x = df_row.pickup_latitude
            y = df_row.pickup_longitude
            u = df_row.dropoff_latitude
            v = df_row.dropoff_longitude
            cid = row_idx

            ip, jp = gps_to_index(x, y, lat_grid, long_grid)
            dip, djp = gps_to_index(u, v, lat_grid, long_grid)
            if (ip < 0) or (jp < 0) or pd.isnull(t0):
                # print(f"Index out of bounds at {t0}, {ip}, {jp}, {x}, {y}")
                continue
            # elif (len(timeq) > 0) and (t0 == timeq[-1][0]):
            #     continue
            else:
                # print(f"Append: {t0}, {ip}, {jp}, {x}, {y}")
                # dprtq[ip, jp].append(pd.to_datetime(t0))
                dprtq[ip,jp].append(cid)
                cid_to_time[cid] = pd.to_datetime(t0)

                timeq.append([t0, ip, jp, cid])
                if (dip < 0) or (djp < 0) or pd.isnull(te):
                    pass
                else:
                    dtimeq.append([pd.to_datetime(te), dip, djp, cid])

        timeidx = [x[0] for x in timeq]
        print("timeq len:", len(timeq))
        dtimeq.sort(key=lambda elem: elem[0])
        dtimeidx = [x[0] for x in dtimeq]
        # print("dtimeq len:", len(dtimeq))

        V_INP_WINDOW = args.lag
        # D_INP_WINDOW = 2 * V_INP_WINDOW  # why???
        # C_INP_WINDOW = args.c2c_window #180 #25 #180 #25 #V_INP_WINDOW * 10  # 8%,10% --> 25 / 20% --> 50 / 100% --> 180
        DAY100 = 86400
        # vmat_inp = np.zeros([len(lat_grid)-1, len(long_grid)-1, V_INP_WINDOW], dtype=np.float32)
        # dvmat_inp = np.zeros([len(lat_grid)-1, len(long_grid)-1, D_INP_WINDOW], dtype=np.float32)
        # vtot_inp = np.zeros([V_INP_WINDOW], dtype=np.float32)
        # cmat_inp = np.log10(DAY100)*np.ones([len(lat_grid)-1, len(long_grid)-1, C_INP_WINDOW], dtype=np.float32)
        vmat_lab = np.zeros([len(lat_grid)-1, len(long_grid)-1, 1], dtype=np.float32)
        # cmat_lab = np.log10(DAY100)*np.ones([len(lat_grid)-1, len(long_grid)-1, 1], dtype=np.float32)
        dvmat_lab = np.zeros([len(lat_grid)-1, len(long_grid)-1, 1], dtype=np.float32)

        print("Queue preprocessing completed!")
        """
        pickle.dump(timeq, open(f"timeq_{int(args.sampling_rate*100)}%_{mode}.pkl", 'wb'))
        pickle.dump(timeidx, open(f"timeidx_{int(args.sampling_rate*100)}%_{mode}.pkl", 'wb'))
        pickle.dump(dtimeq, open(f"dtimeq_{int(args.sampling_rate*100)}%_{mode}.pkl", 'wb'))
        pickle.dump(dtimeidx, open(f"dtimeidx_{int(args.sampling_rate*100)}%_{mode}.pkl", 'wb'))
        pickle.dump(dprtq, open(f"dprtq_{int(args.sampling_rate*100)}%_{mode}.pkl", 'wb'))
        pickle.dump(cid_to_time, open(f"cid_to_time_{int(args.sampling_rate*100)}%_{mode}.pkl", 'wb'))

        print("sampled data pickled.")
        """

    print("cnt:", cnt)
    while True:
        if cnt is not None:
            if not stored:
                print(f"# of samples in the epoch: {cnt}")
                # pickle.dump(dataset, open(f"./datasets/{args.model_name}_{mode}.pkl", 'wb'))
                pickle.dump(dataset, open(f"{savefilename}", 'wb'))
                stored = True
                print(f"Dataset generation took {(time.time() - now)*(1/60)*(1/60):2f} hours.")

                all_labs = np.zeros([len(dataset), 10, 20, 2])
                # all_labs = np.zeros([len(dataset), 10, 20, 1])
                # all_vinps = np.zeros([len(dataset), 10, 20, V_INP_WINDOW])
                # all_dinps = np.zeros([len(dataset), 10, 20, D_INP_WINDOW])
                # all_totinps = np.zeros([len(dataset), V_INP_WINDOW])
                for i in range(len(dataset)):
                    # all_vinps[i] = dataset[i][0][0]
                    # all_dinps[i] = dataset[i][0][1]
                    #print(i, dataset[i].shape, dataset[i][1])
                    all_labs[i] = dataset[i][1]
                    # all_totinps[i] = dataset[i][0][3]

                print("vlab_stat:", np.mean(all_labs[:,:,:,0]), np.std(all_labs[:,:,:,0]), np.amin(all_labs[:,:,:,0]), np.amax(all_labs[:,:,:,0]))
                # print("clab_stat:", np.mean(all_labs[:,:,:,1]), np.std(all_labs[:,:,:,1]), np.amin(all_labs[:,:,:,1]), np.amax(all_labs[:,:,:,1]))
                # print("v_stat:", np.mean(all_vinps), np.std(all_vinps), np.amin(all_vinps), np.amax(all_vinps))
                # print("dv_stat:", np.mean(all_dinps), np.std(all_dinps), np.amin(all_dinps), np.amax(all_dinps))
                # print("tot_stat:", np.mean(all_totinps), np.std(all_totinps), np.amin(all_totinps), np.amax(all_totinps))            # if (mode == 'test') and (cnt != 0):
            #     print("Break data loader.")
            #     break
                if datagen:
                    return   # break the loop

        cnt = 0
        if stored:
            rand_i = [x for x in range(len(dataset))]
            print(f"len(rand_i): {len(rand_i)}")
            if mode == 'train':
                random.shuffle(rand_i)
                print(f"Randomly shuffled: {rand_i[0]} {rand_i[-1]}")
            for r_i in rand_i:
                yield dataset[r_i]
                cnt += 1
        else:
            print("Generating dataset...")
            prev_idx = 0

            # start_idx = bisect(timeidx, start_time + pd.Timedelta(minutes=V_INP_WINDOW*sampling_time))
            # end_idx = bisect(timeidx, end_time - pd.Timedelta(minutes=sampling_time))

            if args.start_date is not None:
                st_idx_time = max([start_time + pd.Timedelta(minutes=V_INP_WINDOW*sampling_time), pd.Timestamp(args.start_date)])
            if args.end_date is not None:
                ed_idx_time = min([end_time - pd.Timedelta(minutes=sampling_time), pd.Timestamp(args.end_date)])
            start_idx = bisect(timeidx, st_idx_time)
            end_idx = bisect(timeidx, ed_idx_time)

            print("start_time, end_time:", start_time, end_time)
            print("st_idx_time, ed_idx_time:", st_idx_time, ed_idx_time)
            print(f"Start datetime idx: {start_idx}, end datetime idx: {end_idx}")
            # sys.exit()

            skip_sample = 0
            # for t0, _, _, cid in timeq[start_idx:end_idx]:
            # for t0 in pd.date_range(start=start_time, end=end_time, freq=f"{sampling_time}min"):
            for t0 in pd.date_range(start=st_idx_time, end=ed_idx_time, freq=f"{sampling_time}min"):
                if sampling_rate != 1.0:
                    # bug below?
                    if sampling_rate > 0.1 and skip_sample % int(sampling_rate*10) != 0:
                        skip_sample += 1
                        continue

                # if (t0.day % 10 == 1) and (t0.hour == 0):
                #     print(t0)
                if t0.minute == 0:
                    print(t0)

                # vmat_inp.fill(0)
                # dvmat_inp.fill(0)
                # vtot_inp.fill(0)
                vmat_lab.fill(0)
                # vtot_lab = 0
                dvmat_lab.fill(0)

                idx = bisect(timeidx, t0)
                # if prev_idx == idx:
                #     continue

                crnt = t0
                start_dt = crnt - pd.Timedelta(minutes=V_INP_WINDOW*sampling_time)
                end_dt = crnt + pd.Timedelta(minutes=sampling_time)

                # Departure volume maps
                for l, tx in enumerate(pd.date_range(start=start_dt, end=end_dt, freq=f"{sampling_time}min")):
                    if l == 0:
                        prev_ti = bisect(timeidx, tx)
                        continue
                    ti = bisect(timeidx, tx)
                    # if l-1 < V_INP_WINDOW:
                    #     if tx > crnt:
                    #         print("Something went wrong while building volume maps!")
                    #     for x in timeq[prev_ti:ti]:
                    #         vmat_inp[x[1], x[2], l-1] += 1
                    #         # vtot_inp[l-1] += 1
                    # else:
                    #     for x in timeq[prev_ti:ti]:
                    #         vmat_lab[x[1], x[2], 0] += 1
                    #         # vtot_lab += 1
                    for x in timeq[prev_ti:ti]:
                        vmat_lab[x[1], x[2], 0] += 1
                    prev_ti = ti

                # vmat_inp = (vmat_inp - v_stat[2])/(v_stat[3] - v_stat[2])
                # vtot_inp = (vtot_inp - tot_stat[2])/(tot_stat[3] - tot_stat[2])
                # vmat_lab = (vmat_lab - v_stat[2])/(v_stat[3] - v_stat[2])
                # vtot_lab = (vtot_lab - tot_stat[2])/(tot_stat[3] - tot_stat[2])
                # print("vmat:", np.amin(vmat_inp), np.amax(vmat_inp))

                # # Destination volume maps
                # start_dt = crnt - pd.Timedelta(minutes=D_INP_WINDOW*sampling_time)
                # end_dt = crnt
                # for l, tx in enumerate(pd.date_range(start=start_dt, end=end_dt, freq=f"{sampling_time}min")):
                #     if l == 0:
                #         prev_ti = bisect(dtimeidx, tx)
                #         continue
                #     ti = bisect(dtimeidx, tx)
                #     for x in dtimeq[prev_ti:ti]:
                #         dvmat_inp[x[1], x[2], l-1] += 1
                #     prev_ti = ti
                # # dvmat_inp = (dvmat_inp - dv_stat[2])/(dv_stat[3] - dv_stat[2])
                for l, tx in enumerate(pd.date_range(start=start_dt, end=end_dt, freq=f"{sampling_time}min")):
                    if l == 0:
                        prev_ti = bisect(dtimeidx, tx)
                        continue
                    ti = bisect(dtimeidx, tx)
                    # if l-1 < V_INP_WINDOW:
                    #     if tx > crnt:
                    #         print("Something went wrong while building volume maps!")
                    #     for x in timeq[prev_ti:ti]:
                    #         vmat_inp[x[1], x[2], l-1] += 1
                    #         # vtot_inp[l-1] += 1
                    # else:
                    #     for x in timeq[prev_ti:ti]:
                    #         vmat_lab[x[1], x[2], 0] += 1
                    #         # vtot_lab += 1
                    for x in dtimeq[prev_ti:ti]:
                        dvmat_lab[x[1], x[2], 0] += 1
                    prev_ti = ti

                # EPS = 1e-1
                # # Call-to-call volume maps
                # for i in range(len(lat_grid) - 1):
                #     for j in range(len(long_grid) - 1):
                #         # ti0 = bisect(dprtq[i, j], crnt)
                #         ti0 = bisect(dprtq[i, j], cid)
                #         l = C_INP_WINDOW
                #         for ti in reversed(range(ti0 - C_INP_WINDOW, ti0)):
                #             l -= 1
                #             if ti < 0:
                #                 cmat_inp[i, j, l] = DAY100
                #             else:
                #                 # if l == C_INP_WINDOW - 1:
                #                 #     cmat_inp[i, j, l] = (crnt - dprtq[i, j][ti]).seconds + EPS
                #                 # else:
                #                 #     cmat_inp[i, j, l] = (dprtq[i, j][ti+1] - dprtq[i, j][ti]).seconds + EPS
                #                 if l == C_INP_WINDOW - 1:
                #                     cmat_inp[i,j,l] = (crnt-cid_to_time[dprtq[i,j][ti]]).seconds+EPS
                #                 else:
                #                     cmat_inp[i,j,l] = (cid_to_time[dprtq[i,j][ti+1]]-cid_to_time[dprtq[i,j][ti]]).seconds+EPS
                # cmat_inp[cmat_inp > DAY100] = DAY100
                # cmat_inp = np.log10(cmat_inp)
                # # print("c2c:", np.amin(cmat_inp), np.amax(cmat_inp))
                #
                # # Call-to-call labels
                # for i in range(len(lat_grid) - 1):
                #     for j in range(len(long_grid) - 1):
                #         ti = bisect(dprtq[i, j], cid)
                #         # Update cmat_lab
                #         if ti == len(dprtq[i, j]):
                #             cmat_lab[i, j, 0] = DAY100
                #         else:
                #             # cmat_lab[i, j, 0] = (dprtq[i, j][ti] - crnt).seconds + EPS
                #             cmat_lab[i,j,0] = (cid_to_time[dprtq[i,j][ti]]-crnt).seconds+EPS
                # cmat_lab[cmat_lab > DAY100] = DAY100
                # cmat_lab = np.log10(cmat_lab)

                # print("here")

                # inputs = [np.array(vmat_inp), np.array(dvmat_inp), np.array(cmat_inp), vtot_inp]
                # inputs = np.array(cmat_inp)
                # labels = np.array(vmat_lab)
                concats = [np.array(vmat_lab), np.array(dvmat_lab)]
                # concats = [vmat_lab, cmat_lab] # vtot_lab?
                labels = np.concatenate(concats, axis=-1)
                # labels = [np.array(vmat_lab), np.array(cmat_lab)]

                # print("before yield.")
                # yield inputs, labels, crnt
                yield labels, crnt
                if not stored:
                    # dataset.append([inputs, labels, crnt])
                    dataset.append([labels, crnt])

                prev_idx = idx
                cnt += 1
                skip_sample += 1
                # print("sample processed.")


# week of the day --> onehot function
# st: starting times (in 10min), pandas.datetime
# index: dataset index
def onehotweek(st, index, to_int=False):
    a_day = 6*24
    a_week = a_day*7
    weekday = st.dayofweek
    hour = st.hour
    minute = st.minute

    #print(weekday, hour, minute, index)
    in_10_min = int((weekday*a_day + hour*6 + minute//10) + index) % a_week

    if to_int:
        return in_10_min

    onehot = np.zeros([a_week], dtype=np.uint8)
    onehot[in_10_min] = 1   # index should be int.

    return np.array(onehot)


def data_wrapper(df, lat_grid, long_grid, args, mode='train', datagen=False):
    batch_size=args.batch_size
    hours2look=args.hours2look
    sampling_time=args.sampling_time
    sampling_rate=args.sampling_rate
    V_INP_WINDOW = args.lag #hours2look*60//sampling_time
    D_INP_WINDOW = 2 * V_INP_WINDOW
    C_INP_WINDOW = args.c2c_window #180 #25 #V_INP_WINDOW * 10
    v_inp = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, V_INP_WINDOW], dtype=np.float32)  # batch, time_window * neighbors
    v_inp_pad = np.zeros([batch_size, 1, len(long_grid)-1, V_INP_WINDOW], dtype=np.float32)
    d_inp = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, D_INP_WINDOW], dtype=np.float32)  # batch, time_window * neighbors
    d_inp_pad = np.zeros([batch_size, 1, len(long_grid)-1, D_INP_WINDOW], dtype=np.float32)
    c_inp = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, C_INP_WINDOW], dtype=np.float32)  # batch, time_window * neighbors
    # c_inp = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, 180], dtype=np.float32)  # batch, time_window * neighbors
    c_inp_pad = np.zeros([batch_size, 1, len(long_grid)-1, C_INP_WINDOW], dtype=np.float32)
#     b_inp = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, C_INP_WINDOW],
#                      dtype=np.float32)  # batch, time_window * neighbors
    n_samples_per_day = int((1/sampling_time)*60*24)*7    # sample/10min * 60min/hr * 24hr/day * 7days = 1008samples
    vtot_inp = np.zeros([batch_size, V_INP_WINDOW], dtype=np.float32)
    t_inp = np.zeros([batch_size, n_samples_per_day], dtype=np.float32)
    mat_lab = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, 2],
                       dtype=np.float32)  # a value in TARGET_CELL after 10~30min
    # vmat_lab = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, 1],
    #                    dtype=np.float32)  # a value in TARGET_CELL after 10~30min
    # cmat_lab = np.zeros([batch_size, len(lat_grid)-1, len(long_grid)-1, 1],
    #                    dtype=np.float32)  # a value in TARGET_CELL after 10~30min
    mat_lab_pad = np.zeros([batch_size, 1, len(long_grid)-1, 2], dtype=np.float32)
    # print(v_inp.shape, d_inp.shape, c_inp.shape)
    # print(f"Mode: {mode}")
    gen = data_loader(df, lat_grid, long_grid, args, mode=mode, datagen=datagen)

    while True:
        for b in range(batch_size):
            # mat_inputs, mat_lab[b], crnt = next(gen)
            mat_lab[b], crnt = next(gen)

            if not datagen:
                v_inp[b], d_inp[b], c_inp[b], vtot_inp[b] = mat_inputs
                # c_inp[b], mat_labels, crnt = next(gen)
                # vmat_lab[b], cmat_lab[b] = mat_labels
                t_inp[b] = onehotweek(crnt, 0)

        if not datagen:
            # normalization
            v_inp = (v_inp - v_stat[2]) / (v_stat[3] - v_stat[2])
            d_inp = (d_inp - dv_stat[2]) / (dv_stat[3] - dv_stat[2])
            vtot_inp = (vtot_inp - tot_stat[2]) / (tot_stat[3] - tot_stat[2])
            mat_lab[:,:,:,0] = (mat_lab[:,:,:,0] - v_stat[2]) / (v_stat[3] - v_stat[2])

            v_inp_padded = [v_inp_pad, v_inp, v_inp_pad]
            v_inp_padded = np.concatenate(v_inp_padded, axis=1)
            d_inp_padded = [d_inp_pad, d_inp, d_inp_pad]
            d_inp_padded = np.concatenate(d_inp_padded, axis=1)
            c_inp_padded = [c_inp_pad, c_inp, c_inp_pad]
            # c_inp_padded = [c_inp_pad, c_inp[:,:,:,-C_INP_WINDOW:], c_inp_pad]
            c_inp_padded = np.concatenate(c_inp_padded, axis=1)
            mat_lab_padded = [mat_lab_pad, mat_lab, mat_lab_pad]
            mat_lab_padded = np.concatenate(mat_lab_padded, axis=1)
            # vmat_lab_padded = [mat_lab_pad, vmat_lab, mat_lab_pad]
            # vmat_lab_padded = np.concatenate(vmat_lab_padded, axis=1)
            # cmat_lab_padded = [mat_lab_pad, cmat_lab, mat_lab_pad]
            # cmat_lab_padded = np.concatenate(cmat_lab_padded, axis=1)
            # print(c_inp_padded.shape)

            inp_padded = [v_inp_padded, d_inp_padded, c_inp_padded]
            inp_padded = np.concatenate(inp_padded, axis=-1)
            # inp_padded = c_inp_padded

            # list_inputs = [v_inp_padded, d_inp_padded, c_inp_padded, vtot_inp, t_inp]
            # print(inp_padded.shape, vtot_inp.shape, t_inp.shape)
            list_inputs = [inp_padded, vtot_inp, t_inp]
            # print(t_inp.shape)
            # list_inputs = [c_inp, t_inp] #[v_inp, d_inp, c_inp, t_inp]
            # list_inputs = [c_inp_padded, t_inp] #[v_inp, d_inp, c_inp, t_inp]
            # list_outputs = [vmat_lab, cmat_lab]
            # list_outputs = [vmat_lab_padded, cmat_lab_padded]
            # if b == 0:
            #     print("data_wrap:", np.mean(c_inp), np.mean(v_inp))
            # print(mat_lab_padded.shape)
            yield list_inputs, mat_lab_padded
            # yield list_inputs, list_outputs #mat_lab


############################################################################
### Metric
############################################################################

def rmse_vol(y_true, y_pred):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def rmse_c2c(y_true, y_pred):
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

# only with normalization (do not allow minus values)
def smape_vol(y_true, y_pred):
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    smape = K.mean(K.abs(y_true - y_pred) / (K.abs(y_true)+K.abs(y_pred)+K.epsilon()))
    return smape

def smape_c2c(y_true, y_pred):
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    smape = K.mean(K.abs(y_true - y_pred) / (K.abs(y_true)+K.abs(y_pred)+K.epsilon()))
    return smape

def rmse_tot(y_true, y_pred):
    y_true = K.sum(y_true[:,:,:,0], axis=(1,2))
    y_pred = K.sum(y_pred[:,:,:,0], axis=(1,2))
    rmse = K.sqrt(K.mean(K.square(y_true - y_pred)))
    return rmse

def smape_tot(y_true, y_pred):
    y_true = K.sum(y_true[:,:,:,0], axis=(1,2))
    y_pred = K.sum(y_pred[:,:,:,0], axis=(1,2))
    smape = K.mean(K.abs(y_true - y_pred) / (K.abs(y_true)+K.abs(y_pred)+K.epsilon()))
    return smape

def custom_loss(y_true, y_pred):
    loss = rmse_c2c(y_true, y_pred) + 5.0 * rmse_vol(y_true, y_pred) + 1e-5 * rmse_tot(y_true, y_pred)
    return loss
