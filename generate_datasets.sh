#!/bin/bash
# Jan.
prev_day='2015-01-01'
DAY=$(seq 2 31)
for i in $DAY
do
    if [ $i -lt 10 ]
    then
        today='2015-01-0'$i
    else
        today='2015-01-'$i
    fi
    python generate_dataset.py --traindata --start_date $(echo $prev_day) --end_date $(echo $today) --sampling_rate 1.0 --c2c_window 180
    prev_day=$today
done

# Feb.
DAY=$(seq 1 28)
for i in $DAY
do
    if [ $i -lt 10 ]
    then
        today='2015-02-0'$i
        mode='traindata'
    else
        today='2015-02-'$i
        mode='testdata'
    fi
    python generate_dataset.py --$(echo $mode) --start_date $(echo $prev_day) --end_date $(echo $today) --sampling_rate 1.0 --c2c_window 180
    prev_day=$today
done

# Mar.
python generate_dataset.py --testdata --start_date $(echo $prev_day) --end_date 2015-03-01 --sampling_rate 1.0 --c2c_window 180
