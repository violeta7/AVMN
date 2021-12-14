# AVMN (Accumulative  Volume  Map  Network)

This is the implementation for my paper: "AVMN: Deep Learning for Forecasting Taxi Demands using Extremely Sparse Trip Records."
It is a deep learning model for taxi demand forecasting using novel AVM (Accumulative Volume Map) and Graph Neural Network using GN Block.
![ScreenShot](/.assets/avmn0.png)
This also includes my implementation of TGNet (for performance comparison). I have refered the original TGNet implementation a lot for my implemetation. The original TGNet sourcecode implemented by its author can be found here: https://github.com/LeeDoYup/TGGNet-keras

1. './datasets_expscale_both/' includes volume maps for NYC TLC from Jan. 2015 to Mar. 2015, which are downloaded from official NYC TLC homepage (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and preprocessed into normal volume map.
    - train (00:00 Jan. 1st ~ 00:00 Feb. 9th 2015)
    - validation (00:00 Feb. 9th ~ 00:00 Mar. 1st 2015)
    - test (00:00 Mar. 1st ~ 00:00 Apr. 1st 2015)
    - Please note that filename is 'nyc_taxi_expscale_100%_1000m_5min_train/test/test2_{date-range}.pkl', respectively.

2. 'data_gen8+9 final.ipynb' provides Sampling and creating training example processes. Set mode = ['train'/'test'/'test2'], sampl_rate = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001] and label_smoothing = False, then 'Restart and Run all.' Please note that the files for the 0.15% sampling case (sampl_rate=0.001) has 0% in their filename. Pre-created training examples are in in_and_out3_x.tar.gz file. Because file size cannot exceed 100MB, we only include training examples for [5%, 1%, 0.15%] sampling rate.
    ```python
    # parameters
    mode = 'test2'        # ['train', 'test', 'test2']
    sampl_rate = 0.001    # [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
    # label smoothing parameters
    label_smoothing = False #True     # [True, False]
    ```

3. Train. main_v8.py for AVMN and main_v9.py for TGNet. You can see training options using --help options. What you have to set is just --sampling_rate = [100%, 50%, 20%, 10%, 5%, 1% ,0.15%] options.
    ```
    $ python main_v8.py --sampling_rate 50%
    $ python main_v9.py --sampling_rate 50%
    ```

4. Evaluate. You can evaluate the model using --test option. Then it will save predicted result and labels in npy file.
    ```
    $ python main_v8.py --sampling_rate 50% --test
    $ python main_v9.py --sampling_rate 50% --test
    ```

5. evaluation.ipynb file provides performance measure calculation and drawing graphs. At the second cell of the notebook, please load proper result npy files what you just trained. You must remove padding only for AVMN.
    ```python
    # AVMN
    pred = np.load("outputs/y_pred0_nyc_taxi_presampled_accvnet1081_50%.npy")
    true = np.load("outputs/y_true0_nyc_taxi_presampled_accvnet1081_50%.npy")
    # only for AVMN (de-padding)
    pred = pred[:,1:-1,:,:]
    true = true[:,1:-1,:,:]
    ```

6. I CANNOT provide KORNATUS and MyTaxi dataset, because they are belong to each companies. Training code for KORNATUS and MyTaxi is not different except they have different number of cells in their volume map. Seoul has 15x20 cells, and Tashkent has 34x34 cells.

## Contact
Keeyoung Kim (kykim@sunykorea.ac.kr), Siho Han
