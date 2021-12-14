from utils_v9 import *
from model_v8 import *

# import output
import os
import math

import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, AveragePooling2D, Conv2DTranspose, Activation, Concatenate, MaxPooling2D
from tensorflow.keras.layers import concatenate, BatchNormalization, Dropout, Add, RepeatVector, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
# from keras.utils.training_utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    "rmse_vol": rmse_vol,
    "smape_vol": smape_vol,
    # "masked_smape_vol": masked_smape_vol,
    # "temporal_loss": temporal_loss,
    # "custom_loss": custom_loss,
})

def gn_block(x_in,
             channels=64,
             kernel_size=(3, 3),
             strides=(1, 1),
             padding='same',
             activation='relu',
             dropout=None,
             regularizer=0.01):
    print("gn_block:", x_in)
    net = AveragePooling2D(pool_size=kernel_size,
                           strides=strides,
                           padding=padding)(x_in)
    net = Conv2D(channels,
                 kernel_size=(1, 1),
                 strides=strides,
                 activation='linear',
                 padding=padding,
                 kernel_regularizer=regularizers.l1(regularizer))(net)
    net_sf = Conv2D(channels,
                    kernel_size=(1, 1),
                    strides=strides,
                    activation='linear',
                    padding=padding,
                    kernel_regularizer=regularizers.l1(regularizer))(x_in)
    net = Add()([net, net_sf])
    net = concatenate([x_in, net])
    net = Conv2D(channels,
                 kernel_size=(1, 1),
                 strides=strides,
                 activation=activation,
                 padding=padding,
                 kernel_regularizer=regularizers.l1(regularizer))(net)
    if dropout == None:
        net = BatchNormalization()(net)
        return net
    else:
        net = Dropout(dropout)(net)
        return net

def deconv_block(x_in,
                 channels=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu',
                 dropout=None,
                 regularizer=0.01):
    net = Conv2DTranspose(channels,
                          kernel_size=kernel_size,
                          strides=strides,
                          activation=activation,
                          padding=padding,
                          kernel_regularizer=regularizers.l1(regularizer))(x_in)
    if dropout == None:
        net = BatchNormalization()(net)
        return net
    else:
        net = Dropout(dropout)(net)
        return net

class TGNet_v9(BaseModel):
    def build_model(self, args):
        input_departure = Input(shape=args.model_input_shape)
        input_destination = Input(shape=args.model_input_shape)
        input_temporal = Input(shape=args.temporal_shape)
        input_coord = Input(shape=args.coord_shape)
        inputs = [input_departure, input_destination, input_temporal, input_coord]

        # Time-guided embedding
        H, W = args.model_input_shape[:2]
        print(args.temporal_shape, "-->", args.f_temporal)
        net_temporal = Dense(units=args.f_temporal*4, activation='relu')(input_temporal)
        net_temporal = Dense(units=args.f_temporal, activation='relu')(net_temporal)
        net_temporal = RepeatVector(H*W)(net_temporal)
        net_temporal = Reshape((H, W, args.f_temporal))(net_temporal)

        # Coordination embedding
        net_coord = Conv2D(filters=args.f_coord,
                           kernel_size=(1,1),
                           padding='same')(input_coord)
        net_coord = Activation('relu')(net_coord)

        # Drop-off embedding
        net_destination = gn_block(x_in=input_destination,
                                   channels=args.f_gnblock*2,
                                   dropout=args.p_dropout,
                                   regularizer=args.reg)
        net_destination = gn_block(x_in=net_destination,
                                   channels=args.f_gnblock*2,
                                   dropout=args.p_dropout,
                                   regularizer=args.reg)

        # Densely connected U-Net (volume maps)
        # Down-sampling
        net0 = concatenate([input_departure, net_temporal], axis=-1)
        net1 = gn_block(x_in=net0,
                        channels=args.f_gnblock,
                        dropout=args.p_dropout,
                        regularizer=args.reg)
        print("net1:", net1)
        net11 = AveragePooling2D(pool_size=(2, 2))(net1)
        net2 = gn_block(x_in=net11,
                        channels=args.f_gnblock*2,
                        dropout=args.p_dropout,
                        regularizer=args.reg)
        net3 = gn_block(x_in=net2,
                        channels=args.f_gnblock*2,
                        dropout=args.p_dropout,
                        regularizer=args.reg)
        net33 = concatenate([net2, net3])
        net4 = gn_block(x_in=net33,
                        channels=args.f_gnblock*2,
                        dropout=args.p_dropout,
                        regularizer=args.reg)

        # Up-sampling
        unet5 = concatenate([net2, net3, net4])
        unet5 = deconv_block(x_in=unet5,
                             channels=args.f_gnblock*4,
                             kernel_size=(2, 2),
                             strides=(2, 2),
                             dropout=args.p_dropout,
                             regularizer=args.reg)
        unet6 = concatenate([unet5, net1])
        unet6 = deconv_block(x_in=unet6,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            channels=args.f_gnblock*4,
                            dropout=args.p_dropout,
                            regularizer=args.reg)

        unet7 = concatenate([unet6, input_departure, net_destination, input_destination, net_temporal, net_coord])

        # Position-wise regression
        unet7 = gn_block(x_in=unet7,
                        channels=args.f_gnblock*4,
                        kernel_size=(1, 1),
                        dropout=args.p_dropout,
                        regularizer=args.reg)
        # unet7 = Conv2D(filters=1024,
        #                 kernel_size=(1,1),
        #                 padding='same',
        #                 activation='relu',
        #                 kernel_regularizer=regularizers.l1(args.reg),
        #                 name='fc0')(unet7)
        # if dropout is not None:
        #     unet7 = Dropout(args.p_dropout)(unet7)
        # unet7 = Conv2D(filters=512,
        #                 kernel_size=(1,1),
        #                 padding='same',
        #                 activation='relu',
        #                 kernel_regularizer=regularizers.l1(args.reg),
        #                 name='fc1')(unet7)
        # if dropout is not None:
        #     unet7 = Dropout(args.p_dropout)(unet7)
        # Output
        output = Conv2D(filters=1,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_regularizer=regularizers.l1(args.reg),
                         name='output')(unet7)
        # output = Activation('relu', name='output')(output)

        model = Model(inputs=inputs, outputs=output)
#         print(type(model))
        print(model.summary())
        return model

    def train(self, args):
        gen = data_wrapper(args=args, mode='train')
        gen_v = data_wrapper(args=args, mode='test')

        losses = {
        	"output": rmse_vol,
        	# "output_temporal": "categorical_crossentropy",
        }
        lossWeights = {
            "output": 1.0,
            # "output_temporal": 0.1
        }

        self.model.compile(optimizer = Adam(lr=args.lr, decay=args.lr/args.epochs),
                        loss = losses,
                        loss_weights = lossWeights,
                        metrics = {
                            "output": smape_vol,
                            # "output": rmse_vol,
                            # "output": masked_smape_vol,
                        })

        es_callback = EarlyStopping(monitor='val_loss',
                                    patience=args.epochs//10,
                                    min_delta=0,
                                    mode='min',
                                    restore_best_weights=True)
        ckpt_callback = ModelCheckpoint(filepath=os.path.join(args.model_dir, args.model_name),
                                        monitor='val_loss', #'val_rmse_c2c', #'val_rmse_vol',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min')
        train_steps = args.train_batches
        test_steps  = args.test_batches
        self.model.fit(gen,
                       epochs=args.epochs,
                       steps_per_epoch=train_steps,
                       # verbose=2,
                       validation_data=gen_v,
                       validation_steps=test_steps,
                       # use_multiprocessing=True,
                       callbacks=[es_callback, ckpt_callback])

    def test(self, args):
        gen_v = data_wrapper(args=args, mode='test')
        # self.model.load_weights(os.path.join(args.model_dir, args.model_name))
        # print(f"{args.model_name} loaded!\n")
        self.model = load_model(os.path.join(args.model_dir, args.model_name))
        print("model loaded:", os.path.join(args.model_dir, args.model_name))

        test_steps  = args.test_batches

        print("Evaluating...")
        metrics = self.model.evaluate_generator(gen_v,
                                                steps=test_steps,
                                                verbose=True)

        gen_v = data_wrapper(args=args, mode='test')
        print("Predicting...")
        # y_pred = self.model.predict_generator(gen_v,
        y_pred = self.model.predict(gen_v,
                                              steps=test_steps,
                                              verbose=True)
        # print(len(y_pred), y_pred[0].shape, y_pred[1].shape)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        np.save(os.path.join(args.output_dir, 'y_pred_'+args.model_name+'_'+args.sampling_rate+'.npy'), y_pred)
        # np.save(os.path.join(args.output_dir, 'y_pred0_'+args.model_name+'_'+args.sampling_rate+'.npy'), y_pred[0])
        # np.save(os.path.join(args.output_dir, 'y_pred1_'+args.model_name+'_'+args.sampling_rate+'.npy'), y_pred[1])
        print(f"Predictions saved in {os.path.join(args.output_dir, 'y_pred[0,1]_'+args.model_name+'_'+args.sampling_rate+'.npy')}")

        # Save labels
        gen_v = data_wrapper(args=args, mode='test')
        print("Saving labels...")
        y_true = np.zeros([args.batch_size*test_steps, args.lat_grid, args.long_grid, 1], dtype=np.float32)
        # y_true0 = np.zeros([args.batch_size*test_steps, args.lat_grid, args.long_grid, 1], dtype=np.float32)
        # y_true1 = np.zeros([args.batch_size*test_steps, args.temporal_shape], dtype=np.float32)
        for i in range(test_steps):
            _, y_i = next(gen_v)
            # print(len(y_i), y_i[0].shape, y_i[1].shape)
            y_true[i*args.batch_size:(i+1)*args.batch_size] = y_i
            # y_true0[i*args.batch_size:(i+1)*args.batch_size], y_true1[i*args.batch_size:(i+1)*args.batch_size] = y_i
        np.save(os.path.join(args.output_dir, 'y_true_'+args.model_name+'_'+args.sampling_rate+'.npy'), y_true)
        # np.save(os.path.join(args.output_dir, 'y_true0_'+args.model_name+'_'+args.sampling_rate+'.npy'), y_true0)
        # np.save(os.path.join(args.output_dir, 'y_true1_'+args.model_name+'_'+args.sampling_rate+'.npy'), y_true1)
        print(f"Labels saved in {os.path.join(args.output_dir, 'y_true[0,1]_'+args.model_name+'_'+args.sampling_rate+'.npy')}")
        return
