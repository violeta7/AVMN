class BaseModel():
    def __init__(self, args):
    # def __init__(self, call2call_shape, temporal_shape, args=None):
        self.args = args

        # self.lr = args.lr
        # self.decay = args.decay
        # self.epoch = args.epochs
        # self.batch_size = args.batch_size
        # self.model_name = args.model_name
        # # self.scale = args.scale
        # self.input_shape = args.model_input_shape
        # self.temporal_shape = args.temporal_shape
        self.model = self.build_model(args)

    def build_model(self, args):
    # def build_model(self, call2call_shape, temporal_shape, args):
        pass

    # def make_callbacks(self):
    #     patience = self.args.patience
    #     es = self.args.es
    #     self.tf_graph_dir = './tfgraph/' + self.model_name + '/'
    #     self.tb_hist = keras.callbacks.TensorBoard(log_dir=self.tf_graph_dir, histogram_freq=0, write_graph=True,
    #                                                write_images=True)
    #     self.early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience,
    #                                                         verbose=0, mode=es)
    #
    #     self.check_point = keras.callbacks.ModelCheckpoint(
    #         self.tf_graph_dir + 'wegihts.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=2,
    #         save_best_only=True)
    #
    # def load_model(self, save_model_name, save_dir='./model_saved', best=False):
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     if not best:
    #         save_name = save_model_name + '.h5'
    #     else:
    #         save_name = save_model_name + '_best.h5'
    #     model_address = os.path.join(save_dir, save_name)
    #     self._model.load_weights(model_address)
    #     print(model_address, ' is loaded')
    #
    # def save_model(self, save_model_name, save_dir='./model_saved', best=False):
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     if not best:
    #         save_name = save_model_name + '.h5'
    #     else:
    #         save_name = save_model_name + '_best.h5'
    #     model_address = os.path.join(save_dir, save_name)
    #     self._model.save(model_address)

#     def train(self, x_train=None, y_train=None):
    def train(self, args):
        pass

    def test(self, args):
        pass
