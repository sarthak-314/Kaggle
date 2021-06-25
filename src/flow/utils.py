"""
Utility functions for training / inference with tensorflow & keras models 
"""
from kaggle_datasets import KaggleDatasets
import tensorflow_addons as tfa
import tensorflow as tf

import src.comp.utils

def auto_select_accelerator():
    """
    Auto Select TPU / GPU / CPU
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def get_ranger(lr, min_lr=0): 
    radam = tfa.optimizers.RectifiedAdam( 
        learning_rate = lr,
        min_lr=min_lr,
        #weight_decay=0.001, # default is 0
        amsgrad = True,
        name = 'Ranger',
        #clipnorm = 10, # Not present by befault
    )
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    return ranger

AUTO = tf.data.experimental.AUTOTUNE
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16
GCS_DS_PATH = KaggleDatasets().get_gcs_path(src.comp.utils.COMP_NAME)





# ---------------------------------------- :KerasTrainerBase: ----------------------------------------
class KerasTrainerBase(metaclass=ABCMeta):
    """
    TODO:
     * See the original implmentation and try to extend the full functionality
     * A lot of improvements to be made. Just get it working and train the bert model quickly for now
     * Good enough for now, but you gotta make huge improvements in this and in PyTorch Model
     Note: The image input should be [0, 1]. Also see if preprocess_input is availible
    """
    def __init__(self, hp, train_ds, valid_ds, test_ds):
        self.hp = hp
        self.debug = True
        
        # Important params
        self.batch_size = hp.batch_size 
        self.lr = hp.lr 
        Color.blue(f'Training model with batch size {self.batch_size} and lr {self.lr}')
        
        # Data generator parameters
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.train_steps_per_epoch = len(train) // self.batch_size
        self.validation_steps = len(valid) // self.batch_size # Not sure about this one
        self.test_steps = len(test) // self.batch_size
        
        # Backbone
        self.backbone_name = hp.backbone_name
        self.backbone_source = hp.backbone_source
        
        # Training parameters
        self.max_epochs = hp.max_epochs
        self.patience = hp.patience
        
        # Model serialization parameters
        self.checkpoint_path = hp.checkpoint_path
        os.makedirs(hp.save_folder, exist_ok=True)
        parent_directory = os.path.dirname(self.checkpoint_path)
        os.makedirs(parent_directory, exist_ok=True)
        
        # Debugging / logging / misc 
        self.tensorboard_log_dir = hp.tensorboard_log_dir
        self.verbose = cfg.verbose
        #self.debug_params = hp.debug #:TODO: 

        # Model-specific member variables that I'll set and use later
        self.model = None
        self.debug_model = None
        self.history = None
        # Training-specific member variable that I'll set and use later
        self.best_epoch = -1
        
        
    # --------------------------------- :PUBLIC METHODS: ---------------------------------
    def compile_model(self): 
        Color.blue('Compiling the model')
        with cfg.tf.strategy.scope(): 
            self.model = self._build_model()
            
            # Compile params
            self.loss_fn = self.hp.loss_fn
            self.mode = self.hp.mode
            self.metrics = [metric for metric in self.hp.metrics]
            self.monitor = self.hp.monitor
            self.optimizer = KerasTrainerBase.get_ranger(self.lr) #TODO: See gradient clipping in optimizer
            self.gradient_clipping =  {'type': 'clip_by_norm', "value": 10} # Not used actually
            self.steps_per_execution = 32 if cfg.using_tpu else None 
            
            self.model.compile(
                optimizer = self.optimizer, 
                loss = self.loss_fn, 
                metrics = self.metrics, 
                steps_per_execution = self.steps_per_execution, 
            )
            
            session.cache.compiled_model = self.model
            Color.blue('Compiled model cached at session.cache.compiled_model')
    
    def train(self):
        Color.blue(f'Running training for {self.backbone_name}')
        # TODO: IMPORTANT - SET UP DEBUGGING
        callbacks = self._get_callbacks()
        print('Fitting the model')
        self.history = self.model.fit(
                self.train_ds, 
                batch_size=None,
                epochs=self.max_epochs, 
                verbose=2,
                callbacks=callbacks, 
                validation_data=self.valid_ds,
                shuffle=True,
                steps_per_epoch=self.train_steps_per_epoch,
                validation_steps=self.validation_steps, 
        )
        
        # After finishing training, save the best weights 
        Color.blue(f'Training completed. Saving best model')
        self.best_epoch = int(np.argmax(self.history.history[self.monitor]))+1
        self.__save_best_model()
        
        Color.blue('Inference on test dataset')
        # Inference on the test dataset
        self.infer(self.test_ds)

    # ------------------------------------------------------------------------------------
    
    # -------------------------------- :ABSTRACT METHODS: --------------------------------
    @abstractmethod
    def _build_model(self):
        """ Function to build the main model. Should return the TF model """
        pass
        
    def score_preds(self, dataset):
        """
        Takes a Dataset, and returns the output of evaluating the model 
        on the dataset along with labels
        """
        pass
    
    def _set_params_from_model(self):
        """
        Called after a model is loaded, use this to update member variables that contain model 
        parameter like max length. May be useful if you want to get train_ds compatibile with model
        """
        pass

    # ------------------------------------------------------------------------------------
    
    # ------------------------------- :PROTECTED METHODS: --------------------------------


    # ------------------------------------------------------------------------------------
    
    # --------------------------------- :PRIVATE METHODS: --------------------------------
    def __save_best_model(self):
        """ Copies the weights from the best epoch to a final weight file """
        epoch_weight_file = self.checkpoint_path.format(epoch=self.best_epoch)
        final_weight_file = f'{self.backbone_name}_weights.h5'
        copyfile(epoch_weight_file, final_weight_file)
        Color.blue(f'Saved the best model to {final_weight_file}')



    @staticmethod
 
    
    @staticmethod
    def get_save_locally(): 
        if cfg.using_tpu: 
            save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
            return save_locally
        return None
    
    @staticmethod
    def get_load_locally(): 
        """ Load models from TF HUB / KaggleDatasets to TPU """
        if cfg.tf.using_tpu: 
            load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
            return load_locally
        return None
    
    @staticmethod
    def get_xxyy(self, ds): 
        """
        For train / valid datasets that are tuples of (X, y)
        """
        for xx, yy in ds.take(1): 
            pass 
        return (xx, yy)

    
# ------------------------------------------- :TFDataBase: -------------------------------------------
class TFDataBase:
    """ 
    Base class to build the tensorflow data pipeline for training keras models 
    Use this class by overriding __init__, get_x and get_y methods
    
    TODO: 
        * Add support for transforms
        * Improve the dataset pipeline
    """
    def __init__(self, backbone_name, batch_size=32, max_len=128): 
        self.batch_size = batch_size
        self.x_col = 'title'
        self.y_col = 'label'
        
        self.backbone_name = backbone_name
        self.max_len = max_len
        
    def get_x(self, df, df_type): 
        titles = df[self.x_col].values
        input_ids = BertWrapper.encode_text(text_values, self.backbone_name, self.max_len)
        return input_ids

    def get_y(self, df, df_type): 
        labels = df[self.y_col].values
        labels = tf.cast(labels, tf.float32)
        return labels
    
    def get_train_ds(self): 
        x_train, y_train = self.get_numpy_arrays('x_train', 'y_train')
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.repeat().shuffle(1024)
        train_ds = train_ds.batch(self.batch_size, drop_remainder=True)
        train_ds = train_ds.prefetch(AUTO)
        return train_ds
    
    def get_valid_ds(self): 
        x_valid, y_valid = self.get_numpy_arrays('x_valid', 'y_valid')
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        valid_ds = valid_ds.repeat().shuffle(1024)
        valid_ds = valid_ds.batch(self.batch_size, drop_remainder=True)
        valid_ds = valid_ds.prefetch(AUTO)
        return valid_ds
        
    def get_test_ds(self): 
        x_test = self.get_x(test, df_type='test')
        test_ds = tf.data.Dataset.from_tensor_slices(x_test)
        test_ds = test_ds.batch(self.batch_size, drop_remainder=False)
        test_ds = test_ds.prefetch(AUTO)
        return test_ds
    
    def get_ds(self): 
        train_ds = self.get_train_ds()
        valid_ds = self.get_valid_ds()
        test_ds = self.get_test_ds()
        return train_ds, valid_ds, test_ds
    
    def get_all_numpy_arrays(self):
        x_train, y_train = self.get_x(train, 'train'), self.get_y(train, 'train')
        x_valid, y_valid = self.get_x(valid, 'valid'), self.get_y(valid, 'valid')
        x_test = self.get_x(test, 'test')

        x_tr, y_tr = self.get_x(tr, 'train'), self.get_y(tr, 'train')
        x_val, y_val = self.get_x(val, 'valid'), self.get_y(val, 'valid')
        numpy_arrays = DotDict(base_dict = dict(
            x_train = x_train, 
            y_train = y_train, 
            x_valid = x_valid, 
            y_valid = y_valid, 
            x_test = x_test, 
            x_tr = x_tr, 
            y_tr = y_tr, 
            x_val = x_val, 
            y_val = y_val, 
        ))
        return numpy_arrays
    
    def get_numpy_arrays(self, *args):
        numpy_arrays = self.get_all_numpy_arrays()
        res = []
        for np_array_name in args: 
            np_array = numpy_arrays[np_array_name]
            res.append(np_array)
        return res
    
    def get_xx_yy(self, ds): 
        for xx, yy in ds.take(1): 
            return xx, yy