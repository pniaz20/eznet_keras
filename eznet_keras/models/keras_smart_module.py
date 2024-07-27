
if __package__=="eznet_keras.models":
    from ..utils import *
else:
    import os, sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from utils import *

class KerasSmartModel(tf.keras.models.Model):
    
    sample_hparams = {
        'model_name': 'KerasSmartModel',
        'l2_reg': 0.0001,
        'l1_reg': None,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': 0.1,
        'early_stopping_patience_epochs': 10,
        'learning_rate': 0.0001,
        'exponential_decay_rate': 0.99,
        'loss_function': 'categorical_crossentropy',
        'loss_function_params': None,
        'metrics':['accuracy'],
        'metrics_params': None,
        'optimizer': 'Adam',
        'optimizer_params': None,
        'checkpoint_path':None,
        'early_stopping_monitor':'loss',
        'early_stopping_mode':'min',
        'early_stopping_value':1.0e-6,
        'other_callbacks': None,
        'custom_schedule': None
    }
    
    def __init__(self, hparams:dict=None):
        """
        Base class for smart, trainable pytorch modules. All hyperparameters are contained within the `hparams`
        dictionary. Some training-related hyperparameters are common across almost all kinds of PyTorch modules,
        which can be overloaded by the child class. The module includes functions for training, evaluation, and
        prediction. These functions cane be modified or overloaded by any child subclass.

        ### Usage

        `net = KerasSmartModel(hparams)` where `hparams` is dictionary of hyperparameters containing the following:
            - `model_name` (str): Name of the model.
            - `batch_size` (int): Minibatch size, the expected input size of the network.
            - `learning_rate` (float): Initial learning rate of training.
            - `exponential_decay_rate` (float): Exponential decay rate for learning rate, if any.
            - `optimizer` (str): Optimizer. Examples are "Adam", "SGD", "RMSprop", "Adagrad", etc. The name of any Keras optimizer can be used.
                This can also be a custom optimizer class (not instance), in which case `optimizer_params` can be specified for its constructor kwargs.
            - `optimizer_params` (dict): Additional parameters of the optimizer constructor, if any.
            - `epochs` (int): Maximum number of epochs for training.
            - `early_stopping_patience_epochs` (int): Epochs to tolerate unimproved (val) loss, before early stopping (i.e., patience).
            - `l2_reg` (float): L2 regularization parameter. Defaults to None.
            - `l1_reg` (float): L1 regularization parameter. Defaults to None.
            - `loss_function` (str): Loss function. It can be a lower-case name such as "mse", "binary_crossentropy", "categorical_crossentropy", etc.,
                the name of a `tf.keras.losses` class such as `BinaryCrossentropy`, `CategoricalCrossentropy`, `MeanSquaredError', etc., or a valid loss class (not instance).
            - `loss_function_params` (dict): Additional kwargs parameters of the loss function constructor, if any. Ignored if the loss function is a lower-case name string.
            - `metrics` (list): list of metrics for Keras compilation. Each member of the list can be a lower-case metric name such as "mse" or "accuracy", the name of a 
                `tf.keras.metrics` class such as `Accuracy`, `MeanSquaredError`, etc., or a valid metric class (not instance).
            - `metrics_params` (list): (list of) additional kwargs parameters of the metrics constructors, if any. Ignored for every metric that is a lower-case name string.
                If this entry is a single dicitonary rather than a list, it will be broadcast to all metrics in the metrics list.
            - `checkpoint_path` (str): Path to the directory where checkpoints will be saved at every epoch.
            - `early_stopping_monitor` (str): Monitor whose critical value will cause early stopping. Default is 'loss', but 'val_loss' is typically used.
            - `early_stopping_mode` (str): Mode of the parameter whose critical value will be used for early stopping. Deafults to 'min' for any error. 'max' is for accuracy, etc.
            - `early_stopping_value` (float): Value of the monitor at which point training will stop becasue the critical value has been reached.
            - `other_callbacks` (list): List of other callbacks to be used during training. Defaults to None.
            - `custom_schedule` (schedule): Custom learning rate schedule inheriting from `tf.keras.optimizers.schedules.LearningRateSchedule`. Defaults to None.

        ### Returns

        Returns a `tf.keras.models.Model` object that can be trained and used accordingly.
        Run `net.summary()` afterwards to see what you have inside the network.
        
        ### Notes:
        
        - `self.batch_input_shape` attribute must be set in the `__init__` method.
        - `self.batch_output_shape` attribute must be set in the `__init__` method.
        - `self._callbacks` is a list of callbacks sent to the training method of Keras. It already includes garbage collection.
        - `self._es` is an EarlyStopping instance of Keras, if specified, otherwise None.
        - `self._chkpt` is the hyperparameter `checkpoint_path` from the input dictionary, if it exists, otherwise None.
        - `self._chk` is a ModelCheckpoint instance sent to the training function of Keras, if specified, otherwise None.
        - `self._es_crit` is an EarlyStopAtCriteria instance that stops the training if 'val_loss' for instance, reaches a certain value.
        - `self.net` **MUST** always exist so that Keras2Cpp can serialize the layers that it understands.
          `self.net` is always used as the model within this module, that contains the network itself. It is typically a Sequential or Functional model built in the `__init__` 
          method.
        """
        super(KerasSmartModel, self).__init__()
        if not hparams: hparams = self.sample_hparams
        self.hparams = hparams
        self._batch_size = int(hparams["batch_size"]) if hparams.get("batch_size") else 32
        self._loss_function = hparams.get("loss_function")
        self._loss_function_params = hparams.get("loss_function_params")
        self._metrics = hparams.get("metrics")
        self._metrics_params = hparams.get("metrics_params")
        self._optimizer = hparams.get("optimizer")
        self._optimizer_params = hparams.get("optimizer_params")
        self._early_stopping_patience_epochs = hparams.get("early_stopping_patience_epochs")
        self._learning_rate = hparams.get("learning_rate")
        self._exponential_decay_rate = hparams.get("exponential_decay_rate")
        self._validation_data = hparams.get("validation_data")
        self._epochs = hparams.get("epochs")
        self._l2_reg = hparams.get("l2_reg") if hparams.get("l2_reg") else None
        self._l1_reg = hparams.get("l1_reg") if hparams.get("l1_reg") else None
        self.history = None
        self.batch_input_shape = (self._batch_size, 1)
        self.batch_output_shape = (self._batch_size, 1)
        self._callbacks = [GarbageCollectionCallback()]
        self._es = tf.keras.callbacks.EarlyStopping(monitor='val_loss' if hparams.get('validation_data') is not None else 'loss', 
                                                    mode="min", patience=self._early_stopping_patience_epochs) if self._early_stopping_patience_epochs else None
        if self._es:
            self._callbacks.append(self._es)
        self._chkpt = hparams.get("checkpoint_path")
        if self._chkpt:
            self._chk = tf.keras.callbacks.ModelCheckpoint(self._chkpt, monitor=('val_loss' if hparams.get('validation_data') is not None else 'loss'), 
                                                           verbose=0, save_best_only=True, mode='min')
        else:
            self._chk = None
        if self._chk:
            self._callbacks.append(self._chk)
        self.early_stopping_monitor = hparams.get("early_stopping_monitor")
        self.early_stopping_mode = hparams.get("early_stopping_mode")
        self.early_stopping_value = hparams.get("early_stopping_value")
        if self.early_stopping_monitor:
            self._es_crit = EarlyStopAtCriteria(monitor=self.early_stopping_monitor, mode=self.early_stopping_mode, value=self.early_stopping_value)
            self._callbacks.append(self._es_crit)
        else:
            self._es_crit = None
        self._other_callbacks = hparams.get("other_callbacks")
        if self._other_callbacks is not None:
            assert isinstance(self._other_callbacks, list), "other_callbacks must be a list of callbacks"
            self._callbacks.extend(self._other_callbacks)
        self._custom_schedule = hparams.get("custom_schedule")
        
    
        # Construct an empty Sequential model
        self.net = tf.keras.models.Sequential()
    
    
    def call(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)
    
    # def build(self):
    #     super().build(input_shape=self.batch_input_shape) 
        
    def summary(self):
        return self.net.summary()
        
    def get_config(self):
        config = super(KerasSmartModel, self).get_config()
        config['hparams'] = self.hparams
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(config['hparams'])
    
    
    def compile_model(self, num_samples=None):
        """Compiles Keras Smart Model based on its constructor hyperparameters

        Args:
            num_samples (int, optional): Number of samples. Defaults to None.
        """
        compile_keras_model(self.net, _batchsize=self._batch_size, _learnrate=self._learning_rate, _optimizer=self._optimizer, 
                            _loss=self._loss_function, _metrics=self._metrics, _optimizer_params=self._optimizer_params,
                            _loss_params=self._loss_function_params, _metrics_params=self._metrics_params, 
                            _exponential_decay_rate=self._exponential_decay_rate, num_samples=num_samples, custom_schedule=self._custom_schedule)
    
    
    def fit_model(self, x_train, y_train, x_val=None, y_val=None, verbose:int=1, **kwargs):
        """Fit (train) the Keras Smart Model to training data using pure numpy arrays.
        NOTE: Use `fit_model_with_dataset` if you use tf.data.Dataset objects rather than pure numpy arrays.

        Args:
            x_train (array): Training inputs
            y_train (array): Training target outputs
            x_val (array, optional): Validation inputs. Defaults to None.
            y_val (array, optional): Validation target outputs. Defaults to None.
            verbose (int, optional): Verbosity passed to the Keras Fit function. Defaults to 1.
            
            Other keyword arguments are passed to the Keras fit function.

        Returns:
            History object returned by the Keras fit function
        """
        self.history = fit_keras_model(self.net, x_train, y_train, x_val, y_val, 
            self._batch_size, self._epochs, self._callbacks, verbose, **kwargs)
        return self.history
    
    
    def fit_model_with_dataset(self, train_dataset, val_dataset=None, verbose:int=1, **kwargs):
        """Fit (train) the Keras Smart Model to training data using datasets rather than pure numpy arrays.
        NOTE: Use `fit_model` if you want to use pure numpy arrays as inputs to the model.

        Args:
            train_dataset (tf.data.Dataset): Training dataset
            val_dataset (tf.data.Dataset, optional): Validation dataset. Defaults to None.
            verbose (int, optional): Verbosity passed to the Keras Fit function. Defaults to 1.
            
            Other keyword arguments are passed to the Keras fit function.

        Returns:
            History object returned by the Keras fit function
        """
        self.history = fit_keras_model_with_dataset(self.net, train_dataset, val_dataset, 
            self._batch_size, self._epochs, self._callbacks, verbose, **kwargs)
        return self.history

    
    def __str__(self):
        s = "KerasSmartModel with the following attributes:\n" + str(self.hparams)
        return s
    
    
    def train_model(self, x_train, y_train, x_val=None, y_val=None, verbose:int=1, save_model_to:str=None, save_hparams_to:str=None, 
                    export_to_file:str=None, save_kwargs:dict={}, **kwargs):
        """Train the model according to its hyperparameters, using pure numpy arrays.
        NOTE: Use `train_model_with_dataset` if you use tf.data.Dataset objects rather than pure numpy arrays.

        ### Args:
            - `x_train` (numpy array): Training inputs
            - `y_train` (numpy array): Training target outputs
            - `x_val` (numpy array, optional): Validation inputs. Defaults to None.
            - `y_val` (numpy array, optional): Validation target outputs. Defaults to None.
            - `verbose` (int, optional): Verbosity of training passed to Keras fit function. Defaults to 1.
            - `save_model_to` (str, optional): Save Keras model in path. Defaults to None. Path does not need to exist.
              The path can have no extension, in which case a SavedModel format will be used, or it can have a .h5 extension, in which case a HDF5 format will be used, or it can have a .keras extension, in which case the new Keras format will be used.
            - `save_hparams_to` (str, optional): Save hyperparameters to path. Defaults to None. Path does not need to exist.
            - `export_to_file` (str, optional): Save Keras model in .model file using keras2cpp for later use in C++. Defaults to None.
              Path does not need to exist.
            - `save_kwargs` (dict, optional): Additional kwargs to pass to the `tf.keras.Model.save()` function. Defaults to None.
            
            Other keyword arguments are passed to the Keras `tf.keras.Model.fit()` function.

        ### Returns:
            Nothing. It modifies the "net" attribute of the model, and the history of the training in self.history.
        
        """
        N = x_train.shape[0]
        self.compile_model(num_samples=N)
        _ = self.fit_model(x_train, y_train, x_val, y_val, verbose=verbose, **kwargs)
        if save_model_to:
            save_keras_model(
                model=self.net, 
                history=self.history.history, 
                save_model_to=save_model_to, 
                save_hparams_to=save_hparams_to, 
                hparams=self.hparams,
                **save_kwargs)
        if export_to_file:
            export_keras_model(self.net, export_to_file)
    
    
    
    def train_model_with_dataset(self, train_dataset, val_dataset, verbose:int=1, save_model_to:str=None, save_hparams_to:str=None, 
                    export_to_file:str=None, save_kwargs:dict={}, **kwargs):
        """Train the model according to its hyperparameters using tf.data.Dataset objects.
        NOTE: Use `train_model` if you use pure numpy arrays rather than tf.data.Dataset objects.

        ### Args:
            - `train_dataset` (tf.data.Dataset): Training dataset
            - `val_dataset` (tf.data.Dataset, optional): Validation dataset. Defaults to None.
            - `verbose` (int, optional): Verbosity of training passed to Keras fit function. Defaults to 1.
            - `save_model_to` (str, optional): Save Keras model in path. Defaults to None. Path does not need to exist.
              The path can have no extension, in which case a SavedModel format will be used, or it can have a .h5 extension, in which case a HDF5 format will be used, or it can have a .keras extension, in which case the new Keras format will be used.
            - `save_hparams_to` (str, optional): Save hyperparameters to path. Defaults to None. Path does not need to exist.
            - `export_to_file` (str, optional): Save Keras model in .model file using keras2cpp for later use in C++. Defaults to None.
              Path does not need to exist.
            - `save_kwargs` (dict, optional): Additional kwargs to pass to the `tf.keras.Model.save()` function. Defaults to None.
            
            Other keyword arguments are passed to the Keras `tf.keras.Model.fit()` function.

        ### Returns:
            Nothing. It modifies the "net" attribute of the model, and the history of the training in self.history.
        
        """
        try:
            N = tf.data.experimental.cardinality(train_dataset).numpy()
        except Exception as e:
            print("Error in eznet_keras.models.keras_smart_module.train_model_with_dataset():")
            print(e)
            print("Cannot determine the number of samples in the training dataset."
                  "This is only necessary for determining number of iterations per epoch for learning rate scheduling."
                  "Without the number of samples being known, learning rate scheduling might be less efficient because"
                  "number of iterations per epoch would be set to 1.")
            N = None
            
        
        self.compile_model(num_samples=N)
        _ = self.fit_model_with_dataset(train_dataset, val_dataset, verbose=verbose, **kwargs)
        if save_model_to:
            save_keras_model(
                model=self.net, 
                history=self.history.history, 
                save_model_to=save_model_to, 
                save_hparams_to=save_hparams_to, 
                hparams=self.hparams,
                **save_kwargs)
        if export_to_file:
            export_keras_model(self.net, export_to_file)



    def plot_history(self, metrics=['loss'], fig_title='model loss', saveto:str=None, close_after_finish:bool=True):
        """Plot the training history of the model after training is done.

        ### Args:
            - `metrics` (list, optional): Metrics list. Defaults to ['loss']. It can also be ['loss','val_loss'] if validation data was provided, or ['accuracy','val_accuracy'].
            - `fig_title` (str, optional): Title of the figure. Defaults to 'model loss'.
            - `saveto` (str, optional): Path to where to save the figure. Defaults to None. Path does not need to exist.
            - `close_after_finish` (bool, optional): Clsoe the figure after saving it with `plt.close()`. Defaults to True. Only applicable if `saveto` is provided.
        """
        plot_keras_model_history(self.history.history, metrics, fig_title, saveto, close_after_finish)
      
        
    def evaluate(self, *args, **kwargs):
        """Evaluate model performance on test data.

        Returns:
            The same thing that Keras evaluate returns.
        """
        return self.net.evaluate(*args, **kwargs)
    
    
    def predict(self, *args, **kwargs):
        """Predict the outputs given inputs

        Returns:
            Teh same thing that Keras predict returns.
        """
        return self.net.predict(*args, **kwargs)
    
    
    def make_regularizer(self):
        if self._l1_reg is not None and self._l1_reg > 0 and (self._l2_reg is None or self._l2_reg == 0): # Only do L1 regularization
            return tf.keras.regularizers.L1(self._l1_reg)
        elif self._l2_reg is not None and self._l2_reg > 0 and (self._l1_reg is None or self._l1_reg == 0): # Only do L2 regularization
            return tf.keras.regularizers.L2(self._l2_reg)
        elif self._l1_reg is not None and self._l1_reg > 0 and self._l2_reg is not None and self._l2_reg > 0: # Do both L1 and L2 regularization
            return tf.keras.regularizers.L1L2(self._l1_reg, self._l2_reg)
        else:
            return None 
    
    
    def save_model(self, save_model_to:str=None, save_hparams_to:str=None, export_to_file:str=None, **kwargs):
        """Save the model to file.

        ### Args:
            - `save_model_to` (str, optional): Save Keras model in path. Defaults to None. Path does not need to exist.
              The path can have no extension, in which case a SavedModel format will be used, or it can have a .h5 extension, in which case a HDF5 format will be used, or it can have a .keras extension, in which case the new Keras format will be used.
            - `save_hparams_to` (str, optional): Save hyperparameters to path. Defaults to None. Path does not need to exist.
            - `export_to_file` (str, optional): Save Keras model in .model file using keras2cpp for later use in C++. Defaults to None.
              Path does not need to exist.
            - **kwargs: Additional kwargs to pass to the `tf.keras.Model.save()` function. Defaults to None.
        """
        save_keras_model(
            model=self.net, 
            history=None, 
            save_model_to=save_model_to, 
            save_hparams_to=save_hparams_to, 
            hparams=self.hparams,
            **kwargs)
        if export_to_file:
            export_keras_model(self.net, export_to_file)
    
    
    def export_model(self, export_to_file:str):
        """Export the model to file.

        ### Args:
            - `export_to_file` (str): Save Keras model in .model file using keras2cpp for later use in C++. Path does not need to exist.
        """
        export_keras_model(self.net, export_to_file)
        

