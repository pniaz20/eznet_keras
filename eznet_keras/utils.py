
# Global Constants
SEED = 42

# General-Purpose Libraries
if __name__ == "eznet_keras.utils":
    from .keras2cpp import export_model
else:
    from keras2cpp import export_model
import os
import warnings
from pathlib import Path
import math
import json
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
from timeit import default_timer as timer
from datetime import datetime
from matplotlib import pyplot as plt

# Tensorflow Libraries
import tensorflow as tf
tf.random.set_seed(SEED)
import gc

# Reset seeds, just in case they were modified during imports
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
GLOROTUNIFORM = tf.keras.initializers.GlorotUniform(seed=SEED)
ORTHOGONAL = tf.keras.initializers.Orthogonal(seed=SEED)


########################################################################################################################
# Global variables, functions, and classes
########################################################################################################################

optdict_keras = {'adam':tf.keras.optimizers.Adam, 'sgd':tf.keras.optimizers.SGD, 'rmsprop':tf.keras.optimizers.RMSprop, 'adagrad':tf.keras.optimizers.Adagrad}
actdict_keras = {
    'relu':tf.keras.activations.relu, 'leakyrelu':tf.keras.layers.LeakyReLU, 
    'sigmoid':tf.keras.activations.sigmoid, 'tanh':tf.keras.activations.tanh, 'softmax':tf.keras.activations.softmax,
    'softplus':tf.keras.activations.softplus, 'softsign':tf.keras.activations.softsign,
    'elu':tf.keras.activations.elu, 'selu':tf.keras.activations.selu}
rnndict_keras = {'LSTM':tf.keras.layers.LSTM, 'GRU':tf.keras.layers.GRU, 'SimpleRNN':tf.keras.layers.SimpleRNN}
convdict_keras = {"conv1d":tf.keras.layers.Conv1D, "conv2d":tf.keras.layers.Conv2D, "conv3d":tf.keras.layers.Conv3D}


def make_path(path:str):
    Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
    return path
    

def plot_keras_model_history(history:dict, metrics:list=['loss'], fig_title:str='model loss', saveto:str=None, close_after_finish:bool=True):
    plt.figure(figsize=(7, 5))
    plt.grid(True)
    plt.plot(history[metrics[0]], label='training')
    if len(metrics) > 1:
        plt.plot(history[metrics[1]], label='validation')
    plt.title(fig_title)
    plt.ylabel(metrics[0])
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    if saveto:
        plt.savefig(make_path(saveto), dpi=600)
        if close_after_finish:
            plt.close()
        



def compile_keras_model(model, _batchsize:int, _learnrate:float, _optimizer:str, _loss:str, _metrics:list, 
                          _optimizer_params:dict=None, _loss_params:dict=None, _metrics_params:list=None, _exponential_decay_rate:float=None, num_samples:int=None, custom_schedule=None):
    if custom_schedule:
        lr = custom_schedule
    elif _exponential_decay_rate:
        itersPerEpoch = (num_samples//_batchsize) if num_samples else 1
        sch = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=_learnrate, 
        decay_steps=itersPerEpoch, decay_rate=_exponential_decay_rate)
        lr = sch
    else:
        lr = _learnrate
    
    _opt_module = getattr(tf.keras.optimizers, _optimizer) if isinstance(_optimizer, str) else _optimizer
    
    if _optimizer_params:
        optparam = _optimizer_params
        # opt = optdict_keras[_optimizer](learning_rate=lr, **optparam)
        opt = _opt_module(learning_rate=lr, **optparam)
    else:
        # opt = optdict_keras[_optimizer](learning_rate=lr)
        opt = _opt_module(learning_rate=lr)
    
    if isinstance(_loss, str):
        if _loss.lower()==_loss: # Put it as is
            _lossfunc = _loss
        else: # It is the name of a Keras losses class
            _loss_module = getattr(tf.keras.losses, _loss)
            if _loss_params:
                _lossfunc = _loss_module(**_loss_params)
            else:
                _lossfunc = _loss_module()
    else: # It is a custom loss function class
        if _loss_params:
            _lossfunc = _loss(**_loss_params)
        else:
            _lossfunc = _loss()
        
    
    assert isinstance(_metrics, list), "Metrics must be a list of strings or a list of metrics"
    if _metrics_params and isinstance(_metrics_params, dict):
        _metrics_params = [_metrics_params]
    if _metrics_params:
        assert len(_metrics_params)==1 or len(_metrics_params)==len(_metrics), \
            "If metrics_params is specified, it must be a list of length 1 or the same length as metrics."
    if _metrics_params:
        _metrics_params_vec = _metrics_params if len(_metrics_params)==len(_metrics) else [_metrics_params[0]]*len(_metrics)
            
    _metrics_list = []
    for i,metric in enumerate(_metrics):
        if isinstance(metric, str):
            if metric.lower()==metric: # Put it as is
                metr = metric
            else: # It is the name of a Keras metrics class
                _metr_module = getattr(tf.keras.metrics, metric)
                if _metrics_params_vec[i] is not None:
                    metr = _metr_module(**_metrics_params_vec[i])
                else:
                    metr = _metr_module()
        else: # It is a custom metric class
            if _metrics_params_vec[i] is not None:
                metr = metric(**_metrics_params_vec[i])
            else:
                metr = metric()
        _metrics_list.append(metr)
    
    
    model.compile(optimizer=opt, loss=_lossfunc, metrics=_metrics_list)
    



def fit_keras_model(model, x_train, y_train, x_val=None, y_val=None, 
    _batchsize:int=None, _epochs:int=1, _callbacks:list=None, verbose:int=1, **kwargs):
    while True:
        try:
            history = model.fit(x_train, y_train, batch_size=_batchsize, epochs=_epochs, 
                validation_data=((x_val, y_val) if x_val is not None and y_val is not None else None), verbose=verbose, 
                callbacks=_callbacks, **kwargs)
            break
        except Exception as e:
            print(e)
            print(("\nTraining failed with batchsize={}. "+\
                "Trying again with a lower batchsize...").format(_batchsize))
            _batchsize = _batchsize // 2
            if _batchsize < 2:
                raise ValueError("Batchsize too small. Training failed.")
    return history


def fit_keras_model_with_dataset(model, train_dataset, val_dataset=None, 
    _batchsize:int=None, _epochs:int=1, _callbacks:list=None, verbose:int=1, **kwargs):
    while True:
        try:
            history = model.fit(train_dataset, batch_size=_batchsize, epochs=_epochs, 
                validation_data=val_dataset, verbose=verbose, 
                callbacks=_callbacks, **kwargs)
            break
        except Exception as e:
            print(e)
            print(("\nTraining failed with batchsize={}. "+\
                "Trying again with a lower batchsize...").format(_batchsize))
            _batchsize = _batchsize // 2
            if _batchsize < 2:
                raise ValueError("Batchsize too small. Training failed.")
    return history




def save_keras_model(model, save_model_to:str, save_hparams_to:str=None, history:dict=None, hparams:dict=None, **kwargs):
    try:
        model.save(make_path(save_model_to), **kwargs)
        if history is not None:
            assert hparams is not None, "If training history is to be saved along with the hyperparameters, then `hparams` must be provided."
            for key in history:
                hparams[key] = history[key]
        if hparams is not None:
            assert save_hparams_to is not None, "`save_hparams_to` must be provided to where the hyperparameters are to be saved."
            if not save_hparams_to.endswith(".json"):
                warnings.warn("save_hparams_to does not end with '.json'. A json extension is highly recommended for hyperparameters.", UserWarning)
            jstr = json.dumps(hparams, indent=4)
            with open(make_path(save_hparams_to), "w") as f:
                f.write(jstr)
    except Exception as e:
        print("Cannot serialize Keras model.")
        raise e
        
        
def export_keras_model(model, path:str):
    try:
        export_model(model, make_path(path))
        print("Model exported successfully.")
    except Exception as e1:
        print(e1)
        print("Cannot export Keras model using keras2cpp on the fly. Will try sequentializing the model layers...")
        try:
            net = tf.keras.models.Sequential(model.layers)
            export_model(net, make_path(path))
            print("Model exported successfully.")
        except Exception as e2:
            print("Cannot export model using Keras2Cpp.")
            raise e2
    

def test_keras_model_class(model_class, hparams:dict=None, save_and_export:bool=True):
    print("Constructing model...\n")
    model = model_class(hparams)
    print("Summary of model:")
    print(model.summary())
    print("\nGenerating random dataset...\n")
    (x_train, y_train) = generate_sample_batch(model)
    (x_val, y_val) = generate_sample_batch(model)
    print("Trying forward pass on training data: ")
    y = model(x_train)
    print("\nOutput shape: ", y.shape)
    print("\nTraining model...\n")
    model_name = model.hparams["model_name"]
    model.train_model(x_train, y_train, x_val, y_val, 
                verbose=1, 
                save_model_to=(f"test_{model_name}" if save_and_export else None),
                save_hparams_to=(f"test_{model_name}_hparams.json" if save_and_export else None), 
                export_to_file=(f"test_{model_name}.model" if save_and_export else None))
    print("\nEvaluating model...\n")
    model.evaluate(x_val, y_val, verbose=1)
    print("Done.")
    

# Perform garbage collection
import gc
class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
        
# Deploy early stopping when performance reaches good values
class EarlyStopAtCriteria(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', mode='min', value=0.001):
        super(EarlyStopAtCriteria, self).__init__()
        self.monitor = monitor
        self.value = value
        self.mode = mode
    def on_epoch_end(self, epoch, logs=None):
        if self.mode == 'min':
            if logs.get(self.monitor) <= self.value:
                print("Early stopping performance criteria has been reached. Stopping training.")
                self.model.stop_training = True
        else:
            if logs.get(self.monitor) >= self.value:
                print("Early stopping performance criteria has been reached. Stopping training.")
                self.model.stop_training = True
        

# Sampling layer used for variational autoencoders
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoname(name):
    """
    Genereate a unique name for a file, based on the current time and the given name.
    Gets the `name` as a string and adds the time stamp to the end of it before returning it.
    """
    return name + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")



def calc_image_size(size_in:int, kernel_size:int, padding:int, stride:int, dilation:int):
    """Calculate image size after convolution or pooling.
    ### Args:
        - `size_in` (int|list): (list of) image input dimension(s).
        - `kernel_size` (int|list): (list of) kernel (or pool) size
        - `padding` (int|list): (list of) padding sizes, or string such as 'valid' and 'same'.
        - `stride` (int|list): (list of) strides.
        - `dilation` (int|list): (list of) dilation rates.
    ### Returns:
        int or list: Output image dimensions
    """
    if padding == 'same':
        return size_in
    else:
        if padding == 'valid':
            padding = 0
        if isinstance(size_in, (list, tuple)):
            if isinstance(padding, int): padding = [padding]*len(size_in)
            if isinstance(kernel_size, int): kernel_size = [kernel_size]*len(size_in)
            if isinstance(stride, int): stride = [stride]*len(size_in)
            if isinstance(dilation, int): dilation = [dilation]*len(size_in)
            return [math.floor((size_in[i] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)/stride[i] + 1) for \
                i in range(len(size_in))]
        else:
            assert isinstance(size_in, int), "size_in must be an integer or a list/tuple of integers."
            assert isinstance(padding, int), "padding must be an integer or a list/tuple of integers."
            assert isinstance(kernel_size, int), "kernel_size must be an integer or a list/tuple of integers."
            assert isinstance(stride, int), "stride must be an integer or a list/tuple of integers."
            assert isinstance(dilation, int), "dilation must be an integer or a list/tuple of integers."
            return math.floor((size_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)


def generate_geometric_array(init, count, direction, powers_of_two=True):
    """Generate array filled with incrementally doubling/halving values, optionally with powers of two.

    ### Args:
    
        - `init` (int): The first value to begin.
        - `count` (int): Number of elements to generate.
        - `direction` (str): Direction of the array. Can be either 'up' or 'down', i.e. increasing or decreasing.
        - `powers_of_two` (bool, optional): Generate numbers that are powers of two. Defaults to True.

    ### Returns:
    
        list: List containing elements
    """
    lst = []
    old = int(2**math.ceil(math.log2(init))) if powers_of_two else init
    new = old
    for _ in range(count):
        lst.append(new)
        old = new
        new = (old * 2) if direction == 'up' else (old // 2)
    return lst


def generate_array_for_hparam(
    hparam, count_if_not_list:int, 
    hparam_name:str='parameter', count_if_not_list_name:str='its count',
    check_auto:bool=False, init_for_auto:int=2, powers_of_two_if_auto:bool=True,
    direction_if_auto:str=None):
    """Generate array for a hyperparameter, regardless of if it is a list or not. This function is for use in APIs
    that generate models with hyperparameters as inputs, which can be lists, a single item, or "auto".
    Examples include width of a neural network's hidden layers, channels of conv layers, etc.
    For these hyperparameters, the user is typically free to specify an array-like, a single item to be repeated,
    or "auto" for automatic calculation of the parameter.
    This function is meant to be used in the body of the code of class constructors and other functions in the API.

    ### Args:
    
        - `hparam` (var): A specific hyperparameter, e.g., input by user to your network's constructor.
        - `count_if_not_list` (int): Number of elements to generate if `hparam` is not an array-like.
        - `hparam_name` (str, optional): Name of the hyperparameter. Defaults to 'parameter'.
        - `count_if_not_list_name` (str, optional): Name of the "count" that must be provided. Defaults to 'its count'.
        - `check_auto` (bool, optional): Check for the "auto" case. Defaults to False.
        - `init_for_auto` (int, optional): Initial value in case of "auto". Defaults to 2.
        - `powers_of_two_if_auto` (bool, optional): Generate powers of two in case of "auto". Defaults to True.
        - `direction_if_auto` (str, optional): Direction of geometric increment in case of "auto". Defaults to None.
           This can be "up" or "down". If check_for_auto is True, then this argument must be specified.

    ### Returns:
    
        list: List containing elements
    """
    assert count_if_not_list is not None, \
        "Since %s may not be a list/tuple, %s must always be specified."%(hparam_name, count_if_not_list_name)
    if isinstance(hparam, (list,tuple)) and len(hparam) == count_if_not_list:
        lst = hparam
    elif hparam == "auto" and check_auto:
        assert init_for_auto is not None, \
            "If %s is 'auto', then %s must be specified."%(hparam_name, "init_for_auto")
        assert direction_if_auto in ['up','down'], \
            "If %s is 'auto', then %s must be specified as 'up' or 'down'."%(hparam_name, "direction_if_auto")
        lst = generate_geometric_array(init_for_auto, count_if_not_list, direction_if_auto, powers_of_two_if_auto)
    else:
        lst = [hparam]*count_if_not_list
    return lst




def generate_sample_batch(model):
    x = np.random.rand(*model.batch_input_shape).astype(np.float32)
    y = np.random.rand(*model.batch_output_shape).astype(np.float32)
    if model.hparams['loss_function'] in ['sparse_categorical_crossentropy']:
        y = np.argmax(y, axis=1)
    return (x,y)
    