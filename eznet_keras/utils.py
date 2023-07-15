
# Global Constants
SEED = 42

# General-Purpose Libraries
if __name__ == "eznet_keras.utils":
    from .keras2cpp import export_model
else:
    from keras2cpp import export_model
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
                          _optimizerparams:dict=None, _learnrate_decay_gamma:float=None, num_samples:int=None):
    if _learnrate_decay_gamma:
        itersPerEpoch = (num_samples//_batchsize) if num_samples else 1
        sch = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=_learnrate, 
        decay_steps=itersPerEpoch, decay_rate=_learnrate_decay_gamma)
        lr = sch
    else:
        lr = _learnrate
    if _optimizerparams:
        optparam = _optimizerparams
        # opt = optdict_keras[_optimizer](learning_rate=lr, **optparam)
        opt = getattr(tf.keras.optimizers, _optimizer)(learning_rate=lr, **optparam)
    else:
        # opt = optdict_keras[_optimizer](learning_rate=lr)
        opt = getattr(tf.keras.optimizers, _optimizer)(learning_rate=lr)
    model.compile(optimizer=opt, loss=_loss, metrics=_metrics)
    

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


def save_keras_model(model, history:dict, path:str, hparams:dict):
    try:
        model.save(make_path(path))
        for key in history:
            hparams[key] = history[key]
        jstr = json.dumps(hparams, indent=4)
        with open(path+"/hparams.json", "w") as f:
            f.write(jstr)
    except Exception as e:
        print(e)
        print("Cannot serialize Keras model.")
        
        
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
            print(e2)
            print("Cannot export model using Keras2Cpp.")
    

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
    model.train_model(x_train, y_train, x_val, y_val, 
                verbose=1, 
                saveto=(("test_"+model.hparams["model_name"]) if save_and_export else None), 
                export=(("test_"+model.hparams["model_name"]+".model") if save_and_export else None))
    print("\nEvaluating model...\n")
    model.evaluate(x_val, y_val, verbose=1)
    print("Done.")
    


# Perform garbage collection
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
    
    **NOTE** grouping is not supported yet.

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
            return [math.floor((size_in[i] + 2*padding[i] - dilation[i]*(kernel_size[i]-1) - 1)/stride[i] + 1) for i in range(len(size_in))]
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
    