# ezNet-Keras

Keras (TensorFlow) implementation of **ezNet** ("easy net"), a package containing "easy" implementation of a collection of basic and widely-used deep learning models.  
This implementation is for Keras (Tensorflow). See [Here](https://github.com/pniaz20/eznet_torch) for an identical PyTorch implementation.

Author: Pouya P. Niaz (<pniaz20@ku.edu.tr> , <pouya.p.niaz@gmail.com>)  
Version: 1.1.3  
Last Update: July 27, 2024

Install with:

```bash
pip install eznet-keras
```

-----------------------------------

## 1- Intro

You can build, train and evaluate all manner of Tensorflow/Keras models using the utilities in this package.
Furthermore, there is a collection of basic and widely-used deep learning models ready to be used immediately.
This package also offers `KerasSmartModel`, a sublass of `tf.keras.Model` that has built-in functions for manipulating its hyperparameters,
training, evaluation, and testing.

Note that for all the functions and classes described briefly here, the docstrings provide much more detailed information.

### 1-1- Implementation notes

Unlike, e.g., TensorFlow, you may not be able to just import everything together as in `import eznet_keras` and then use dot indexing to access everything underneath
(I would love to do that, I haven't yet quite figured out how to. I am a newbie.)

Instead, import the specific module, class or function, e.g.,

```python
from eznet_keras.models import ANN
# or 
from eznet_keras.utils import calc_image_size
```

-----------------------------------

## 2- Applications

### 2-1- Smart Model for Convenient DL Training and Deployment

The `KerasSmartModel` class enables you to write any kind of TensorFlow model, and it has built-in functions for training, testing, evaluaiton, plotting training history, exporting to C++ using Keras2Cpp, and so forth.

```python
from eznet_keras.models import KerasSmartModel

class MyModel(KerasSmartModel):
    def __init__(self, hparams:dict=None):
        super(MyModel, self).__init__(hparams)
        #
        # Some code defining final_model
        #
        self.net = final_model
    
    def call(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

sample_hparams = {
    'model_name': 'KerasSmartModel',                    # Name of the model
    'l2_reg': 0.0001,                                   # L2 regularization parameter
    'batch_size': 16,                                   # Mini-batch size
    'epochs': 40,                                       # Maximum training epochs
    'validation_data': [0.05,'trainset'],               # Portion of train set used for validation (if necessary)
    'early_stopping_patience_epochs': 10,               # Validation patience for early stopping
    'learning_rate': 0.0001,                            # (Initial) Learning rate
    'exponential decay rate': 0.99,                     # Learning rate exponential decay factor
    'loss_function': 'categorical_crossentropy',        # Loss function string
    'metrics':['accuracy'],                             # List of metrics
    'optimizer': 'Adam',                                # Optimizer string
    'optimizer_params': None,                           # Dictionary of parameters to pass to the optimizer constructor
    'checkpoint_path':None,                             # Path to store checkpoints in every epoch
    'early_stopping_monitor':'loss',                    # Early stopping criteria (stop when alue is reached and performance is good enough)
    'early_stopping_mode':'min',                        # Early stopping variable mode (minimization or maximization)
    'early_stopping_value':1.0e-6                       # Threshold for early stopping
}

model = MyModel(sample_hparams)

x_train = ...
y_train = ...
x_test = ...
y_test = ...

# If you're working with pure numpy arrays as input and output
model.train_model(x_train, y_train, x_val=None, y_val=None, verbose=1, save_model_to="my_model", save_hparams_to="my_model_hparams.json",
    export_to_file="my_model.model", save_kwargs={})
# Or, alternatively, if you are working with tf.data.Dataset objects (tensorflow datasets)
model.train_model_with_dataset(train_dataset, val_dataset=None, verbose=1, save_model_to="my_model", save_hparams_to="my_model_hparams.json",
    export_to_file="my_model.model", save_kwargs={})

# Plot model learning history
model.plot_history(metrics=['loss','val_loss'], fig_title='model loss', saveto="training_history.png")
```

A few implementation notes:

- Whatever hyperparameters you have for the model itself, can be included in the hparams dictionary.
- The model must always have a `self.net` atttribute which contains the network itself. It can be a Sequential or Functional model.
- The `__init__` method must always calculate the lists `self.batch_input_shape` and `batch_output_shape`.
- The `__init__` method of the `KerasSmartModel` class will create attributes with the same name as the hyperparameters shown above.
- If possible, do not touch the `call` method, leave it be.
- Tha `__init__` method of the class may alkter some of its attributes if the input hyperparameters don't exist or don't make sense.
- The `get_config()` method returns the usual configuration, plus a key called `hparams` holding the (perhaps modified) input hyperparameters.
- The `from_config()` method gets a config that must have the `hparams` key, whose value is the same as the `hparams` input of the constructor.
- `model.summary()` will return `model.net.summary()`. Other training and evaluation functions will also use the `self.net` attribute.
- The `self.history` attribute is `None` at the beginning, but is set to the return value of the `fit()` method.
- The `save_model_to` argument of the `model.train_model()` function can be used to save the trained model as `.h5` file or as a folder in the TensorFlow **SavedModel** format.
- The `save_hparams_to` argument of the `model.train_model()` function can be used to save the trained model's hyperparameter dictionary in a text file using the json format.
- The `export_to_file` argument in the `model.train_model()` function (attempts to) use Gosha20777's [Keras2Cpp](https://github.com/gosha20777/keras2cpp) package to export the model
  to a `.model` file which can be imported in a `C++ 2017` compiler to implement the same model in C++.
  Comes in handy if you work with robotic/mechatronic hardware that use limited versions and builds of C++ like I do :-).

### 2-2- Utility Functions for Manipulating Keras Models

In `eznet_keras.utils`, there are some functions for manipulating Keras models, that come in handy if you frequently work with custom DL models in TensorFlow.

- `make_path(path)` creates a path for a folder or file, if some folders in the path don't exist.
  Anywhere you want to save something, instead of `path/to/foo.bar` you can just use `make_path("path/to/foo.bar")` so if `path` or `to` directories
  don't exist they will be created.
- `plot_keras_model_history(history, **kwargs)` function gets a `history` object returned by any `tf.keras.Model.fit()` function, and plots its training history.
- `compile_keras_model(model, **kwargs)` gets some strings as inputs like learning rate decay, batch size, etc., and compiles a keras model.
- `fit_keras_model(model, **kwargs)` runs the `fit()` function for a Keras model, using appropriate callbacks and early stopping criteria.
- `save_keras_model(model, **kwargs)` attempts to serialize and save a Keras model in either `.h5` format or a diretory with TensorFlow SavedModel format.
- `export_keras_model(model, path)` attempts to export a Keras model as a `.model` file using the Keras2Cpp package.
- `autoname(name)` gets a string as a name, and appends the current time stamp to it. Comes in handy when trying to time stamp the multiple training runs you'll do.
- `calc_image_size(size_in, kernel_size, padding, stride, dilation)` gets the input image dimension (1D, 2D or 3D), along with the
  parameters of a convolution or pooling operation, and returns the output image dimensions. Comes in handy when you want to check to see if your
  CNN layers are not shrinking the image too much.
- `generate_geometric_array()` gets an initial count and returns an array where the count doubles or halves along the array.
  Comes in handy when you want to automatically assign number of filters/channels or hidden sizes in ANNs and CNNs.
- `generate_array_for_hparam()` gets the value of a hyperparameter specified by the user (e.g. ANN width), computes whether the hyperparameter needs to be an array
  (e.g. the user input an integer but ANN width should be an array with length equal to ANN depth), and then returns an array that properly holds the hyperparameter values.
- `generate_sample_batch(model)` gets `model.batch_input_shape` and `model.batch_output_shape` and returns random input and output batches.

### 2-3- Functions and Classes for Adding and Manipulating Modular Dense Blocks and Conv Blocks

This package also has classes and functions that can be used to create entire Dense blocks and Conv blocks.

```python
class eznet_keras.models.DenseBlock(tf.keras.layers.Layer)
```

Gets some arguments to the constructor and returns a layer instance that holds
a **Dense** layer, followed optionally by a **normlization** layer, an **activation** layer, and a **dropout** layer. The arguments to the constructor are
sufficient to build any kind of modular `DenseBlock` layer, stacking it on top of other layers in your model.
I created this class becasue Dense blocks almost always have a widely-used format: dense, norm, activation, dropout.

- Function `eznet_keras.models.dense_block.add_dense_block()` does the same, except rather than returning a `tf.keras.Layer` instance,
  adds the layers to a `tf.keras.models.Sequential()` model that is also the input of the function.

```python
class eznet_keras.models.ConvBlock(tf.keras.layers.Layer)
```

Gets some arguments to the constructor and returns a layer instance that contains a **convolution** layer, followed optionally
by a **normalization** layer, an **activation** layer, a **pooling** layer, and a **dropout** layer.
Again, the inputs are fully sufficient to make any kind of `ConvBlock` and stack it on top of other layers in your CNN.
Similar to the previous case, I created this class becasue it has easy-to-use modular capabilities to build widely-used CNN blocks
that have these kinds of layers in them.

- Function `eznet_keras.models.conv_block.add_conv_block()` does the same, except it gets a `Sequential()` model as input,
  and adds the layers described above to it.

### 2-4- Easy to Use Famous Deep Learning Models for Convenience

This package also holds some widely-used and basic deep learning models as in MLp, CNN, RNN, etc. that can get a dictionary of
well-defined hyperparameters, and return a `KerasSmartModel` instance that can be easily trained, evaluated, stored and deployed.

All of the following models reside in the `eznet_keras.models` submodule.

**NOTE** To see a list of all hyperparameters that each of the following classes use, simply invoke the `class.sample_hparams` class attribute.
Also, you can simply call the `help(class)` function to read the docstrings.
For `ANN`, for instance,

```python
from eznet_keras.models import ANN

print(ANN.sample_hparams)

help(ANN)
```

- `ANN` is a multi-layer perceptron containing many `DenseBlock` blocks, stacked together. For all hyperparameters such as width,
  you can specify an integer to be used for all hidden layers (i.e., all blocks), or an array of different values for each hidden layer (i.e., block).
  For every hyperparameter such as normalization layer type, use `None` in its place in the array of hyperparameters to indicate that the corresponding
  Dense block in that place does not have any normalization layers at all. The same goes for many other hyperparameters.
- `Conv_Network` is a CNN where you not only choose the dimensionality (1D, 2D or 3D convolutions) but also all the other hyperparameters of all
  Convolution blocks and Dense blocks residing in the network. This network is some Convulution blocks, followed by some Dense blocks.
  You get to choose which Conv block has what kind of Conv, Norm (if any), Activation, Pooling (if any), or Dropout (if any) layer.
  You also get to choose custom parametrers (**kwargs) for Conv, Pooling, Norm, Activation and Dense layer constructors, so that you can add additional parameters,
  or overwrite the ones used by the class itself. You have full freedom.
- `Recurrent_Network` is an RNN containing some RNN layers (SimpleRNN, GRU, LSTM, etc.) followed by some Dense blocks. Again, the whole thing is fully modular and you have full freedom.
- `Keras_VarAE` is a standard Variational Autoencoder. It is not a `KerasSmartModel` instance (yet, keep tuned in), but it is a fully developed
  autoencoder model which subclasses the `tf.keras.Model` class and also has KL-divergence loss integrated into it.

More model varieties with modular and easy-to-use functionality will be added in time.

-----------------------------------

## 3- License

This package itself has MIT license, but Keras and TensorFlow have different licenses, which need to be accounted for when using this package.

-----------------------------------

## 4- Credits

Keras  
<https://github.com/keras-team/keras>  
<https://keras.io/>

TensorFlow  
<https://github.com/tensorflow>  
<https://www.tensorflow.org/>

Keras2Cpp  
Gosha20777  
<https://github.com/gosha20777/keras2cpp>

-----------------------------------
