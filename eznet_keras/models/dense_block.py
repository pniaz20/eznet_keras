

if __package__=="eznet_keras.models":
    from ..utils import *
else:
    import os, sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from utils import *


class Dense_Block(tf.keras.layers.Layer):
    def __init__(self, input_shape:list, output_size:int=None, activation:str=None, activation_params:dict=None, norm_layer_type:str=None, norm_layer_position:str='before', 
                 norm_layer_params:dict=None, dropout:float=None, kernel_regularizer:tf.keras.regularizers.Regularizer=None):
        """Dense (fully connected) block containing one linear layer, followed optionally by a normalization layer, an activation function and a Dropout layer.

        ### Args:
            - `input_shape` (list|tuple): Shape of the input.
            - `output_size` (int, optional): Number of output features. Defaults to None, in which case it will be input_size.
            - `activation` (str, optional): Activation function in string form. Defaults to None. Examples: 'relu', 'leakyrelu', 'tanh', 'sigmoid', etc.
            - `activation_params` (dict, optional): kwargs to pass to the activation function constructor. Defaults to None.
            - `norm_layer_type` (str, optional): Type of normalization layer. Defaults to None. Examples: 'BatchNormalization', 'LayerNormalization', etc.
            - `norm_layer_position` (str, optional): Position of norm layer relative to activation. Defaults to 'before'. Alternative is 'after'.
            - `norm_layer_params` (dict, optional): kwargs to pass to the norm layer constructor. Defaults to None.
            - `dropout` (float, optional): Dropout rate at the end. Defaults to None. Must be a float between 0 and 1.
            - `kernel_regularizer` (regularizer, optinal): Regularizer to be used for the kernel weights in the layers.
            
        ### Returns:
        A `tf.keras.layers.Layer` object.
        """
        super(Dense_Block, self).__init__()
        if output_size is None: output_size = input_shape[-1]
        self._input_shape = input_shape
        self._output_size = output_size
        self._activation = activation
        self._activation_params = activation_params
        self._norm_layer_type = norm_layer_type
        self._norm_layer_position = norm_layer_position
        self._norm_layer_params = norm_layer_params
        self._dropout = dropout
        self._activation_module = actdict_keras[activation] if activation else None
        self._norm_layer_module = getattr(tf.keras.layers, norm_layer_type) if norm_layer_type else None
        self._dropout_module = tf.keras.layers.Dropout if dropout else None
        self._kernel_regularizer = kernel_regularizer
        self.net = tf.keras.models.Sequential()
        self.net.add(tf.keras.layers.Dense(output_size, input_shape=input_shape, kernel_regularizer=kernel_regularizer))
        if norm_layer_type and norm_layer_position=='before': 
            if norm_layer_params: self.net.add(self._norm_layer_module(**norm_layer_params))
            else: self.net.add(self._norm_layer_module())
        if activation: 
            if activation_params: self.net.add(tf.keras.layers.Activation(self._activation_module(**activation_params)))
            else: self.net.add(tf.keras.layers.Activation(self._activation_module))
        if norm_layer_type and norm_layer_position=='after': 
            if norm_layer_params: self.net.add(self._norm_layer_module(**norm_layer_params))
            else: self.net.add(self._norm_layer_module())
        if dropout: self.net.add(self._dropout_module(dropout))
    
    def call(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)
    
    def get_config(self):
        config = super(Dense_Block, self).get_config()
        hparams = {
            'input_shape':self._input_shape,
            'output_size':self._output_size,
            'activation':self._activation,
            'activation_params':self._activation_params,
            'norm_layer_type':self._norm_layer_type,
            'norm_layer_position':self._norm_layer_position,
            'norm_layer_params':self._norm_layer_params,
            'dropout':self._dropout,
            'kernel_regularizer':self._kernel_regularizer
        }
        config['hparams'] = hparams
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config['hparams'])
    
    def summary(self):
        return self.net.summary()
        



def add_dense_block(model:tf.keras.models.Sequential, output_size:int, input_shape:list=None, activation:str=None, activation_params:dict=None, 
                    norm_layer_type:str=None, norm_layer_position:str='before', norm_layer_params:dict=None, dropout:float=None, 
                    kernel_regularizer:tf.keras.regularizers.Regularizer=None):
    """Add a Dense (fully connected) block containing one linear layer, followed optionally by a normalization layer, an activation function and a Dropout layer, 
    to a `tf.keras.models.Sequential` instance.

        ### Args:
            - `input_shape` (list|tuple, optional): Shape of the input.
            - `output_size` (int, optional): Number of output features. Defaults to None, in which case it will be input_size.
            - `activation` (str, optional): Activation function in string form. Defaults to None. Examples: 'relu', 'leakyrelu', 'tanh', 'sigmoid', etc.
            - `activation_params` (dict, optional): kwargs to pass to the activation function constructor. Defaults to None.
            - `norm_layer_type` (str, optional): Type of normalization layer. Defaults to None. Examples: 'BatchNormalization', 'LayerNormalization', etc.
            - `norm_layer_position` (str, optional): Position of norm layer relative to activation. Defaults to 'before'. Alternative is 'after'.
            - `norm_layer_params` (dict, optional): kwargs to pass to the norm layer constructor. Defaults to None.
            - `dropout` (float, optional): Dropout rate at the end. Defaults to None. Must be a float between 0 and 1.
            - `kernel_regularizer` (regularizer, optinal): Regularizer to be used for the kernel weights in the layers.
            
        ### Returns:
        Nothing. It modifies the `model` argument passed to it.
        """
    _activation_module = actdict_keras[activation] if activation else None
    _norm_layer_module = getattr(tf.keras.layers, norm_layer_type) if norm_layer_type else None
    _dropout_module = tf.keras.layers.Dropout if dropout else None
    if input_shape:
        model.add(tf.keras.layers.Dense(output_size, input_shape=input_shape, kernel_regularizer=kernel_regularizer))
    else:
        model.add(tf.keras.layers.Dense(output_size, kernel_regularizer=kernel_regularizer))
    if norm_layer_type and norm_layer_position=='before': 
        if norm_layer_params: model.add(_norm_layer_module(**norm_layer_params))
        else: model.add(_norm_layer_module())
    if activation: 
        if activation_params: model.add(tf.keras.layers.Activation(_activation_module(**activation_params)))
        else: model.add(tf.keras.layers.Activation(_activation_module))
    if norm_layer_type and norm_layer_position=='after': 
        if norm_layer_params: model.add(_norm_layer_module(**norm_layer_params))
        else: model.add(_norm_layer_module())
    if dropout: model.add(_dropout_module(dropout))