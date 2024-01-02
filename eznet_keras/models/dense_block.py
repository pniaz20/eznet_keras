

if __package__=="eznet_keras.models":
    from ..utils import *
else:
    import os, sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from utils import *


class Dense_Block(tf.keras.layers.Layer):
    def __init__(self, input_shape:list=None, output_size:int=None, activation:str=None, activation_params:dict=None, dense_params:dict=None, norm_layer_type:str=None, norm_layer_position:str='before', 
                 norm_layer_params:dict=None, dropout:float=None, kernel_regularizer:tf.keras.regularizers.Regularizer=None):
        """Dense (fully connected) block containing one linear layer, followed optionally by a normalization layer, an activation function and a Dropout layer.

        ### Args:

            - `input_shape` (list|tuple): Shape of the input without the batch size.
            - `output_size` (int, optional): Number of output features. Defaults to None, in which case it will be input_size.
            - `activation` (str, optional): Activation. It can be an activation function name ("relu","sigmoid","tanh", etc.),
                an activation layer name ("ReLU", "LeakyReLU", "Softmax", etc.), or a custom Keras Layer class (not instance). Defaults to None.
            - `activation_params` (dict, optional): kwargs to pass to the activation function constructor. Defaults to None.
                Ignored if `activation` is a lower-case function name. By the way, the slope of the negative section in `LeakyReLU` is `alpha`.
            - `dense_params` (dict, optional): kwargs to pass to the dense layer constructor. Defaults to None.
                This will overwrite any other parameters such as regularizers, etc.
            - `norm_layer_type` (str, optional): Type of normalization layer. Defaults to None. Examples: 'BatchNormalization',
                'LayerNormalization', etc. It can also be a Keras Layer class (not instance).
            - `norm_layer_position` (str, optional): Position of norm layer relative to activation. Defaults to 'before'. Alternative is 'after'.
            - `norm_layer_params` (dict, optional): kwargs to pass to the norm layer constructor. Defaults to None.
            - `dropout` (float, optional): Dropout rate at the end. Defaults to None. Must be a float between 0 and 1.
            - `kernel_regularizer` (regularizer, optinal): Regularizer instance to be used for the kernel weights in the layers.
        
        
        **NOTE**: 
        Activation functions come from `tf.keras.activations` and need to be wrapped in a `tf.keras.layers.Activation` layer, and do not accept any
        kwargs. Activation layers come from `tf.keras.layers`, can accept kwargs in their constructors, and do not need to be wrapped in an
        `Activation` layer. Activation functions all have lower-case names, but activation layers are classes and have every-word-capitalized names.
        We will use this as a clue to recognize what the user wants, and perform accordingly.
        Most activation layers also have correpsonding activation functions (like `tf.keras.activations.relu` function and
        `tf.keras.layers.ReLU` layer). However, some activations are only available as functions (such as `tf.keras.activations.sigmoid`),
        and some are only available as layers (such as `tf.keras.layers.LeakyReLU`). Choose accordingly.
            
        ### Returns:
        
        A `tf.keras.layers.Layer` object.
        """
        super(Dense_Block, self).__init__()
        if output_size is None: 
            if input_shape:
                output_size = input_shape[-1]
            else:
                raise ValueError("Either input_shape or output_size must be provided.")
        self._input_shape = input_shape
        self._output_size = output_size
        self._activation = activation
        self._activation_params = activation_params
        self._norm_layer_type = norm_layer_type
        self._norm_layer_position = norm_layer_position
        self._norm_layer_params = norm_layer_params
        self._dense_params = dense_params
        self._dropout = dropout
        if activation is not None:
            if isinstance(activation, str):
                if activation.lower()==activation:
                    self._activation_module = getattr(tf.keras.activations, activation)
                else:
                    self._activation_module = getattr(tf.keras.layers, activation)
            else:
                self._activation_module = activation
        else:
            self._activation_module = None
        if norm_layer_type is not None:
            if isinstance(norm_layer_type, str):
                self._norm_layer_module = getattr(tf.keras.layers, norm_layer_type)
            else:
                self._norm_layer_module = norm_layer_type
        else:
            self._norm_layer_module = None
        self._dropout_module = tf.keras.layers.Dropout if dropout else None
        self._kernel_regularizer = kernel_regularizer
        self.net = tf.keras.models.Sequential()
        _kwargs = {
            'units': output_size
        }
        if input_shape:
            _kwargs.update({'input_shape':input_shape})
        if kernel_regularizer is not None:
            _kwargs.update({'kernel_regularizer':kernel_regularizer})
        if dense_params is not None:
            _kwargs.update(dense_params)
        self.net.add(tf.keras.layers.Dense(**_kwargs))
        if norm_layer_type and norm_layer_position=='before': 
            if norm_layer_params: self.net.add(self._norm_layer_module(**norm_layer_params))
            else: self.net.add(self._norm_layer_module())
        if activation:
            if isinstance(activation, str) and activation.lower()==activation: 
                self.net.add(tf.keras.layers.Activation(self._activation_module))
            else:
                if activation_params: self.net.add(self._activation_module(**activation_params))
                else: self.net.add(self._activation_module())
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
            'kernel_regularizer':self._kernel_regularizer,
            'dense_params':self._dense_params
        }
        config['hparams'] = hparams
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config['hparams'])
    
    def summary(self):
        return self.net.summary()
        



def add_dense_block(model:tf.keras.models.Sequential, output_size:int, input_shape:list=None,
                    activation:str=None, activation_params:dict=None, dense_params:dict=None,
                    norm_layer_type:str=None, norm_layer_position:str='before', norm_layer_params:dict=None, dropout:float=None, 
                    kernel_regularizer:tf.keras.regularizers.Regularizer=None):
    """
    Add a Dense (fully connected) block containing one linear layer, followed optionally by a normalization layer, an activation function and a Dropout layer, 
    to a `tf.keras.models.Sequential` instance.

    ### Args:
        
        - `input_shape` (list|tuple, optional): Shape of the input disregarding the batch size.
        - `output_size` (int, optional): Number of output features. Defaults to None, in which case it will be input_size.
        - `activation` (str, optional): Activation. It can be an activation function name ("relu","sigmoid","tanh", etc.),
            an activation layer name ("ReLU", "LeakyReLU",
            "Softmax", etc.), or a custom Keras Layer class (not instance). Defaults to None.
        - `activation_params` (dict, optional): kwargs to pass to the activation function constructor. Defaults to None.
            Ignored if `activation` is a lower-case function name. By the way, the slope of the negative section in `LeakyReLU` is `alpha`.
        - `dense_params` (dict, optional): kwargs to pass to the dense layer constructor. Defaults to None.
            This will overwrite any other parameters such as regularizers, etc.
        - `norm_layer_type` (str, optional): Type of normalization layer. Defaults to None. Examples: 'BatchNormalization',
            'LayerNormalization', etc. It can also be a Keras Layer class (not instance).
        - `norm_layer_position` (str, optional): Position of norm layer relative to activation. Defaults to 'before'. Alternative is 'after'.
        - `norm_layer_params` (dict, optional): kwargs to pass to the norm layer constructor. Defaults to None.
        - `dropout` (float, optional): Dropout rate at the end. Defaults to None. Must be a float between 0 and 1.
        - `kernel_regularizer` (regularizer, optinal): Regularizer to be used for the kernel weights in the layers.
            
    ### Returns:
        
        Nothing. It modifies the `model` argument passed to it.
        
    """
    
    _dropout_module = tf.keras.layers.Dropout if dropout else None
    
    if activation is not None:
        if isinstance(activation, str):
            if activation.lower()==activation:
                _activation_module = getattr(tf.keras.activations, activation)
            else:
                _activation_module = getattr(tf.keras.layers, activation)
        else:
            _activation_module = activation
    else:
        _activation_module = None
        
    if norm_layer_type is not None:
        if isinstance(norm_layer_type, str):
            _norm_layer_module = getattr(tf.keras.layers, norm_layer_type)
        else:
            _norm_layer_module = norm_layer_type
    else:
        _norm_layer_module = None
    
    
    _kwargs = {
        'units': output_size
    }
    if input_shape is not None:
        _kwargs.update({'input_shape':input_shape})
    if kernel_regularizer is not None:
        _kwargs.update({'kernel_regularizer':kernel_regularizer})
    if dense_params is not None:
        _kwargs.update(dense_params)
    model.add(tf.keras.layers.Dense(**_kwargs))
        
    if norm_layer_type and norm_layer_position=='before': 
        if norm_layer_params: model.add(_norm_layer_module(**norm_layer_params))
        else: model.add(_norm_layer_module())
        
    if activation: 
        if isinstance(activation, str) and activation.lower()==activation: 
            model.add(tf.keras.layers.Activation(_activation_module))
        else:
            if activation_params: model.add(_activation_module(**activation_params))
            else: model.add(_activation_module())
            
    if norm_layer_type and norm_layer_position=='after': 
        if norm_layer_params: model.add(_norm_layer_module(**norm_layer_params))
        else: model.add(_norm_layer_module())
        
    if dropout: model.add(_dropout_module(dropout))
    
    
    

if __name__=='__main__':
    
    # Test Dense_Block class
    print("Testing Dense_Block class...")
    block = Dense_Block(input_shape=[5], output_size=20, activation="LeakyReLU", activation_params={'alpha':0.1}, norm_layer_type="BatchNormalization", 
                        norm_layer_position='before', norm_layer_params=None, dropout=0.1, kernel_regularizer=None)
    block.summary()
    y = block(tf.random.normal([32,5]))
    print("Shape of output tensor:", y.shape)
    
    print("\n")
    print("Testing add_dense_block function...")
    model = tf.keras.models.Sequential()
    add_dense_block(model, output_size=20, input_shape=[10], activation="LeakyReLU", activation_params={'alpha':0.1}, 
                    norm_layer_type="BatchNormalization", norm_layer_position='before', norm_layer_params=None, dropout=0.5, 
                    kernel_regularizer=None)
    model.summary()
    y = model(tf.random.normal([32,10]))
    print("Shape of output tensor:", y.shape)
    