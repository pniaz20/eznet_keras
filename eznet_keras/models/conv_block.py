if __package__=="eznet_keras.models":
    from ..utils import *
else:
    import os, sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from utils import *
import warnings

class Conv_Block(tf.keras.layers.Layer):
    def __init__(self, input_shape:list, out_channels:int=None, conv_dim:int=1, input_image:list=[30], conv_kernel_size=3, conv_padding='valid', conv_stride=1, conv_dilation=1, 
                 conv_params:dict=None, conv_activation:str='relu', conv_activation_params:dict=None, norm_layer_position:str=None, norm_layer_type:str=None, 
                 norm_layer_params:dict=None, pool_type:str=None, pool_kernel_size=2, pool_padding:str='valid', pool_stride=1, pool_params:dict=None, 
                 dropout:float=None, min_image_dim:int=1, kernel_regularizer:tf.keras.regularizers.Regularizer=None):
        """Convolutional block, containing one convolution layer, followed optionally by a normalization layer, an activation layer, a pooling layer and a dropout layer. The
        convolution layer is mandatory, but the other ones are optional. The convolution layer can be 1D, 2D or 3D. The normalization layer can be any such layer defined
        in Keras. The activation layer can also be anything with lower-case strings, and the pooling layer can be any of the pooling layers defined for Keras. 
        The dropout layer is a spatial dropout layer. The dimension of any dropout layer will match the dimension of the convolution layer. For `Conv1D` for instance, 
        `SpatialDropout1D` will be used, if desired.
        
        This module is meant to be used as a building block for larger modules.

        ### Args:
        
        - `input_shape` (list|tuple): Input shape.
        - `out_channels` (int, optional): Number of convolution filters. Defaults to the input channels size.
        - `conv_dim` (int, optional): Dimension of the convolution. Defaults to 1. 1 means Conv1d, 2 means Conv2d etc.
        - `input_image` (list, optional): Size of the input image. Defaults to [30]. This must be a list/tuple of integers, with legth equal to `conv_dim`.
        - `conv_kernel_size` (int, optional): Convolution kernel size. Defaults to 3. It is strongly recommended to provide a list of integers, with length equal to `conv_dim`.
        - `conv_padding` (str, optional): Convolution padding. Defaults to 'same'. Arrays are recommended over integers.
        - `conv_stride` (int, optional): Convolution stride. Defaults to 1.
        - `conv_dilation` (int, optional): Convolution dilation. Defaults to 1.
        - `conv_params` (dict, optional): Additional dictionary of kwargs for Conv?d module, to add or overwrite other parameters. Defaults to None.
          No matter what parameters are set by this class by default for Con?D layer, providing this dictionary will overwrite all of them.
        - `conv_activation` (str, optional): String representing activation function. Defaults to 'relu'. Examples: 'leakyrelu', 'sigmoid', 'tanh' etc.
        - `conv_activation_params` (dict, optional): kwargs dictionary for activation function. Defaults to None.
        - `norm_layer_position` (str, optional): Position of the normalization layer relative to activation. Defaults to None. It should be 'before' or 'after' or None.
        - `norm_layer_type` (str, optional): Type of the normalization layer. Defaults to None. Examples: 'BatchNormalization', 'LayerNormalization', etc.
        - `norm_layer_params` (dict, optional): kwargs dictionary for normalization layer. Defaults to None.
        - `pool_type` (str, optional): Type of pooling layer, if any. Defaults to None. For example, 'Max', 'Avg', 'GlobalMax', 'GlobalAvg' etc.
        - `pool_kernel_size` (int, optional): Pooling kernel size. Defaults to 2. Arrays are recommended over integers.
        - `pool_padding` (int, optional): Padding for pooling layer. Defaults to 0. 'same' is NOT an option here.
        - `pool_stride` (int, optional): Pooling stride. Defaults to 1.
        - `pool_params` (dict, optional): kwargs dictionary for pooling layer module to add to or overwrite its arguments. Defaults to None.
        - `dropout` (float, optional): Dropout rate, if any. Defaults to None. for Conv?D, SpatialDropout?D is used.
        - `min_image_dim` (int, optional): Minimum image dimension. Defaults to 1. This is used for preventing the image dimension from becoming too small. 
            It can automatically adjust padding and stride for convolution and pooling layers to keep the image dimensions larger than this argument.
        - `kernel_regularizer` (regularizer, optinal): Regularizer to be used for the kernel weights in the layers.
        
        ### Returns:
        A `tf.keras.layers.Layer` instance representing a single convolutional block.
        
        ### Attributes:
        `self.output_image` (list): Size of the output image.
        `self.net` (tf.keras.models.Sequential): The actual network, a `tf.keras.models.Sequential` instance.
        """
        super(Conv_Block, self).__init__()
        self.net = tf.keras.models.Sequential()
        ret = add_conv_block(self.net, out_channels, input_shape, conv_dim, input_image, conv_kernel_size, conv_padding, conv_stride, conv_dilation, conv_params,
                             conv_activation, conv_activation_params, norm_layer_position, norm_layer_type, norm_layer_params, 
                             pool_type, pool_kernel_size, pool_padding, pool_stride, pool_params, dropout, min_image_dim, kernel_regularizer)
        self.net = ret['model']
        self._input_shape = input_shape
        self.output_image = ret['output_image']
        self._out_channels = ret['out_channels']
        self._norm_layer_position = ret['norm_layer_position']
        self._norm_layer_type = ret['norm_layer_type']
        self._min_image_dim = ret['min_image_dim']
        self._conv_padding = ret['conv_padding']
        self._conv_stride = ret['conv_stride']
        self._pool_padding = ret['pool_padding']
        self._pool_stride = ret['pool_stride']
        self._conv_module = ret['conv_module']
        self._conv_activation_module = ret['conv_activation_module']
        self._pool_module = ret['pool_module']
        self._dropout_module = ret['dropout_module']
        self._norm_layer_module = ret['norm_layer_module']
        
        # Store parameters as class attributes
        self._conv_kernel_size = conv_kernel_size
        self._conv_dilation = conv_dilation
        self._conv_params = conv_params
        self._conv_activation = conv_activation
        self._conv_activation_params = conv_activation_params
        self._norm_layer_params = norm_layer_params
        self._pool_type = pool_type
        self._pool_kernel_size = pool_kernel_size
        self._pool_params = pool_params
        self._dropout = dropout if dropout else None
        self._kernel_regularizer = kernel_regularizer

    def call(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)
    
    def get_config(self):
        config = super(Conv_Block, self).get_config()
        hparams = {
            "input_shape":self._input_shape,
            "out_channels":self._out_channels,
            "conv_dim":self._conv_dim,
            "input_image":self._input_image,
            "conv_kernel_size":self._conv_kernel_size,
            "conv_padding":self._conv_padding,
            "conv_stride":self._conv_stride,
            "conv_dilation":self._conv_dilation,
            "conv_params":self._conv_params,
            "conv_activation":self._conv_activation,
            "conv_activation_params":self._conv_activation_params,
            "norm_layer_position":self._norm_layer_position,
            "norm_layer_type":self._norm_layer_type,
            "norm_layer_params":self._norm_layer_params,
            "pool_type":self._pool_type,
            "pool_kernel_size":self._pool_kernel_size,
            "pool_padding":self._pool_padding,
            "pool_stride":self._pool_stride,
            "pool_params":self._pool_params,
            "dropout":self._dropout,
            "min_image_dim":self._min_image_dim,
            "kernel_regularizer":self._kernel_regularizer
        }
        config['hparams'] = hparams
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config['hparams'])
    
    def summary(self):
        return self.net.summary()



def add_conv_block(model:tf.keras.models.Sequential, out_channels:int=None, input_shape:list=None, conv_dim:int=1, input_image:list=[30], conv_kernel_size=3, conv_padding='valid',
    conv_stride=1, conv_dilation=1, conv_params:dict=None, conv_activation:str='relu', conv_activation_params:dict=None, norm_layer_position:str=None, norm_layer_type:str=None, 
    norm_layer_params:dict=None, pool_type:str=None, pool_kernel_size=2, pool_padding:str='valid', pool_stride=1, pool_params:dict=None, dropout:float=None, min_image_dim:int=1,
    kernel_regularizer:tf.keras.regularizers.Regularizer=None):
    """ Add a Convolutional block, containing one convolution layer, followed optionally by a normalization layer, an activation layer, a pooling layer and a dropout layer,
    to the end of an existing `tf.keras.models.Sequential` instance.
    The convolution layer is mandatory, but the other ones are optional. The convolution layer can be 1D, 2D or 3D. The normalization layer can be any such layer defined
    in Keras. The activation layer can also be anything with lower-case strings, and the pooling layer can be any of the pooling layers defined for Keras. 
    The dropout layer is a spatial dropout layer. The dimension of any dropout layer will match the dimension of the convolution layer. For `Conv1D` for instance, 
    `SpatialDropout1D` will be used, if desired.

    ### Args:
    
    - `input_shape` (list|tuple): Input shape.
    - `out_channels` (int, optional): Number of convolution filters. Defaults to the input channels size.
    - `conv_dim` (int, optional): Dimension of the convolution. Defaults to 1. 1 means Conv1d, 2 means Conv2d etc.
    - `input_image` (list, optional): Size of the input image. Defaults to [30]. This must be a list/tuple of integers, with legth equal to `conv_dim`.
    - `conv_kernel_size` (int, optional): Convolution kernel size. Defaults to 3. It is strongly recommended to provide a list of integers, with length equal to `conv_dim`.
    - `conv_padding` (str, optional): Convolution padding. Defaults to 'same'. Arrays are recommended over integers.
    - `conv_stride` (int, optional): Convolution stride. Defaults to 1.
    - `conv_dilation` (int, optional): Convolution dilation. Defaults to 1.
    - `conv_params` (dict, optional): Additional dictionary of kwargs for constructor of Conv?d module. Defaults to None.
      If provided, this dictionary will not only add to, but also overwrite any existing arguments passed to the Conv?D constructor.
    - `conv_activation` (str, optional): String representing activation function. Defaults to 'relu'. Examples: 'leakyrelu', 'sigmoid', 'tanh' etc.
    - `conv_activation_params` (dict, optional): kwargs dictionary for activation function. Defaults to None.
    - `norm_layer_position` (str, optional): Position of the normalization layer relative to activation. Defaults to None. It should be 'before' or 'after' or None.
    - `norm_layer_type` (str, optional): Type of the normalization layer. Defaults to None. Examples: 'BatchNormalization', 'LayerNormalization', etc.
    - `norm_layer_params` (dict, optional): kwargs dictionary for normalization layer. Defaults to None.
    - `pool_type` (str, optional): Type of pooling layer, if any. Defaults to None. For example, 'Max', 'Avg', 'GlobalMax', 'GlobalAvg' etc.
    - `pool_kernel_size` (int, optional): Pooling kernel size. Defaults to 2. Arrays are recommended over integers.
    - `pool_padding` (int, optional): Padding for pooling layer. Defaults to 0. 'same' is NOT an option here.
    - `pool_stride` (int, optional): Pooling stride. Defaults to 1.
    - `pool_params` (dict, optional): kwargs dictionary for pooling layer module, to add to or overwrite existing parameters. Defaults to None.
    - `dropout` (float, optional): Dropout rate, if any. Defaults to None. for Conv?D, SpatialDropout?D is used.
    - `min_image_dim` (int, optional): Minimum image dimension. Defaults to 1. This is used for preventing the image dimension from becoming too small. 
        It can automatically adjust padding and stride for convolution and pooling layers to keep the image dimensions larger than this argument.
    - `kernel_regularizer` (regularizer, optional): Regularizer to be used for the kernel weights in the layers.
    
    ### Returns:
    A dictionary with the following fields:
    ```python
    {
        'model'
        'output_image'
        'out_channels'
        'norm_layer_position'
        'norm_layer_type'
        'min_image_dim'
        'conv_padding'
        'conv_stride'
        'pool_padding'
        'pool_stride'
        'conv_module'
        'conv_activation_module'
        'pool_module'
        'dropout_module'
        'norm_layer_module'
    }
    ```
    """
    # Output channels check
    if not (isinstance(out_channels,int) and out_channels > 0):
        warnings.warn("Invalid value out_channels={}. Using value {} equal to input_shape[-1].".format(out_channels,input_shape), UserWarning)
        out_channels = input_shape[-1]
    # Convolution dimension check    
    assert isinstance(conv_dim, int) and conv_dim in [1,2,3], "`conv_dim` must be an integer among [1,2,3], not {} which has type {}.".format(conv_dim, str(type(conv_dim)))
    # Determine convolution module used
    _conv_module = convdict_keras["conv{}d".format(conv_dim)]
    # Input image size check
    assert isinstance(input_image, (list,tuple)) and len(input_image)==conv_dim, \
        "`input_image` must be a list or tuple of length equal to `conv_dim`, not {} which has type {}.".format(input_image, str(type(input_image)))
    # Check activation function and module
    _conv_activation_module = actdict_keras[conv_activation]  if conv_activation else None
    # Check position, type and parameters of normalization layer
    if not norm_layer_position in ['before', 'after', None]:
        warnings.warn(("Invalid value {} for `norm_layer_position`: It can only be 'before' (before activation), 'after' (after activation) or None. "+
                        "Using default value of None. There will be no normalization.").format(norm_layer_position), UserWarning)
        norm_layer_position = None
    if norm_layer_position is None: norm_layer_type = None
    if norm_layer_type is None: norm_layer_position = None
    _norm_layer_module = getattr(tf.keras.layers, norm_layer_type) if norm_layer_type else None
    # Check pooling layer type, module and parameters
    _pool_module = getattr(tf.keras.layers, "{}Pool{}D".format(pool_type, conv_dim)) if pool_type else None
    # Check Dropout parameters
    _dropout_module = getattr(tf.keras.layers, 'SpatialDropout{}D'.format(conv_dim)) if dropout else None
    # Store minimum desired image size
    min_image_dim = min_image_dim if min_image_dim>0 else 1
    img_size = input_image
    # -----------------------------------------------------------------------------        
    # Check if output image size is smaller than min_image_dim, and adjust parameters if necessary
    # print("Running calc_size for input image size {}".format(img_size))
    temp_img_size = calc_image_size(img_size, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, dilation=conv_dilation)
    if min(temp_img_size) < min_image_dim:
        warnings.warn(
            "Output image ({}) of the convolution operation is smaller in one or more dimensions than min_image_dim={} for add_conv_block(). ".format(temp_img_size,min_image_dim)+ 
            "Using padding='same' and stride=1 instead of padding={} and stride={}".format(conv_padding, conv_stride), UserWarning)
        conv_padding = 'same'
        conv_stride = 1
    # Construct convolutional layer
    _kwargs = {'filters':out_channels, 'kernel_size':conv_kernel_size, 'strides':conv_stride, 'padding':conv_padding, 'dilation_rate':conv_dilation, 
               'kernel_regularizer':kernel_regularizer}
    if input_shape: _kwargs.update({'input_shape':input_shape})
    if conv_params: _kwargs.update(conv_params)
    model.add(_conv_module(**_kwargs))
    # Calculate output image size
    img_size = calc_image_size(img_size, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding, dilation=conv_dilation)
    # ---------------------------------------------------------------------------
    # Construct normalization layer, if it should be here.
    if norm_layer_position=='before':
        if norm_layer_params: model.add(_norm_layer_module(**norm_layer_params))
        else: model.add(_norm_layer_module())
    # Construct activation layer
    if conv_activation:
        if conv_activation_params: model.add(tf.keras.layers.Activation(_conv_activation_module(**conv_activation_params)))
        else: model.add(tf.keras.layers.Activation(_conv_activation_module))
    # Construct normalization layer, if it should be here.
    if norm_layer_position=='after':
        if norm_layer_params: model.add(_norm_layer_module(**norm_layer_params))
        else: model.add(_norm_layer_module())
    # ---------------------------------------------------------------------------
    # Check type and parameters of the pooling layer, and calculate output image size
    if pool_type is not None and 'global' in pool_type.lower():
        if pool_params: model.add(_pool_module(**pool_params))
        else: model.add(_pool_module())
        img_size = [1]*conv_dim
    elif pool_type is not None:
        temp_img_size = calc_image_size(img_size, kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding, dilation=1) # Keras pool has no support for dilation :(
        if min(temp_img_size) < min_image_dim:
            warnings.warn(
                "Output image ({}) of the pooling operation is smaller in one or more dimensions than min_image_dim={} for add_conv_block(). ".format(temp_img_size,min_image_dim)+ 
                "Using padding={} and stride=1 instead of padding={} and stride={}".format('same', pool_padding, pool_stride), UserWarning)
            pool_padding = 'same'
            pool_stride = 1
        _kwargs = {'pool_size':pool_kernel_size, 'strides':pool_stride, 'padding':pool_padding}
        if pool_params: _kwargs.update(pool_params)
        model.add(_pool_module(**_kwargs))
        img_size = calc_image_size(img_size, kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding, dilation=1)
    # ---------------------------------------------------------------------------
    # Construct Dropout layer    
    if dropout: model.add(_dropout_module(dropout))
    # Store output image size as attribute    
    output_image = img_size
    d = {
        'model':model, 'output_image':output_image, 'out_channels':out_channels, 'norm_layer_position':norm_layer_position,
        'norm_layer_type':norm_layer_type, 'min_image_dim':min_image_dim, 'conv_padding':conv_padding, 'conv_stride':conv_stride, 'pool_padding':pool_padding,
        'pool_stride':pool_stride, 'conv_module':_conv_module, 'conv_activation_module':_conv_activation_module, 'pool_module':_pool_module,
        'dropout_module':_dropout_module, 'norm_layer_module':_norm_layer_module
    }
    return d

