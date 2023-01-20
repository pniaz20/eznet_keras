if __package__=="eznet_keras.models":
    from .keras_smart_module import *
    from .conv_block import *
    from .dense_block import *
else:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from keras_smart_module import *
    from conv_block import *
    from dense_block import *

class Conv_Network(KerasSmartModel):
    sample_hparams = {
        "model_name": "Conv_Network",
        # I/O shapes (without the batch dimension)
        "input_shape": [28, 28, 3],
        "output_shape": [10],
        # Convolution blocks
        "num_conv_blocks": 2,
        "conv_dim": 2,
        "conv_params": None,
        "conv_channels": "auto",
        "conv_kernel_size": 3,
        "conv_padding": "valid",
        "conv_stride": 1,
        "conv_dilation": 1,
        "conv_activation": "relu",
        "conv_activation_params": None,
        "conv_norm_layer_type": "BatchNormalization",
        "conv_norm_layer_position": "before",
        "conv_norm_layer_params": None,
        "conv_dropout": 0.1,
        "pool_type": "Max",
        "pool_kernel_size": 2,
        "pool_padding": 'valid',
        "pool_stride": 1,
        "pool_params": None,
        "min_image_size": 4,
        # Fully connected blocks
        "dense_width": "auto",
        "dense_depth": 2,
        "dense_activation": "relu",
        "dense_activation_params": None,
        "output_activation": "softmax",
        "output_activation_params": None,
        "dense_norm_layer_type": "BatchNormalization",
        "dense_norm_layer_position": "before",
        "dense_norm_layer_params": None,
        "dense_dropout": 0.1,
        # Training procedure
        "l2_reg": 0.0001,
        "batch_size": 32,
        "epochs": 2,
        "validation_data": [0.05,'testset'],
        "validation_tolerance_epochs": 5,
        "learning_rate": 0.01,
        "learning_rate_decay_gamma": 0.9,
        "loss_function": "categorical_crossentropy",
        "optimizer": "Adam",
        "optimizer_params": None,
        'metrics':['accuracy'],
        'checkpoint_path':None,
        'early_stopping_monitor':'loss',
        'early_stopping_mode':'min',
        'early_stopping_value':1.0e-6
    }
    
    
    def __init__(self, hparams:dict=None):
        """Standard Convolutional Neural Network, containing convolutional blocks followed by fully-connected blocks. It supports 1D, 2D, and 3D convolutions, and can be used for 
        image classification, timeseries classification, video classification, and so forth. The module can easily be trained and evaluated using its own methods,
        because it inherits from `KerasSmartModel`.

        ### Usage

        `model = Conv_Network(hparams)` where `hparams` is dictionary of hyperparameters containing the following:

        #### I/O shapes
        
        - `input_shape` (list): Input shape *WITHOUT* the batch dimension. For instance, for 2D images, input should be [N, H, W, C], therefore `input_shape` should be [H, W, C].
        - `output_shape` (int): Output shape *WITHOUT* the batch dimension. For instance, for K-class classification, model outputs can be [N, K], so `output_shape` should be [K].
            
        #### Convolution blocks
        
        - `num_conv_blocks` (int): Number of convolutional blocks. Every block contains a convolutional layer, and
            optionally a normalization layer, an activation layer, a pooling layer, and finally a dropout layer.
        - `conv_dim` (int): Dimensionality of the convolution. 1, 2, or 3.
        - `conv_params` (dict): (list of) kwargs dict to pass to the convolution constructor in each block. Defaults to None.
        - `conv_channels` (int|list|str): (list of) Number of filters of the convolution layer in each conv block. If `auto`, it will start
            with the input channels, and double with every block, in powers of two. If `list`, it should be a list
            of channels for each conv block. If `int`, it will be the same for all conv blocks. Default is `auto`.
        - `conv_kernel_size` (int|list): (list of) Kernel size of the convolution layers. Should be a list of integers,
            a list of tuples of integers (for conv2d or conv3d), or an integer. If it is a list, it MUST have the same 
            length as `num_conv_blocks`. If it is an integer, it will be the same for all conv blocks. Defaults to 3.
        - `conv_padding` (int|str|list): (list of) Paddings of convolution layers. Format is as `conv_kernel_size`. Defaults to "valid".
        - `conv_stride` (int|list): (list of) Stride of convolution layers. Format is as `conv_kernel_size`. Defaults to 1.
        - `conv_dilation` (int|list): (list of) Dilation of convolution layers. Format is as `conv_kernel_size`. Defaults to 1.
        - `conv_activation` (str|list): (list of) string(s) representing activation func of the convolution layers. Examples: 'relu', 'leakyrelu', 'sigmoid', 'tanh', etc.
        - `conv_activation_params` (dict|list): (list of) dicts for the convolution activation functions' constructors. Defaults to None.
        - `conv_norm_layer_type` (str|list): (list of) types of normalization layers to use in the conv blocks. Examples: 'BatchNormalization', 'LayerNormalization', etc.
            Defaults to None.
        - `conv_norm_layer_position` ("before"|"after"|list): (list of) positions of the normalization layers in the 
            convolutional blocks relative to the activation functions. Defaults to "before". If it is a list, it should be a list of strings of the same length as `num_conv_blocks`
        - `conv_norm_layer_params` (dict|list): (list of) kwargs dict for the convolution normalization layers' constructors. Defaults to None.    
        - `conv_dropout` (float|list): (list of) Dropout rates of the convolution blocks. Defaults to None.
        - `pool_type` (str|list): (list of) types of pooling layer. "Max", "Avg", "GlobalMax", "GlobalAvg", etc. Defaults to None, in which case there will be no pooling layer.
        - `pool_kernel_size` (int|list): (list of) kernel sizes of the pooling layers, with similar format to 
            `conv_kernel_size`. Again, it can be a list of integers, a list of tuples of integers, or an integer.
        - `pool_padding` (str|list): (list of) paddings of the pooling layers.
        - `pool_stride` (int|list): (list of) strides of the pooling layers.
        - `pool_params` (dict|list): (list of) kwargs dicts for the pooling layers' constructors.
        - `min_image_size` (int): Minimum size of the image to be reduced to in convolutions and poolings.
            After this point, the padding and striding will be chosen such that image size does not decrease further. Defaults to 1.
            
        #### Dense blocks
        
        - `dense_width` ("auto"|int|list): Width of the hidden layers of the Dense network. "auto", a number (for all of them) or a list holding width of each hidden layer.
            If "auto", it will start with the output size of the Flatten() layer, halving at every Dense block.
        - `dense_depth` (int): Depth (number of hidden layers) of the Dense network.
        - `dense_activation` (str|list): (list of) activation function for hidden layers of the Dense network. Examples: 'relu', 'leakyrelu', 'sigmoid', 'tanh', etc.
        - `dense_activation_params` (dict|list): (list of) dicts for the dense activation functions' constructors.
        - `output_activation` (str): Activation function for the output layer of the Dense network, if any.
            **NOTE** Depending on the loss function, you may not need an activation function.
        - `output_activation_params` (dict): Dictionary of parameters for the output activation function's constructor.
        - `dense_norm_layer_type` (str|list): (list of) types of normalization layers to use in the dense blocks. Examples: 'BatchNormalization', 'LayerNormalization', etc.
            Defaults to None, in which case no normalization layer will be used.
        - `dense_norm_layer_position` ("before"|"after"|list): (list of) positions of the normalization layers in the dense blocks relative to the activation functions. 
            Defaults to "before". If it is a list, it should be a list of strings of the same length as `dense_depth`.
        - `dense_norm_layer_params` (dict|list): (list of) kwargs dict for the dense normalization layers' constructors.
        - `dense_dropout` (float|list): (list of) Dropout rates (if any) for the hidden layers of the Dense network.
        
        #### Training procedure
        
        - `batch_size` (int): Minibatch size, the expected input size of the network.
        - `learning_rate` (float): Initial learning rate of training.
        - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
        - `optimizer` (str): Optimizer. Examples: 'Adam', 'SGD', 'RMSprop', etc.
        - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
        - `epochs` (int): Maximum number of epochs for training.
        - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
        - `l2_reg` (float): L2 regularization parameter.
        - `loss_function` (str): Loss function. Examples: 'mse','categorical_crossentropy',etc.
        - `loss_function_params` (dict): Additional parameters for the loss function, if any.
        - `validation_data` (tuple): Validation data, if any. It should be a tuple of (portion, from_dataset). For instance, [0.05, 'testset'] means 5% of the testset will be used 
            for validation.The second element of the tuple can only be 'trainset' and 'testset'. The first element must be a float between 0 and 1. 
            If the second element is not specified, testset will be used by default.
        - `metrics` (list): list of metrics for Keras compilation, e.g. ['accuracy'].
        - `checkpoint_path` (str): Path to the directory where checkpoints will be saved at every epoch.
        - `early_stopping_monitor` (str): Monitor whose critical value will cause early stopping. Default is 'loss', but 'val_loss' is typically used.
        - `early_stopping_mode` (str): Mode of the parameter whose critical value will be used for early stopping. Deafults to 'min' for any error. 'max' is for accuracy, etc.
        - `early_stopping_value` (float): Value of the monitor at which point training will stop becasue the critical value has been reached.
        
        ### Returns
        
        - Returns a `tf.keras.models.Model` object that can be trained and used accordingly.
        - Run `net.summary()` afterwards to see what you have inside the network.
        - A `KerasSmartModel` object is returned. This module has its own functions for training, evaluation, etc.
        """
        super(Conv_Network, self).__init__(hparams)
        if not hparams: hparams = self.sample_hparams
        # Input and output shapes
        self.model_name = hparams["model_name"] if hparams.get("model_name") else "Conv_Network"
        self._input_shape = hparams["input_shape"]
        self._output_shape = hparams["output_shape"]
        self._N = int(hparams["batch_size"])
        self.batch_input_shape = list(self._input_shape).copy()
        self.batch_input_shape.insert(0, self._N)
        self.batch_output_shape = list(self._output_shape).copy()
        self.batch_output_shape.insert(0, self._N)
        self.size_list = [self._input_shape]
    
        # Initialiaing sequential network
        self.net = tf.keras.models.Sequential()
        
        # Convolutional layers hyperparameters
        self._num_conv_blocks = hparams.get("num_conv_blocks")
        self._conv_dim = hparams.get("conv_dim")
        self._conv_params = hparams.get("conv_params")    
        self._conv_channels = hparams.get("conv_channels") if hparams.get("conv_channels") else "auto"
        self._conv_kernel_size = hparams.get("conv_kernel_size") if hparams.get("conv_kernel_size") else 3
        self._conv_padding = hparams["conv_padding"] if hparams.get("conv_padding") else "valid"
        self._conv_stride = hparams["conv_stride"] if hparams.get("conv_stride") else 1
        self._conv_dilation = hparams["conv_dilation"] if hparams.get("conv_dilation") else 1
        self._conv_activation = hparams["conv_activation"] if hparams.get("conv_activation") else "relu"
        self._conv_activation_params = hparams.get("conv_activation_params")
        self._conv_norm_layer_type = hparams.get("conv_norm_layer_type")
        self._conv_norm_layer_position = hparams.get("conv_norm_layer_position")
        self._conv_norm_layer_params = hparams.get("conv_norm_layer_params")
        self._conv_dropout = hparams.get("conv_dropout")
        self._pool_type = hparams.get("pool_type")
        self._pool_kernel_size = hparams.get("pool_kernel_size") if hparams.get("pool_kernel_size") else 2
        self._pool_padding = hparams["pool_padding"] if hparams.get("pool_padding") else 'valid'
        self._pool_stride = hparams["pool_stride"] if hparams.get("pool_stride") else 1
        self._pool_params = hparams.get("pool_params")
        self._min_image_size = hparams["min_image_size"] if hparams.get("min_image_size") else 1        
        
        # Generate lists of hyperparameters for conv/pool layers
        self._conv_channels_vec = self._gen_hparam_vec_for_conv(self._conv_channels, "conv_channels", 
            check_auto=True, init_for_auto=self._input_shape[-1], powers_of_two_if_auto=True, direction_if_auto="up")
        self._conv_kernel_size_vec = self._gen_hparam_vec_for_conv(self._conv_kernel_size, "conv_kernel_size")
        self._pool_kernel_size_vec = self._gen_hparam_vec_for_conv(self._pool_kernel_size, 'pool_kernel_size')
        self._conv_padding_vec = self._gen_hparam_vec_for_conv(self._conv_padding, 'conv_padding')
        self._pool_padding_vec = self._gen_hparam_vec_for_conv(self._pool_padding, 'pool_padding')
        self._conv_stride_vec = self._gen_hparam_vec_for_conv(self._conv_stride, 'conv_stride')
        self._pool_stride_vec = self._gen_hparam_vec_for_conv(self._pool_stride, 'pool_stride')
        self._conv_dilation_vec = self._gen_hparam_vec_for_conv(self._conv_dilation, 'conv_dilation')
        self._conv_activation_vec = self._gen_hparam_vec_for_conv(self._conv_activation, 'conv_activation')
        self._conv_activation_params_vec = self._gen_hparam_vec_for_conv(self._conv_activation_params, 'conv_activation_params')
        self._pool_type_vec = self._gen_hparam_vec_for_conv(self._pool_type, 'pool_type')
        self._pool_params_vec = self._gen_hparam_vec_for_conv(self._pool_params, 'pool_params')
        self._conv_params_vec = self._gen_hparam_vec_for_conv(self._conv_params, 'conv_params')
        self._conv_norm_layer_type_vec = self._gen_hparam_vec_for_conv(self._conv_norm_layer_type, 'conv_norm_layer_type')
        self._conv_norm_layer_params_vec = self._gen_hparam_vec_for_conv(self._conv_norm_layer_params, 'conv_norm_layer_params')
        self._conv_norm_layer_position_vec = self._gen_hparam_vec_for_conv(self._conv_norm_layer_position, 'conv_norm_layer_position')
        self._conv_dropout_vec = self._gen_hparam_vec_for_conv(self._conv_dropout, 'conv_dropout')
        
        # Constructing the encoder (convolutional blocks)
        # print("input_shape: ", self._input_shape)
        in_channels = self._input_shape[-1]
        input_image = list(self._input_shape[:-1])
        for i in range(self._num_conv_blocks):
            out_channels = self._conv_channels_vec[i]
            # print("in_channels: ", in_channels)
            # print("out_channels: ", out_channels)
            # print("input_image: ", input_image)
            _kwargs = {
                'model':self.net, 
                'out_channels':out_channels, 
                'conv_dim':self._conv_dim, 
                'input_image':input_image, 
                'conv_kernel_size':self._conv_kernel_size_vec[i], 
                'conv_padding':self._conv_padding_vec[i],
                'conv_stride':self._conv_stride_vec[i], 
                'conv_dilation':self._conv_dilation_vec[i], 
                'conv_params':self._conv_params_vec[i], 
                'conv_activation':self._conv_activation_vec[i], 
                'conv_activation_params':self._conv_activation_params_vec[i], 
                'norm_layer_position':self._conv_norm_layer_position_vec[i], 
                'norm_layer_type':self._conv_norm_layer_type_vec[i], 
                'norm_layer_params':self._conv_norm_layer_params_vec[i], 
                'pool_type':self._pool_type_vec[i], 
                'pool_kernel_size':self._pool_kernel_size_vec[i], 
                'pool_padding':self._pool_padding_vec[i], 
                'pool_stride':self._pool_stride_vec[i], 
                'pool_params':self._pool_params_vec[i], 
                'dropout':self._conv_dropout_vec[i], 
                'min_image_dim':self._min_image_size,
                'kernel_regularizer':(tf.keras.regularizers.L2(self._l2_reg) if self._l2_reg>0 else None)
            }
            if i==0:
                _kwargs.update({'input_shape':self._input_shape})
            d = add_conv_block(**_kwargs)
            self.net = d['model']
            output_image = d['output_image']
            self.size_list.append(output_image+[out_channels])
            in_channels = out_channels
            input_image = output_image
            
        # Flattening (Image embedding)
        self.net.add(tf.keras.layers.Flatten())
        self._dense_input_size = np.prod(output_image) * out_channels
        self.size_list.append([self._dense_input_size])
        
        # Dense layers hyperparameters
        self._dense_width = hparams["dense_width"]
        self._dense_depth = hparams["dense_depth"]
        self._dense_activation = hparams["dense_activation"] if hparams.get("dense_activation") else "relu"
        self._dense_activation_params = hparams.get("dense_activation_params")
        self._output_activation = hparams.get("output_activation") if hparams.get("output_activation") else None
        self._output_activation_params = hparams.get("output_activation_params")
        self._dense_norm_layer_type = hparams.get("dense_norm_layer_type")
        self._dense_norm_layer_params = hparams.get("dense_norm_layer_params")
        self._dense_norm_layer_position = hparams.get("dense_norm_layer_position")
        self._dense_dropout = hparams.get("dense_dropout")
        self._out_activation_module = actdict_keras[self._output_activation] if self._output_activation else None
        
        # Generate lists of hyperparameters for the dense layers
        self._dense_width_vec = self._gen_hparam_vec_for_dense(self._dense_width, 'dense_width',
            check_auto=True, init_for_auto=self._dense_input_size, powers_of_two_if_auto=True, direction_if_auto="down")
        self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._dense_activation, 'dense_activation')
        self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._dense_activation_params, 'dense_activation_params')
        self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_type, 'dense_norm_layer_type')
        self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_params, 'dense_norm_layer_params')
        self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_position, 'dense_norm_layer_position')
        self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._dense_dropout, 'dense_dropout')
        
        # Construct the dense layers
        in_size = self._dense_input_size
        for i in range(self._dense_depth):
            out_size = self._dense_width_vec[i]
            temp_dropout_rate = self._dense_dropout_vec[i] if (i != self._dense_depth-1) else None # The hidden layer just before the output layer rarely has Dropout.
            _kwargs = {
                'model':self.net,
                'output_size':self._dense_width_vec[i],
                'activation':self._dense_activation_vec[i],
                'activation_params':self._dense_activation_params_vec[i],
                'norm_layer_type':self._dense_norm_layer_type_vec[i],
                'norm_layer_position':self._dense_norm_layer_position_vec[i],
                'norm_layer_params':self._dense_norm_layer_params_vec[i],
                'dropout':temp_dropout_rate,
                'kernel_regularizer':(tf.keras.regularizers.L2(self._l2_reg) if self._l2_reg>0 else None)
            }
            add_dense_block(**_kwargs)
            in_size = out_size
            self.size_list.append([out_size])
        
        # Output layer
        self.net.add(tf.keras.layers.Dense(self._output_shape[-1]))
        if self._output_activation:
            if self._output_activation_params: self.net.add(tf.keras.layers.Activation(self._out_activation_module(**self._output_activation_params)))
            else: self.net.add(tf.keras.layers.Activation(self._out_activation_module))

    
    def _gen_hparam_vec_for_conv(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._num_conv_blocks, hparam_name=hparam_name, count_if_not_list_name='num_conv_blocks', **kwargs)
    
    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._dense_depth, hparam_name=hparam_name, count_if_not_list_name='dense_depth', **kwargs)
    
    def call(self, inputs, **kwargs):
        return self.net(inputs, **kwargs)
