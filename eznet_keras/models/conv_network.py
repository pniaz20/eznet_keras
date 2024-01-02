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
        "conv_spatial_dropout": None,
        "pool_type": "Max",
        "pool_kernel_size": 2,
        "pool_padding": 'valid',
        "pool_stride": 1,
        "pool_params": None,
        "min_image_size": 4,
        "flatten_after_conv": True,
        # Fully connected blocks
        "include_dense_layers": True,
        "dense_width": "auto",
        "dense_depth": 2,
        "dense_params": None,
        "dense_activation": "relu",
        "dense_activation_params": None,
        "dense_norm_layer_type": "BatchNormalization",
        "dense_norm_layer_position": "before",
        "dense_norm_layer_params": None,
        "dense_dropout": 0.1,
        "include_output_layer": True,
        "output_dense_params": None,
        "output_activation": "softmax",
        "output_activation_params": None,
        # Training procedure
        "l2_reg": 0.0001,
        "l1_reg": None,
        "batch_size": 32,
        "epochs": 2,
        "validation_data": [0.05,'testset'],
        "early_stopping_patience_epochs": 5,
        "learning_rate": 0.01,
        "exponential_decay_rate": 0.9,
        "loss_function": "categorical_crossentropy",
        "loss_function_params": None,
        "optimizer": "Adam",
        "optimizer_params": None,
        'metrics':['accuracy'],
        'metrics_params':None,
        'checkpoint_path':None,
        'early_stopping_monitor':'loss',
        'early_stopping_mode':'min',
        'early_stopping_value':1.0e-6,
        'other_callbacks': None,
        'custom_schedule': None,
    }
    
    
    def __init__(self, hparams:dict=None):
        """Standard Convolutional Neural Network, containing convolutional blocks followed by fully-connected blocks. It supports 1D, 2D, and 3D
        convolutions, and can be used for image classification, timeseries classification, video classification, and so forth.
        The module can easily be trained and evaluated using its own methods, because it inherits from `KerasSmartModel`.
        The architecture consists of conv blocks followed by dense (fully connected) blocks. Each conv block is assumed to contain a convolution
        layer, followed optionally be a normalization layer, an activation, pooling, and finally a dropout or spatial dropout layer.
        Each dense block is assumed to contain a dense layer, followed optionally be a normalziation layer, an activation, and a dropout layer.
        Convolution blocks are mandatory, but Dense layers and an output layer are optional.
        Using this class, any kind of serial CNN architecture can be built. However, no parallelism or skip connections are supported in the
        architecture.

        ### Usage

        `model = Conv_Network(hparams)` where `hparams` is dictionary of hyperparameters containing the following keys. 
        
        - Inspect the `sample_hparams` class attribute for a template of the hyperparameters dictionary.
        - Many keys, especially those that have "(list of)" in their description, can be either scalar items or lists of items.
          If they are scalar, they will be broadcasted to all blocks. If they are lists, they must have the same length as the number (depth) of
          convolutional blocks, i.e. `num_conv_blocks` hyperparameter.
        - Every block by default is assumed to have a convolution operation, a normalization layer, an activation, some form of pooling, and finally
          a dropout or spatial dropout. 
          For every convolutional block, the convolution operation is mandatory but the rest are optional. This way, any kind of CNN with any
          (fully serial) architecture can be built. If the key to some hyperparameter is a list, it must have a length equal to the number of
          convolutional blocks, that is, the `num_conv_blocks` hyperparameter. 
          There must be one item per convolutional block. `None` is typically put for every block that doesn't have that layer or that hyperparameter
          is inapplicable to that block.
          Also note that we assume every convolutional block has a spatial dropout layer followed by a normal dropout layer, but this is not common
          in reality, which means for blocks that have spatial dropout the normal dropout rate should be `None`, and vice versa.
        - The Training Procedure section of the hyperparameters are optional, and will only be used by the parent class `KerasSmartModel` if the
          model is trained using its methods. Otherwise, it is not necessary at all.
        - Activation functions come from `tf.keras.activations` and need to be wrapped in a `tf.keras.layers.Activation` layer, and do not accept any
          kwargs. Activation layers come from `tf.keras.layers`, can accept kwargs in their constructors, and do not need to be wrapped in an
          `Activation` layer. Activation functions all have lower-case names, but activation layers are classes and have every-word-capitalized
          names. We will use this as a clue to recognize what the user wants, and perform accordingly.
          Most activation layers also have correpsonding activation functions (like `tf.keras.activations.relu` function and
          `tf.keras.layers.ReLU` layer). However, some activations are only available as functions (such as `tf.keras.activations.sigmoid`),
          and some are only available as layers (such as `tf.keras.layers.LeakyReLU`). Choose accordingly.

        #### I/O shapes
        
        - `input_shape` (list): Input shape *WITHOUT* the batch dimension. For instance, for 2D images, input should be [N, H, W, C],
          therefore `input_shape` should be [H, W, C].
        - `output_shape` (int): Output shape *WITHOUT* the batch dimension. For instance, for K-class classification, model outputs can be [N, K],
          so `output_shape` should be [K].
            
        #### Convolution blocks
        
        - `num_conv_blocks` (int): Number of convolutional blocks. Every block contains a convolutional layer, and
            optionally a normalization layer, an activation layer, a pooling layer, and finally a dropout or spatial dropout layer.
        - `conv_dim` (int): Dimensionality of the convolution. 1, 2, or 3.
        - `conv_params` (dict): (list of) kwargs dict to pass to the convolution constructor in each block. Defaults to None.
        - `conv_channels` (int|list|str): (list of) Number of filters of the convolution layer in each conv block. If `"auto"`, it will start
            with the input channels, and double with every block, in powers of two. If `list`, it should be a list
            of channels for each conv block. If `int`, it will be the same for all conv blocks. Default is `"auto"`.
        - `conv_kernel_size` (int|list): (list of) Kernel size of the convolution layers. Should be a list of integers,
            a list of tuples of integers (for conv2d or conv3d), or an integer. If it is a list, it MUST have the same 
            length as `num_conv_blocks`. If it is an integer, it will be the same for all conv blocks. Defaults to `3`.
        - `conv_padding` (int|str|list): (list of) Paddings of convolution layers. Format is as `conv_kernel_size`. Defaults to `"valid"`.
        - `conv_stride` (int|list): (list of) Strides of convolution layers. Format is as `conv_kernel_size`. Defaults to `1`.
        - `conv_dilation` (int|list): (list of) Dilations of convolution layers. Format is as `conv_kernel_size`. Defaults to `1`.
        - `conv_activation` (str|list): (list of) activations of the convolution blocks. Defaults to None.
            For each block, this entry can be an activation function ("relu", "sigmoid", "tanh", etc.), an activation layer 
            ("ReLU", "LeakyReLU", "Softmax", etc.), or a custom Layer class (not instance).
            `None` will assume no activation for the convolution layer.
        - `conv_activation_params` (dict|list): (list of) dicts for the convolution activation constructors. Defaults to None.
            This will be ignored if lower-case activation
            function names are provided. This is because in Keras, activation layers are classes with constructors, but activation functions are
            just functions, and have lower-case names as mentioned earlier. By the way, the slope of the negative section in `LeakyReLU` is `alpha`.
        - `conv_norm_layer_type` (str|list): (list of) types of normalization layers to use in the conv blocks. Examples: 'BatchNormalization',
            'LayerNormalization', etc. Defaults to None. Instead of a string, it can also be a custom Keras Layer class (not instance).
        - `conv_norm_layer_position` ("before"|"after"|None|list): (list of) positions of the normalization layers in the 
            convolutional blocks relative to the activation functions. Defaults to "before". If it is a list, it should be a list of strings of the same length as `num_conv_blocks`
        - `conv_norm_layer_params` (dict|list): (list of) kwargs dicts for the convolution normalization layers' constructors. Defaults to None.    
        - `conv_dropout` (float|list): (list of) Dropout rates of the convolution blocks. Defaults to None.
        - `conv_spatial_dropout` (float|list): (list of) Spatial dropout rates of the convolution blocks. Defaults to None.
        - `pool_type` (str|list): (list of) types of pooling layers. "Max", "Avg", "GlobalMax", and "GlobalAvg" are acceptable.
            Defaults to None, in which case there will be no pooling layer. Instead of the strings mentioned above, a custom Keras Layer class
            (not instance) can also be provided. In this case, one should note that the pooling
            hyperparameters below will be ignored, and the custom pooling layer will be constructed with the `pool_params` kwargs only.
            Also, in this case, image dimensions will not be calculated for this pooling layer because it is unknown since it is a custom layer.
        - `pool_kernel_size` (int|list): (list of) kernel sizes of the pooling layers, with similar format to `conv_kernel_size`.
            Again, it can be a list of integers, where every integer will be broadcasted to all dimensions, a list of tuples of integers for
            2D or 3D images, or an integer to be boradcasted across dimensions and conv blocks.
        - `pool_padding` (str|list): (list of) paddings of the pooling layers.
        - `pool_stride` (int|list): (list of) strides of the pooling layers.
        - `pool_params` (dict|list): (list of) kwargs dicts for the pooling layers' constructors.
        - `min_image_size` (int): Minimum size of the image to be reduced to in convolutions and poolings.
            After this point, the padding and striding will be chosen such that image size does not decrease further. Defaults to 1.
        - `flatten_before_dense` (bool): Whether to flatten the output of the convolutional blocks before the dense blocks.
            If it is False, all dense layers (if any) will be applied to all pixels, acting on channels. If this argument is not provided, the default will depend on whether there are dense layers after conv layers,
            in which case `dense_depth` would be greater than 0 and the default would be True, or not, in which case the default would be
            False since the absence of dense layers negates any reason for flattening. Defaults to None.
            
        #### Dense blocks
        
        - `dense_width` ("auto"|int|list): Width of the hidden layers of the Dense network. "auto", a number (for all of them) or a list
            holding width of each hidden layer.
            If "auto", it will start with the output size of the Flatten() layer, halving at every Dense block.
        - `dense_depth` (int): Depth (number of hidden layers) of the Dense network. `0` will mean no hidden layers,
            meaning Flatten will be directly followed by the output layer.
        - `dense_params` (dict): (list of) kwargs dict to pass to the dense layer constructor in each block. Defaults to None.
        - `dense_activation` (str|list): (list of) activation function for hidden layers of the Dense network.
            These can be activation functions ("relu", "sigmoid", "tanh", etc.),
            activation layers ("ReLU", "LeakyReLU", "Softmax", etc.), or custom Layer classes (not instances). Defaults to None,
            in which case no activation function will be used.
        - `dense_activation_params` (dict|list): (list of) dicts for the dense activation functions' constructors.
            Ignored if lower-case activation functions are provided.
            By the way, the slope of the negative section in `LeakyReLU` is `alpha`.
        - `dense_norm_layer_type` (str|list): (list of) types of normalization layers to use in the dense blocks. Examples:
            'BatchNormalization', 'LayerNormalization', etc.
            Defaults to None, in which case no normalization layer will be used. Instead of strings, custom Keras Layer classes (not instances)
            can also be provided.
        - `dense_norm_layer_position` ("before"|"after"|list): (list of) positions of the normalization layers in the dense blocks relative to the
            activation functions. Defaults to "before". If it is a list, it should be a list of strings of the same length as `dense_depth`.
        - `dense_norm_layer_params` (dict|list): (list of) kwargs dict for the dense normalization layers' constructors.
        - `dense_dropout` (float|list): (list of) Dropout rates (if any) for the hidden layers of the Dense network.
        - `include_output_layer` (bool): Whether to include an output layer. Defaults to True.
        - `output_dense_params` (dict): kwargs dict for the output layer's Dense constructor. Defaults to None.
        - `output_activation` (str): Activation (with the same format as dense activation) for the output layer, if any.
            **NOTE** Depending on the loss function, you may not need an activation function.
            For classification problems, you may want to choose "sigmoid" or "softmax".
            That being said, you usually don't need to specify an activation for the output layer at all, if e.g. 'from_logits' is used.
            For regression problems, no activation is needed. It is by default linear, unless you want to manually specify an activation.
        - `output_activation_params` (dict): Dictionary of parameters for the output activation function's constructor.
            Ignored if a lower-case function name is provided.
        
        #### Training procedure
        
        - `batch_size` (int): Minibatch size, the expected input size of the network.
        - `learning_rate` (float): Initial learning rate of training.
        - `exponential_decay_rate` (float): Exponential decay rate gamma for learning rate, if any.
        - `optimizer` (str): Optimizer. Examples: 'Adam', 'SGD', 'RMSprop', etc. It can be the name of any Keras optimizer class.
            This can also be a custom optimizer class (not instance), in which case `optimizer_params` can be specified for its constructor kwargs.
        - `optimizer_params` (dict): Additional parameters of the optimizer constructor, if any.
        - `epochs` (int): Maximum number of epochs for training.
        - `early_stopping_patience_epochs` (int): Epochs to tolerate unimproved (val) loss, before early stopping.
        - `l2_reg` (float): L2 regularization parameter.
        - `l1_reg` (float): L1 regularization parameter.
        - `loss_function` (str): Loss function. It can be a lower-case name such as "mse", "binary_crossentropy", "categorical_crossentropy", etc.,
            the name of a `tf.keras.losses` class such as `BinaryCrossentropy`, `CategoricalCrossentropy`, `MeanSquaredError', etc., or a valid loss
            class (not instance).
        - `loss_function_params` (dict): Additional kwargs parameters of the loss function constructor, if any. Ignored if the loss function is a
            lower-case name string.
        - `metrics` (list): list of metrics for Keras compilation. Each member of the list can be a lower-case metric name such as
            "mse" or "accuracy", the name of a `tf.keras.metrics` class such as `Accuracy`, `MeanSquaredError`, etc., or a valid metric class
            (not instance).
        - `metrics_params` (list): (list of) additional kwargs parameters of the metrics constructors, if any.
            Ignored for every metric that is a lower-case name string.
            If this entry is a single dicitonary rather than a list, it will be broadcast to all metrics in the metrics list.
        - `validation_data` (tuple): Validation data, if any. It should be a tuple of `(portion, from_dataset)`.
            For instance, `[0.05, 'testset']` means 5% of the testset will be used for validation.
            The second element of the tuple can only be `'trainset'` and `'testset'`. The first element must be a float between 0 and 1. 
            If the second element is not specified, testset will be used by default.
        - `checkpoint_path` (str): Path to the directory where checkpoints will be saved at every epoch.
        - `early_stopping_monitor` (str): Monitor whose critical value will cause early stopping. Default is 'loss', but 'val_loss' is typically used.
        - `early_stopping_mode` (str): Mode of the parameter whose critical value will be used for early stopping.
            Defaults to 'min' for any error. 'max' is for accuracy, etc.
        - `early_stopping_value` (float): Value of the monitor at which point training will stop because the critical value has been reached.
        - `other_callbacks` (list): List of other callbacks to be used during training, if any. Defaults to None.
        - `custom_schedule` (schedule): Custom learning rate schedule inheriting from `tf.keras.optimizers.schedules.LearningRateSchedule`.
            Defaults to None.
        
        
        ### Returns
        
        - Returns a `tf.keras.models.Model` object that can be trained and used accordingly.
        - Run `net.summary()` afterwards to see what you have inside the network.
        - A `KerasSmartModel` object is returned. This module has its own functions for training, evaluation, etc.
        """
        super(Conv_Network, self).__init__(hparams)
        if not hparams: hparams = self.sample_hparams
        # Input and output shapes
        self._model_name = hparams["model_name"] if "model_name" in hparams else "Conv_Network"
        self._input_shape = hparams["input_shape"]
        self._output_shape = hparams["output_shape"]
        self._N = int(hparams["batch_size"]) if "batch_size" in hparams else 32
        self.batch_input_shape = list(self._input_shape).copy()
        self.batch_input_shape.insert(0, self._N)
        self.batch_output_shape = list(self._output_shape).copy()
        self.batch_output_shape.insert(0, self._N)
        self.size_list = [self._input_shape]
    
        # Initialiaing sequential network
        self.net = tf.keras.models.Sequential(name=self._model_name)
        
        # Convolutional layers hyperparameters
        self._num_conv_blocks = hparams["num_conv_blocks"]
        self._conv_dim = hparams["conv_dim"]
        self._conv_params = hparams.get("conv_params")    
        self._conv_channels = hparams.get("conv_channels") if hparams.get("conv_channels") else "auto"
        self._conv_kernel_size = hparams.get("conv_kernel_size") if hparams.get("conv_kernel_size") else 3
        self._conv_padding = hparams["conv_padding"] if hparams.get("conv_padding") else "valid"
        self._conv_stride = hparams["conv_stride"] if hparams.get("conv_stride") else 1
        self._conv_dilation = hparams["conv_dilation"] if hparams.get("conv_dilation") else 1
        self._conv_activation = hparams.get("conv_activation")
        self._conv_activation_params = hparams.get("conv_activation_params")
        self._conv_norm_layer_type = hparams.get("conv_norm_layer_type")
        self._conv_norm_layer_position = hparams.get("conv_norm_layer_position")
        self._conv_norm_layer_params = hparams.get("conv_norm_layer_params")
        self._conv_dropout = hparams.get("conv_dropout")
        self._conv_spatial_dropout = hparams.get("conv_spatial_dropout")
        self._pool_type = hparams.get("pool_type")
        self._pool_kernel_size = hparams["pool_kernel_size"] if hparams.get("pool_kernel_size") else 2
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
        self._conv_spatial_dropout_vec = self._gen_hparam_vec_for_conv(self._conv_spatial_dropout, 'conv_spatial_dropout')
        
        # Constructing the encoder (convolutional blocks)
        # print("input_shape: ", self._input_shape)
        # in_channels = self._input_shape[-1]
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
                'spatial_dropout':self._conv_spatial_dropout_vec[i], 
                'min_image_dim':self._min_image_size,
                'kernel_regularizer':self.make_regularizer()
            }
            if i==0:
                _kwargs.update({'input_shape':self._input_shape})
            d = add_conv_block(**_kwargs)
            # self.net = d['model']
            output_image = d['output_image']
            self.size_list.append(output_image+[out_channels])
            # in_channels = out_channels
            input_image = output_image
            
        # Including Dense or Output layers
        self._dense_depth = hparams.get("dense_depth")
        self._include_dense_layers = hparams["include_dense_layers"] if "include_dense_layers" in hparams else \
            (self._dense_depth is not None and self._dense_depth > 0)
        self._include_output_layer = hparams["include_output_layer"] if "include_output_layer" in hparams else True
        
        # Flattening
        self._flatten_after_conv = hparams["flatten_after_conv"] if "flatten_after_conv" in hparams else (len(self._output_shape)==1)
        if self._flatten_after_conv:
            self.net.add(tf.keras.layers.Flatten())
            self._dense_input_size = np.prod(output_image) * out_channels
        else:
            self._dense_input_size = out_channels
        self.size_list.append([self._dense_input_size])
        
        # Construct dense layers
        if self._include_dense_layers:
            
            # Dense layers hyperparameters
            self._dense_width = hparams.get("dense_width")
            self._dense_params = hparams.get("dense_params")
            self._dense_activation = hparams.get("dense_activation")
            self._dense_activation_params = hparams.get("dense_activation_params")
            self._dense_norm_layer_type = hparams.get("dense_norm_layer_type")
            self._dense_norm_layer_params = hparams.get("dense_norm_layer_params")
            self._dense_norm_layer_position = hparams.get("dense_norm_layer_position")
            self._dense_dropout = hparams.get("dense_dropout")
            
            # Generate lists of hyperparameters for the dense layers
            self._dense_width_vec = self._gen_hparam_vec_for_dense(self._dense_width, 'dense_width',
                check_auto=True, init_for_auto=self._dense_input_size, powers_of_two_if_auto=True, direction_if_auto="down")
            self._dense_params_vec = self._gen_hparam_vec_for_dense(self._dense_params, 'dense_params')
            self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._dense_activation, 'dense_activation')
            self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._dense_activation_params, 'dense_activation_params')
            self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_type, 'dense_norm_layer_type')
            self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_params, 'dense_norm_layer_params')
            self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._dense_norm_layer_position, 'dense_norm_layer_position')
            self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._dense_dropout, 'dense_dropout')
            
            # Construct the dense layers
            # in_size = self._dense_input_size
            if self._dense_depth > 0:
                for i in range(self._dense_depth):
                    out_size = self._dense_width_vec[i]
                    _kwargs = {
                        'model':self.net,
                        'output_size':self._dense_width_vec[i],
                        'activation':self._dense_activation_vec[i],
                        'activation_params':self._dense_activation_params_vec[i],
                        'norm_layer_type':self._dense_norm_layer_type_vec[i],
                        'norm_layer_position':self._dense_norm_layer_position_vec[i],
                        'norm_layer_params':self._dense_norm_layer_params_vec[i],
                        'dropout': self._dense_dropout_vec[i],
                        'kernel_regularizer':self.make_regularizer(),
                        'dense_params':self._dense_params_vec[i]
                    }
                    add_dense_block(**_kwargs)
                    # in_size = out_size
                    self.size_list.append([out_size])
        
        # Construct output layer
        if self._include_output_layer:
            # Output hyperparameters
            self._output_dense_params = hparams.get("output_dense_params")
            self._output_activation = hparams.get("output_activation")
            self._output_activation_params = hparams.get("output_activation_params")
        
            # Output layer
            _kwargs = {'units':self._output_shape[-1], 'kernel_regularizer':self.make_regularizer()}
            if self._output_dense_params:
                _kwargs.update(self._output_dense_params)
            self.net.add(tf.keras.layers.Dense(**_kwargs))
            if self._output_activation:
                if isinstance(self._output_activation, str):
                    if self._output_activation.lower() == self._output_activation:
                        self.net.add(tf.keras.layers.Activation(self._output_activation))
                    elif self._output_activation_params:
                        self.net.add(getattr(tf.keras.layers, self._output_activation)(**self._output_activation_params))
                    else:
                        self.net.add(getattr(tf.keras.layers, self._output_activation)())
                elif self._output_activation_params:
                    self.net.add(self._output_activation(**self._output_activation_params))
                else:
                    self.net.add(self._output_activation())

    
    def _gen_hparam_vec_for_conv(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._num_conv_blocks, hparam_name=hparam_name, count_if_not_list_name='num_conv_blocks', **kwargs)
    
    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._dense_depth, hparam_name=hparam_name, count_if_not_list_name='dense_depth', **kwargs)
    
    def call(self, inputs, **kwargs):
        return self.net(inputs, **kwargs)





if __name__ == '__main__':
    # Test Conv_Network
    print("Minimal example with only convolution layers and no dense or output layers:")
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
        "conv_activation": "LeakyReLU",
        "conv_activation_params": {'alpha':0.1},
        "conv_norm_layer_type": "BatchNormalization",
        "conv_norm_layer_position": ["before", "after"],
        "conv_norm_layer_params": None,
        "conv_dropout": [0.1, None],
        "conv_spatial_dropout": None,
        "pool_type": "Max",
        "pool_kernel_size": 2,
        "pool_padding": 'valid',
        "pool_stride": 1,
        "pool_params": None,
        "min_image_size": 4
    }
    model = Conv_Network(sample_hparams)
    model.summary()
    print("Trying with sample data ...")
    y = model(tf.random.normal([32, 28, 28, 3]))
    print("Output shape:", y.shape)
    
    
    print("")
    print("Testing full example with convolution layers, dense layers and output layer:")
    test_keras_model_class(Conv_Network)
    