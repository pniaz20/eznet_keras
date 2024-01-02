if __package__=="eznet_keras.models":
    from .keras_smart_module import *
    from .dense_block import *
else:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from keras_smart_module import *
    from dense_block import *

import warnings

class ANN(KerasSmartModel):
    
    sample_hparams = {
        # General and I/O parameters
        "model_name": "ANN",
        "input_shape": [10],
        "output_size": 3,
        # Architecture parameters
        "width": 32,
        "depth": 2,
        "hidden_params": None,
        "hidden_activation": "LeakyReLU",
        "hidden_activation_params": {'alpha': 0.1},
        "norm_layer_type":"BatchNormalization",
        "norm_layer_position": "before",
        "norm_layer_params": None,
        "dropout": 0.2,
        "include_output_layer": True,
        "output_dense_params": None,
        "output_activation": "Softmax",
        "output_activation_params": None,
        # Training procedure parameters
        "learning_rate": 0.001,
        "exponential_decay_rate": 0.9,
        "batch_size": 32,
        "epochs": 2,
        "early_stopping_patience_epochs": 2,
        "validation_data": [0.05,'trainset'],
        "l2_reg": 0.0001,
        "l1_reg": None,
        "loss_function": "categorical_crossentropy",
        "loss_function_params": None,
        'optimizer': 'Adam',
        'optimizer_params': None,
        'metrics': ['accuracy'],
        "metrics_params": None,
        'checkpoint_path': None,
        'early_stopping_monitor': 'loss',
        'early_stopping_mode': 'min',
        'early_stopping_value': 1.0e-6,
        'other_callbacks': None,
        'custom_schedule': None
    }
    
    
    def __init__(self, hparams:dict=None):
        """
        Typical Artificial Neural Network class, also known as multilayer perceptron. This class will create a fully connected feedforward
        artificial neural network.
        It can be used for classification, regression, etc. It basically encompasses enough options to build all kinds of serial ANNs with any number
        of inputs, outputs, layers with custom or arbitrary width or depth, etc. Supports multiple activation functions for hidden layers and the
        output layer. The netwrok consists of many `Dense_Block` layers, each consisting of a linear (dense) layer, followed optionally by a
        normalization, activation, and dropout layers. There is optionally an output layer after all the `Dense_Block` instances.
        This class can be used for making virtually any kind of serial MLP. No parallelism or skip connections are supported in this class yet.
        
        ### Usage
        
        `net = ANN(hparams)` where `hparams` is the dictionary of hyperparameters.
        
        - Inspect the `sample_hparams` class attribute to find a complete sample of hyperparameters dictionary.
        - The training procedure section of the hyperparameters is the same as in `KerasSmartModel` class, and is only used in training.
          Otherwise, it is not necessary for building the model itself.
        - Many hyperparameters, especially those that have a (list of) in their description below, can be either None,
          in which case that item will not be included, or a scalar value which will be broadcasted to all dense blocks, or a list with the same
          length as the `depth` hyperparameter, in which case each item will be used for the corresponding dense block.
          Use None for any dense block for which an item or hyperparameter should not be included or applied.
        - Every `Dense_Block` by default is assumed to have a dense layer, followed by a normalization, activation, and dropout layers.
          We also assume that there will be an output layer at the end, unless `include_output_layer` hyperparameter is set to `False`.
          Normalization layer can come `before` or `after` the activation.
        - Activation functions come from `tf.keras.activations` and need to be wrapped in a `tf.keras.layers.Activation` layer,
          and do not accept any kwargs. Activation layers come from `tf.keras.layers`, can accept kwargs in their constructors, and do not need to be
          wrapped in an `Activation` layer. Activation functions all have lower-case names, but activation layers are classes and have
          every-word-capitalized names. We will use this as a clue to recognize what the user wants, and perform accordingly.
          Most activation layers also have correpsonding activation functions (like `tf.keras.activations.relu` function and
          `tf.keras.layers.ReLU` layer). However, some activations are only available as functions (such as `tf.keras.activations.sigmoid`),
          and some are only available as layers (such as `tf.keras.layers.LeakyReLU`). Choose accordingly.

        The hyperparameters dictionary should include the following keys:
        
            - `input_shape` (list): shape of the input layer minus the batch size. It can have any number of elements.
                In higher dimensionalities the dense layer will be applied to all feature maps, such that only the last dimension will change.
                An input of `[batch, dim1, dim2, in_dim]` will come out as `[batch, dim1, dim2, layer_size]` from a dense block.
            - `output_size` (int): number of outputs to predict, i.e. size of the output layer, if there is going to be an output layer.
            - `width` (int|list): (list of) hidden layer widths. 
                A number sets them all the same, and a list/array sets each hidden layer according to the list.
            - `depth` (int): Specifies the depth of the network (number of hidden layers).
                It must be specified unless `width` is provided as a list. Then the depth will be inferred form it.
            - `hidden_params` (dict): (list of) kwargs parameters for the hidden layer constructor, if any. Defaults to None.
                It will overwrite everything else if specified.
            - `hidden_activation` (str): (list of) Activations. It can be an activation function name ("relu","sigmoid","tanh", etc.),
                an activation layer name ("ReLU", "LeakyReLU", "Softmax", etc.), or a custom Keras Layer class (not instance). Defaults to None.
            - `hidden_activation_params` (dict): (list of) kwargs parameters for the hidden layer activation constructor, if any.
                Ignored for any dense block if the activation is provided as a lower-case function name. By the way, the slope of the negative
                section in `LeakyReLU` is `alpha`. Defaults to None.
            - `norm_layer_type` (str): (list of) Types of normalization layers to use for each hidden layer.
                Options are "BatchNormalization", "LayerNormalization", etc.
                This can also be a Keras Layer class (not instance) rather than a string. Defaults to None.
            - `norm_layer_position` (str): (list of) where the normalization layer should be included relative to the activation function (if any),
                'before' or 'after'.
            - `norm_layer_params` (dict): (list of) Dictionaries of kwargs parameters for the normalization layers' constructors. Defaults to None.
            - `dropout` (float): (list of) the dropout rates after every hidden layer.
                It should be a probability value between 0 and 1, or None by default.
            - `include_output_layer` (bool): Whether to include an output layer at the end of the network. Defaults to True.
            - `output_dense_params` (dict): Parameters for the output dense layer, if any. Defaults to None.
                It will overwrite everything else if specified.
            - `output_activation` (str): Activation of the output layer, if any. Defaults to None.
                For classification problems, you may want to choose "sigmoid" or "softmax".
                That being said, you usually don't need to specify an activation for the output layer at all, if e.g. 'from_logits' is used.
                For regression problems, no activation is needed. It is by default linear, unless you want to manually specify an activation.
            - `output_activation_params` (dict): Parameters for the output activation function, if any.
                Ignored if lower-case function name is provided for activation. Defaults to None.
            - `learning_rate` (float): Initial learning rate of training.
            - `exponential_decay_rate` (float): Exponential decay rate for learning rate, if any. Defaults to None.
            - `optimizer` (str): Optimizer. Examples: "Adam", "SGD" ,"RMSProp", etc. The name of any optimizer class in `tf.keras.optimizers`
                can be used, or a custom class. This can also be a custom optimizer class (not instance), in which case `optimizer_params` can be
                specified for its constructor kwargs.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any. Defaults to None.
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
            - `early_stopping_patience_epochs` (int): Epochs to tolerate unimproved (val) loss, before early stopping.
            - `validation_data` (list): List of [validation_split, 'trainset'|'testset']. Defaults to None.
            - `l2_reg` (float): L2 regularization parameter. Defaults to None.
            - `l1_reg` (float): L1 regularization parameter. Defaults to None.
            - `loss_function` (str): Loss function. It can be a lower-case name such as "mse", "binary_crossentropy",
                "categorical_crossentropy", etc., the name of a `tf.keras.losses` class such as `BinaryCrossentropy`, `CategoricalCrossentropy`,
                `MeanSquaredError`, etc., or a valid loss class (not instance).
            - `loss_function_params` (dict): Additional kwargs parameters of the loss function constructor, if any.
                Ignored if the loss function is a lower-case name string.
            - `metrics` (list): list of metrics for Keras compilation. Each member of the list can be a lower-case metric name such as
                "mse" or "accuracy", the name of a `tf.keras.metrics` class such as `Accuracy`, `MeanSquaredError`, etc., or a valid metric
                class (not instance).
            - `metrics_params` (list): (list of) additional kwargs parameters of the metrics constructors, if any.
                Ignored for every metric that is a lower-case name string. If this entry is a single dicitonary rather than a list,
                it will be broadcast to all metrics in the metrics list.
            - `checkpoint_path` (str): Path to the directory where checkpoints will be saved at every epoch.
            - `early_stopping_monitor` (str): Monitor whose critical value will cause early stopping. Default is 'loss',
                but 'val_loss' is typically used.
            - `early_stopping_mode` (str): Mode of the parameter whose critical value will be used for early stopping.
                Defaults to 'min' for any error. 'max' is for accuracy, etc.
            - `early_stopping_value` (float): Value of the monitor at which point training will stop becasue the critical value has been reached.
            - `other_callbacks` (list): List of other callbacks to be used in training, if any. Defaults to None.
            - `custom_schedule` (schedule): Custom learning rate schedule inheriting from `tf.keras.optimizers.schedules.LearningRateSchedule`.
                Defaults to None.
            
        
        Note that for all such hyperparameters that have a (list of) at the beginning, the entry can be a single item repeated for all hidden layers,
        or it can be a list of items for all hidden layers. If a list is provided, it must have the same length as the depth of the network. 
        
        Also note that depth does not include the input and output layers.
        This gives you the ability to specify different width, dropout rate, normalization layer and its parameters, and so forth.

        ### Returns
        
        It returns a `tf.keras.models.Model` object that corresponds with an ANN model.
        run `net.summary()` afterwards to see what the ANN holds.
        The returned module is a `KerasSmartModel` instance, which is a subclass. It has built-in functions for training, evaluation, etc.
        """
        super(ANN, self).__init__(hparams)
        # Read and store hyperparameters
        if not hparams: hparams=self.sample_hparams
        self._model_name = hparams.get("model_name") if "model_name" in hparams else "ANN"
        self._input_shape = hparams.get("input_shape")
        if self._input_shape is not None:
            assert isinstance(self._input_shape, (list,tuple)), "input_shape must be a list or tuple."
        self._output_size = hparams.get("output_size")
        self._dropout = hparams.get("dropout")
        self._width = hparams.get("width")
        self._depth = hparams.get("depth")
        self._hidden_params = hparams.get("hidden_params")
        if self._depth is None:
            assert isinstance(self._width, (list,tuple)), "If depth is not provided, width must be a list or tuple."
            self._depth = len(self._width)
        elif self._width is None:
            assert isinstance(self._depth, int), "If width is not provided, depth must be an integer."
            warnings.warn("Width is not provided. It will be set to the same value as the final dimension of the input shape.", UserWarning)
            self._width = self._input_shape[-1]
        elif isinstance(self._width, (list,tuple)):
            assert len(self._width) == self._depth, "If width is a list or tuple, it must have length equal to the depth."
            
            
        self._hidden_activation = hparams.get("hidden_activation")
        self._hidden_activation_params = hparams.get("hidden_activation_params")
        self._include_output_layer = hparams.get("include_output_layer")
        self._output_dense_params = hparams.get("output_dense_params")
        self._output_activation = hparams.get("output_activation")
        self._output_activation_params = hparams.get("output_activation_params")
        self._norm_layer_type = hparams.get("norm_layer_type")
        self._norm_layer_position = hparams.get("norm_layer_position")
        self._norm_layer_params = hparams.get("norm_layer_params")
        self.batch_input_shape = (self._batch_size,) + tuple(self._input_shape)
        self.batch_output_shape = (self._batch_size,) + tuple(self._input_shape[:-1]) + (self._output_size,)
        if self._output_activation is not None:
            if isinstance(self._output_activation, str):
                if self._output_activation.lower()==self._output_activation:
                    self._output_activation_module = getattr(tf.keras.activations, self._output_activation)
                else:
                    self._output_activation_module = getattr(tf.keras.layers, self._output_activation)
            else:
                self._output_activation_module = self._output_activation
        else:
            self._output_activation_module = None
        
        # Generate arrays containing parameters of each Dense Block (Every block contains a linear, normalization, activation, and dropout layer).
        self._dense_width_vec = self._gen_hparam_vec_for_dense(self._width, 'width')
        self._dense_params_vec = self._gen_hparam_vec_for_dense(self._hidden_params, 'hidden_params')
        self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._hidden_activation, 'hidden_activation')
        self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._hidden_activation_params, 'hidden_activation_params')
        self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._norm_layer_type, 'norm_layer_type')
        self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._norm_layer_params, 'norm_layer_params')
        self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._norm_layer_position, 'norm_layer_position')
        self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._dropout, 'dropout')
        
        # Initialize sequential model
        self.net = tf.keras.models.Sequential(name=self._model_name)
        
        # Construct the dense layers
        # in_size = self._input_shape
        for i in range(self._depth):
            # out_size = self._dense_width_vec[i]
            _kwargs = {
                'model':self.net,
                'output_size':self._dense_width_vec[i],
                'activation':self._dense_activation_vec[i],
                'activation_params':self._dense_activation_params_vec[i],
                'norm_layer_type':self._dense_norm_layer_type_vec[i],
                'norm_layer_position':self._dense_norm_layer_position_vec[i],
                'norm_layer_params':self._dense_norm_layer_params_vec[i],
                'dropout':self._dense_dropout_vec[i],
                'kernel_regularizer':self.make_regularizer(),
                'dense_params':self._dense_params_vec[i],
            }
            if i==0 and self._input_shape is not None:
                _kwargs.update({'input_shape':self._input_shape})
            add_dense_block(**_kwargs)
            # in_size = out_size
        
        # Output layer
        if self._include_output_layer:
            _kwargs = {'units':self._output_size, 'kernel_regularizer':self.make_regularizer()}
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

    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._depth, hparam_name=hparam_name, count_if_not_list_name='depth', **kwargs)
        
    def call(self, x, **kwargs):
        return self.net(x, **kwargs)





if __name__ == '__main__':
    
    hparams = {
        # General and I/O parameters
        "model_name": "My_Model",
        "input_shape": [28, 28, 3], #[10],
        "output_size": 24,
        # Architecture parameters
        "width": 32,
        "depth": 3,
        "hidden_activation": "LeakyReLU",
        "hidden_activation_params": {'alpha':0.1},
        "norm_layer_type":"BatchNormalization",
        "norm_layer_position": "before",
        "norm_layer_params": None,
        "dropout": 0.2,
        "include_output_layer": True,
        "output_activation": "Softmax",
        "output_activation_params": None,
        # Training procedure parameters
        "learning_rate": 0.001,
        "exponential_decay_rate": 0.9,
        "batch_size": 32,
        "epochs": 2,
        "early_stopping_patience_epochs": 2,
        "validation_data": [0.05,'trainset'],
        "l2_reg": 0.0001,
        "loss_function": "categorical_crossentropy",
        'optimizer': 'Adam',
        'optimizer_params': None,
        'metrics': ['accuracy'],
        'checkpoint_path': None,
        'early_stopping_monitor': 'loss',
        'early_stopping_mode': 'min',
        'early_stopping_value': 1.0e-6
    }
    test_keras_model_class(ANN, hparams, False)
    