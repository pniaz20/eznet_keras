if __package__=="eznet_keras.models":
    from ..utils import *
    from .keras_smart_module import *
    from .dense_block import *
else:
    import os, sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    sys.path.append(current_dir)
    from utils import *
    from keras_smart_module import *
    from dense_block import *

import warnings

class Recurrent_Network(KerasSmartModel):
    
    sample_hparams = {
        # General and I/O parameters
        'model_name': 'Recurrent_Network',
        'in_features': 10,
        'out_features': 3,
        'sequence_length': 13,
        'final_rnn_return_sequences': False,
        'flatten_after_rnn': False,
        'permute_output': False,
        # RNN layer parameters
        'rnn_type': 'LSTM',
        'rnn_hidden_size': 20,
        'rnn_bidirectional': False,
        'rnn_depth': 2,
        'rnn_input_dropout': None,
        'rnn_recurrent_dropout': None,
        'rnn_layer_dropout': 0.1,
        'rnn_params': None,
        # Dense layer parameters
        'include_dense_layers': True,
        'dense_width': 30,
        'dense_depth': 2,
        'dense_params': None,
        'dense_dropout': 0.2,
        'dense_activation': 'relu',
        'dense_activation_params': None,
        'norm_layer_type': 'BatchNormalization',
        'norm_layer_params': None,
        'norm_layer_position': 'before',
        # Output layer parameters
        'include_output_layer': True,
        'output_dense_params': None,
        'output_activation': "softmax",
        'output_activation_params': None,
        # Training procedure parameters
        'l2_reg': 0.0001,
        'l1_reg': None,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': [0.05,'testset'],
        'early_stopping_patience_epochs': 10,
        'learning_rate': 0.001,
        'exponential_decay_rate': 0.95,
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
        """Sequence to Dense network with RNN for time-series classification, regression, and forecasting, as well as NLP applications.
        This network uses any RNN layers as encoders to extract information from input sequences, and (optionally) fully-connected 
        multilayer perceptrons (Dense) to decode the sequence into an output, which can be class probabilitites (timeseries classification),
        a continuous number (regression), or an unfolded sequence (forecasting) of a target timeseries. Multiple stacked RNN layers are supported,
        as well as bidirectional RNNs. Furthermore, the dense blocks in the decoder part are by default assumed to contain a dense layer, followed
        (optionally) by a normalization, activation, and dropout layer. Multiple activation functions, loss functions, optimizers, etc. are
        supported, meaning this class can be used to construct virtually any serial RNN network possible, with or without dense layers, with or
        without output layers. In this class, parallelization or skip connections are not yet supported.
        The RNN section supports input dropout, recurrent dropout, and layer dropout (dropout between two stacked layers of RNN, for example).

        ### Usage

        `net = Recurrent_Network(hparams)` where `hparams` is dictionary of hyperparameters containing the keys described below.
        
        - Inspect the `sample_hparams` class attribute for a template of the hyperparameters dictionary.
        - Many keys, especially those that have "(list of)" in their description, can be either scalar items or lists of items.
          If they are scalar, they will be broadcasted to all RNN/Dense blocks. If they are lists, they must have the same length as the number
          (depth) of RNN/Dense blocks, i.e. `rnn_depth`/`dense_depth` hyperparameters.
        - The Training Procedure section of the hyperparameters are optional, and will only be used by the parent class `KerasSmartModel`
          if the model is trained using its methods. Otherwise, it is not necessary at all for building a network.
        - Activation functions come from `tf.keras.activations` and need to be wrapped in a `tf.keras.layers.Activation` layer,
          and do not accept any kwargs. Activation layers come from `tf.keras.layers`, can accept kwargs in their constructors, and do not need to be
          wrapped in an `Activation` layer. Activation functions all have lower-case names, but activation layers are classes and have
          every-word-capitalized names. We will use this as a clue to recognize what the user wants, and perform accordingly.
          Most activation layers also have correpsonding activation functions (like `tf.keras.activations.relu` function and `tf.keras.layers.ReLU`
          layer). However, some activations are only available as functions (such as `tf.keras.activations.sigmoid`), and some are only available as
          layers (such as `tf.keras.layers.LeakyReLU`). Choose accordingly.
        
        The keys of the hyperparameter dictionary are as follows:
        
        #### General and I/O hyperparameters
        
            - `model_name` (str): Name of the model, can be used later for saving, etc.
            - `in_features` (int): Number of features of the input.
            - `out_features` (int): Number of features of the output, assuming there is an output layer
                (including a dense layer and an activation function)
            - `sequence_length` (int): Length of the sequence.
            - `final_rnn_return_sequences` (bool): Whether the final RNN returns a whole sequence rather than its last time step. Defaults to False.
                This is useful if a sequence-to-sequence architecture is intended, or all information of all time steps of the final RNN layer is
                desired.
            - `flatten_after_rnn` (bool): If `final_rnn_return_sequences` is True, whether to flatten the final RNN output sequence into a single
                feature vector. This way, all information of all time steps of the final RNN layer will be preserved, and the 3D output of the final
                RNN layer will be flattened to a 2D tensor. If `final_rnn_return_sequences` is False, this parameter is ignored. Defaults to False.
            - `permute_output` (bool): If `'final_rnn_return_sequences' = True` and `'flatten_after_rnn' = False`, whether to permute the output
                sequence to be `(batch, features, sequence_length)`. Default is False.
            
            
        #### RNN architecture hyperparameters
        
            - `rnn_type` (str|list): (list of) RNN types, options are "LSTM", "GRU", "SimpleRNN", etc.
                They should be class names or custom RNN layer classes (not instances).
            - `rnn_hidden_size` (int|list): (list of) RNN layer hidden sizes. Default is 16.
            - `rnn_bidirectional` (bool): Whether the RNN layers are bidirectional or not. Default is False.
            - `rnn_depth` (int): Number of stacked RNN layers. Default is 1.
            - `rnn_input_dropout` (float|list): (list of) Dropout rates, if any, of the RNN layer inputs. 
                Please note that using dropout in RNN layers is generally discouraged, for it decreases determinism during inference.
            - `rnn_recurrent_dropout` (float|list): (list of) Dropout rates, if any, of the RNN layer recurrent states.
            - `rnn_layer_dropout` (float|list): (list of) Dropout rates, if any, of the RNN layer outputs.
                This will be applied to all time steps equally.
            - `rnn_params` (dict): A single dictionary of kwargs for the RNN layers' constructors. Default is None. If specified, the keys in this
            dictionary will not only add to, but also overwrite any existing arguments this class passes to the RNN layer constructor.
            
            
        #### Dense network architecture hyperparameters
        
            - `include_dense_layers` (bool): Whether to include a Dense network after the RNN layers. Default is True.
            - `dense_width` (int|list): (list of) Widths of the Dense network. It can be a number (for all) or a list holding width of each hidden
                layer.
            - `dense_depth` (int): Depth (number of hidden layers) of the Dense network, not including the output layer.
            - `dense_params` (dict): (list of) dictionaries of kwargs for the Dense layers' constructors. Default is None.
            - `dense_activation` (str|list): (list of) Activation functions for hidden layers of the Dense network.
                It can be activation function ("relu", "sigmoid", "softmax", etc.), activation layer
                ("ReLU", "LeakyReLU", "Softmax", etc.), or a custom Keras layer class (not instance).
            - `dense_activation_params` (dict|list): (list of) Dictionaries of parameters for the activation function constructors of the Dense
                network. Ignored if the activation function is a lower-case function name. By the way, for `LeakyReLU`, the left-hand slope is 
                specified by the `alpha` key.
            - `norm_layer_type` (str|list): (list of) Types of normalization layers to use in the dense section, if any. 
                Options are "BatchNormalization", "LayerNormalization", etc. It can be a layer name or a Keras Layer class (not instance).
            - `norm_layer_params` (dict|list): (list of) Dictionaries of parameters for the normalization layer constructors.
            - `norm_layer_position` (str|list): (list of) Whether the normalization layer should come 'before' or 'after' the activation of each
                hidden layer in the dense network.
            - `dense_dropout` (float|list): (list of) Dropout rates (if any) for the hidden layers of the Dense network.

        #### Output layer hyperparameters
        
            - `include_output_layer` (bool): Whether to include an output layer, regardless of whether there was a Dense network. Default is True.
                The output layer is a dense layer followed optionally by an activation.
            - `output_dense_params` (dict): Dictionary of kwargs for the output layer's Dense constructor, if any. Defaults to None.
            - `output_activation` (str): Activation function for the output layer of the Dense network, if any.
                Like the `dense_activation` key, it can be activation function name, layer name, or a custom Keras Layer class (not instance).
                **NOTE** If the `loss_function` is `sparse_categorical_crossentropy`, then no output activation is erquired.
                However, if it is `categorical_crossentropy` (a.k.a. negative log-likelihood), then you must specify an output activation as in
                "softmax".
            - `output_activation_params` (dict): Dictionary of parameters for the activation function constructor of the output layer.
            
        #### Training procedure hyperparameters
        
            - `batch_size` (int): Minibatch size, the expected input size of the network. Defaults to 32.
            - `learning_rate` (float): Initial learning rate of training. Defaults to 0.001. Will be given directly to the optimization function.
            - `exponential_decay_rate` (float): Exponential decay rate for learning rate, if any.
            - `optimizer` (str): Optimizer. Examples: 'Adam', 'SGD', 'RMSProp', etc. It can be the name of any Keras optimizer class.
                This can also be a custom optimizer class (not instance), in which case `optimizer_params` can be specified for its constructor
                kwargs.
            - `optimizer_params` (dict): Additional parameters of the optimizer constructor, if any. Defaults to None.
            - `epochs` (int): Maximum number of epochs for training. Defaults to 2.
            - `early_stopping_patience_epochs` (int): Epochs to tolerate unimproved (val) loss, before early stopping. Defaults to None.
            - `validation_data` (list): Portion of validation data. Should be a tuple like [validation split, dataset as in 'trainset' or
                'testset']. Defaults to None.
            - `l2_reg` (float): L2 regularization parameter. Defaults to None.
            - `l1_reg` (float): L1 regularization parameter. Defaults to None.
            - `loss_function` (str): Loss function. It can be a lower-case name such as "mse", "binary_crossentropy",
                "categorical_crossentropy", etc., the name of a `tf.keras.losses` class such as `BinaryCrossentropy`, `CategoricalCrossentropy`, 
                `MeanSquaredError`, etc., or a valid loss class (not instance).
            - `loss_function_params` (dict): Additional kwargs parameters of the loss function constructor, if any.
                Ignored if the loss function is a lower-case name string.
            - `metrics` (list): list of metrics for Keras compilation. Each member of the list can be a lower-case metric name such as
                "mse" or "accuracy", the name of a `tf.keras.metrics` class such as `Accuracy`, `MeanSquaredError`, etc.,
                or a valid metric class (not instance).
            - `metrics_params` (list): (list of) additional kwargs parameters of the metrics constructors, if any.
                Ignored for every metric that is a lower-case name string.
                If this entry is a single dicitonary rather than a list, it will be broadcast to all metrics in the metrics list.
            - `checkpoint_path` (str): Path to the directory where checkpoints will be saved at every epoch.
                The path does not need to exist beforehand.
            - `early_stopping_monitor` (str): Monitor whose critical value will cause early stopping.
                Default is 'loss', but 'val_loss' is typically used.
            - `early_stopping_mode` (str): Mode of the parameter whose critical value will be used for early stopping.
                Defaults to 'min' for any error. 'max' is for accuracy, etc.
            - `early_stopping_value` (float): Value of the monitor at which point training will stop becasue the critical value has been reached.
            - `other_callbacks` (list): List of other callbacks to be used during training. Defaults to None.
            - `custom_schedule` (schedule): Custom learning rate schedule inheriting from
                `tf.keras.optimizers.schedules.LearningRateSchedule`. Defaults to None.

        ### Returns
        
        Returns a `tf.keras.models.Model` object that can be trained and used accordingly.
        Run `net.summary()` afterwards to see what you have inside the network.
        The returned model is an instance of `KerasSmartModel`, which has built-in functions for training, evaluation, etc.
        """
        super(Recurrent_Network, self).__init__(hparams)
        hparams = hparams if hparams is not None else self.sample_hparams
        
        # Read General and I/O hyperparameters
        self._model_name = hparams.get("model_name") if "model_name" in hparams else "Recurrent_Network"
        self._in_features = hparams["in_features"]
        self._out_features = hparams.get("out_features")
        self._L = int(hparams["sequence_length"])
        self._final_rnn_return_sequences = True if hparams.get("final_rnn_return_sequences") else False
        self._flatten_after_rnn = True if hparams.get("flatten_after_rnn") else False
        if self._flatten_after_rnn:
            assert self._final_rnn_return_sequences, "If `flatten_after_rnn` is True, then `final_rnn_return_sequences` must be True as well. Check your hyperparameters."
        self._permute_output = True if hparams.get("permute_output") else False
        if self._permute_output:
            assert self._final_rnn_return_sequences and not self._flatten_after_rnn, \
                "If `permute_output` is True, then `final_rnn_return_sequences` must be True and `flatten_after_rnn` must be False. Check your hyperparameters."
            
        # Read RNN hyperparameters
        self._rnn_type = hparams["rnn_type"]
        self._rnn_params = hparams.get("rnn_params")
        self._rnn = getattr(tf.keras.layers, self._rnn_type) if isinstance(self._rnn_type, str) else self._rnn_type
        self._rnn_hidden_size = hparams["rnn_hidden_size"] if hparams.get("rnn_hidden_size") else 16
        self._rnn_depth = hparams["rnn_depth"] if hparams.get("rnn_depth") else 1
        self._bidirectional = True if hparams.get("rnn_bidirectional") else False
        self._rnn_input_dropout = hparams.get("rnn_input_dropout")
        self._rnn_recurrent_dropout = hparams.get("rnn_recurrent_dropout")
        self._rnn_layer_dropout = hparams.get("rnn_layer_dropout")        
            
        # Read Dense hyperparameters
        self._include_dense_layers = hparams.get("include_dense_layers") if "include_dense_layers" in hparams else \
            (hparams.get('dense_depth') is not None and hparams.get('dense_depth') > 0)
        self._dense_width = hparams["dense_width"]
        self._dense_depth = hparams["dense_depth"] if hparams.get("dense_depth") else 0
        self._dense_params = hparams.get("dense_params")
        self._dense_activation = hparams.get("dense_activation")
        self._dense_activation_params = hparams.get("dense_activation_params")
        self._norm_layer_type = hparams.get("norm_layer_type")
        self._norm_layer_params = hparams.get("norm_layer_params")
        self._norm_layer_position = hparams.get("norm_layer_position")
        self._dense_dropout = hparams.get("dense_dropout")
        
        # Read Output hyperparameters
        self._include_output_layer = hparams.get("include_output_layer") if "include_output_layer" in hparams else True
        if not self._include_output_layer:
            assert self._out_features is None, "If `include_output_layer` is False, then `out_features` must be None. Check your hyperparameters."
        self._output_dense_params = hparams.get("output_dense_params")
        self._output_activation = hparams.get("output_activation")
        self._output_activation_params = hparams.get("output_activation_params")
        
        # Calculate shape-related variables
        self._N = int(self._batch_size)
        self._D = int(2 if self._bidirectional else 1)
        self._H_in = int(self._in_features)
        self._H_cell = self._rnn_hidden_size
        self._H_out = self._H_cell # Projection is not supported in Keras, unlike PyTorch.
        
        # Calculate batch input shape
        self.batch_input_shape = (self._N, self._L, self._H_in)
        
        # Calculate width (number of features) of the output layer (final layer, whatever that is)
        if self._include_output_layer:
            self._output_width = self._out_features
        elif self._include_dense_layers:
            self._output_width = self._dense_width if isinstance(self._dense_width, int) else self._dense_width[-1]
        else:
            self._output_width = (self._L if self._flatten_after_rnn else 1) * self._D * self._H_out
        
        # Calculate batch output shape
        self._output_is_sequence = (self._final_rnn_return_sequences and not self._flatten_after_rnn)
        if self._output_is_sequence:
            self.batch_output_shape = (self._N, self._output_width, self._L) if self._permute_output else (self._N, self._L, self._output_width)
        else:
            self.batch_output_shape = (self._N, self._output_width)
        
        # Initializing sequential network
        self.net = tf.keras.models.Sequential(name=self._model_name)
        self.net.add(tf.keras.Input((self._L, self._H_in)))
        
        # Generate vectors out of scalar hyperparameters
        vectorify = lambda x: x if isinstance(x, (list, tuple)) else [x] * self._rnn_depth
        self._H_cell_vec = vectorify(self._H_cell)
        self._rnn_input_dropout_vec = vectorify(self._rnn_input_dropout)
        self._rnn_recurrent_dropout_vec = vectorify(self._rnn_recurrent_dropout)
        self._rnn_layer_dropout_vec = vectorify(self._rnn_layer_dropout)
        
        # Construct RNN layers
        for i in range(self._rnn_depth):
            _kwargs = {
                'units': self._H_cell_vec[i],
                'dropout': self._rnn_input_dropout_vec[i] if self._rnn_input_dropout_vec[i] is not None else 0.0,
                'recurrent_dropout': self._rnn_recurrent_dropout_vec[i] if self._rnn_recurrent_dropout_vec[i] is not None else 0.0,
                'return_sequences':(True if self._rnn_depth > 1 else self._final_rnn_return_sequences), 
                'kernel_regularizer':self.make_regularizer()
            }
            if self._rnn_params is not None: 
                _kwargs.update(self._rnn_params)
            if i != self._rnn_depth-1:
                _kwargs.update({'return_sequences':True})
            else:
                _kwargs.update({'return_sequences':self._final_rnn_return_sequences})
            self.net.add(tf.keras.layers.Bidirectional(self._rnn(**_kwargs)) if self._bidirectional else self._rnn(**_kwargs))
            if self._rnn_layer_dropout_vec[i] is not None:
                self.net.add(tf.keras.layers.Dropout(self._rnn_layer_dropout_vec[i], 
                    noise_shape=((self._N, 1, self._H_cell_vec[i] * self._D) if self._final_rnn_return_sequences else None)))
        
        # Flatten the output sequence before decoding if necessary
        if self._flatten_after_rnn:
            self.net.add(tf.keras.layers.Flatten())
        
        # Construct Dense Network
        if self._include_dense_layers and self._dense_depth > 0:
                
            # Generate arrays containing parameters of each Dense Block (Every block contains a linear, normalization, activation, and dropout layer).
            self._dense_width_vec = self._gen_hparam_vec_for_dense(self._dense_width, 'dense_width')
            self._dense_params_vec = self._gen_hparam_vec_for_dense(self._dense_params, 'dense_params')
            self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._dense_activation, 'dense_activation')
            self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._dense_activation_params, 'dense_activation_params')
            self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._norm_layer_type, 'norm_layer_type')
            self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._norm_layer_params, 'norm_layer_params')
            self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._norm_layer_position, 'norm_layer_position')
            self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._dense_dropout, 'dense_dropout')
            
            # Construct the dense layers
            # in_size = self._dense_input_size
            for i in range(self._dense_depth):
                out_size = self._dense_width_vec[i]
                add_dense_block(self.net, output_size=out_size, input_shape=None, activation=self._dense_activation_vec[i], activation_params=self._dense_activation_params_vec[i], 
                        norm_layer_type=self._dense_norm_layer_type_vec[i], norm_layer_position=self._dense_norm_layer_position_vec[i], 
                        norm_layer_params=self._dense_norm_layer_params_vec[i], dropout=self._dense_dropout_vec[i], 
                        kernel_regularizer=self.make_regularizer(), dense_params=self._dense_params_vec[i])
                # in_size = out_size
        
        # Output layer
        if self._include_output_layer:
            _kwargs = {'units':self._output_width, 'kernel_regularizer':self.make_regularizer()}
            if self._output_dense_params is not None:
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
            
        # Permute if necessary
        if self._permute_output:
            self.net.add(tf.keras.layers.Permute((2,1)))
            
            
        
    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._dense_depth, hparam_name=hparam_name, count_if_not_list_name='dense_depth', **kwargs)
    
    def call(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)



if __name__ == '__main__':
    
    # Everything default
    # test_keras_model_class(Recurrent_Network, save_and_export=False)
    # SUCCESS!
    
    # No dense or output, just an RNN
    # hparams = Recurrent_Network.sample_hparams.copy()
    # hparams.update({
    #     'include_dense_layers': False,
    #     'include_output_layer': False,
    #     'out_features': None,
    #     'final_rnn_return_sequences': True
    # })
    # test_keras_model_class(Recurrent_Network, hparams=hparams, save_and_export=True)
    # SUCCESS!
    
    # RNN and Dense, no output
    # hparams = Recurrent_Network.sample_hparams.copy()
    # hparams.update({
    #     'include_output_layer': False,
    #     'out_features': None,
    #     'final_rnn_return_sequences': True, #False
    #     'permute_output': True #False
    # })
    # test_keras_model_class(Recurrent_Network, hparams=hparams, save_and_export=False)
    # SUCCESS!
    
    # Include everything, but have flattened sequences
    # hparams = Recurrent_Network.sample_hparams.copy()
    # hparams.update({
    #     'final_rnn_return_sequences': True, #False
    #     'flatten_after_rnn': True, #False
    #     'permute_output': False #True
    # })
    # test_keras_model_class(Recurrent_Network, hparams=hparams, save_and_export=False)
    # SUCCESS!
    
    pass
    
    
    