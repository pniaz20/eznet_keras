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


class Recurrent_Network(KerasSmartModel):
    
    sample_hparams = {
        'model_name': 'Recurrent_Network',
        'in_features': 10,
        'out_features': 3,
        'in_seq_len': 13,
        'out_seq_len': 1,
        'rnn_type': 'LSTM',
        'rnn_hidden_sizes': 8,
        'rnn_bidirectional': False,
        'rnn_depth': 2,
        'rnn_dropout': 0.1,
        'rnn_params': None,
        'final_rnn_return_sequences': False,
        'apply_dense_for_each_time_step': True,
        'permute_output': False,
        'dense_width': 16,
        'dense_depth': 2,
        'dense_dropout': 0.2,
        'dense_activation': 'relu',
        'dense_activation_params': None,
        'output_activation': None,
        'output_activation_params': None,
        'norm_layer_type': 'BatchNormalization',
        'norm_layer_params': None,
        'norm_layer_position': 'before',
        'l2_reg': 0.0001,
        'batch_size': 16,
        'epochs': 2,
        'validation_data': [0.05,'testset'],
        'validation_tolerance_epochs': 10,
        'learning_rate': 0.0001,
        'learning_rate_decay_gamma': 0.99,
        'loss_function': 'categorical_crossentropy',
        'loss_function_params': None,
        'metrics':['accuracy'],
        'optimizer': 'Adam',
        'optimizer_params': None,
        'checkpoint_path':None,
        'early_stopping_monitor':'loss',
        'early_stopping_mode':'min',
        'early_stopping_value':1.0e-6
    }
    
    def __init__(self, hparams:dict=None):
        """Sequence to Dense network with RNN for time-series classification, regression, and forecasting, as well as NLP applications.
        This network uses any RNN layers as encoders to extract information from input sequences, and fully-connected 
        multilayer perceptrons (Dense) to decode the sequence into an output, which can be class probabilitites 
        (timeseries classification), a continuous number (regression), or an unfolded sequence (forecasting) of a 
        target timeseries.

        ### Usage

        `net = Recurrent_Network(hparams)` where `hparams` is dictionary of hyperparameters containing the following:
        
            - `model_name` (str): Name of the model, can be used later for saving, etc.
            - `rnn_type` (str): RNN type, options are "LSTM", "GRU", "SimpleRNN", etc.
            - `in_seq_len` (int): Input sequence length, in number of timesteps
            - `out_seq_len` (int): Output sequence length, in number of timesteps, assuming output is also a sequence. This will affect the output layer in the dense section.
                Use 1 for when the output is not a sequence, or do not supply this key.
            - `in_features` (int): Number of features of the input.
            - `out_features` (int): Number of features of the output.
            - `rnn_hidden_sizes` (int): RNN layer hidden size. A number sets them all the same. Default is 16.
            - `rnn_bidirectional` (bool): Whether the RNN layers are bidirectional or not. Default is False.
            - `rnn_depth` (int): Number of stacked RNN layers. Default is 1.
            - `rnn_dropout` (float): Dropout rates, if any, of the RNN layers. 
                Please note that using dropout in RNN layers is generally discouraged, for it decreases determinism during inference.
            - `rnn_params` (dict): Dictionary of kwargs for the RNN layer constructor. Default is None.
              If specified, the keys in this dictionary will not only add to, but also overwrite any existing arguments this class passes to the RNN layer constructor.
            - `final_rnn_return_sequences` (bool): Whether the final RNN layer returns sequences of hidden state. 
                **NOTE** Setting this to True will make the model much, much larger.
            - `apply_dense_for_each_time_step` (bool): Whether to apply the Dense network to each time step of the 
                RNN output. If False, the Dense network is applied to the last time step only if 
                `final_rnn_retrurn_sequences` is False, or applied to a flattened version of the output sequence
                otherwise (the dimensionality of the input feature space to the dense network will be multiplied
                by the sequence length. PLEASE NOTE that this only works if the entered sequence is exactly as long
                as the priorly defined sequence length according to the hyperparameters).
            - `permute_output` (bool): Whether to permute the output sequence to be (N, D*H_out, L_out)
            - `dense_width` (int|list): (list of) Widths of the Dense network. It can be a number (for all) or a list holding width of each hidden layer.
            - `dense_depth` (int): Depth (number of hidden layers) of the Dense network.
            - `dense_activation` (str|list): (list of) Activation functions for hidden layers of the Dense network. Examples: "relu", "leaklyrelu", "sigmoid", "tanh", etc.
            - `dense_activation_params` (dict|list): (list of) Dictionaries of parameters for the activation func constructors of the Dense network.
            - `output_activation` (str): Activation function for the output layer of the Dense network, if any. Examples: "softmax", "sigmoid", etc.
                **NOTE** If the loss function is `sparse_categorical_crossentropy`, then no output activation is erquired.
                However, if the loss function is `categorical_crossentropy` (a.k.a. negative loglikelihood), then you must specify an output activation as in "softmax".
            - `output_activation_params` (dict): Dictionary of parameters for the activation func constructor of the output layer.
            - `norm_layer_type` (str|list): (list of) Types of normalization layers to use in the dense section, if any. Options are "BatchNormalization", "LayerNormalization",etc.
            - `norm_layer_params` (dict|list): (list of) Dictionaries of parameters for the normalization layer constructors.
            - `norm_layer_position` (str|list): (list of) Whether the normalization layer should come 'before' or 'after' the activation of each hidden layer in the dense network.
            - `dense_dropout` (float|list): (list of) Dropout rates (if any) for the hidden layers of the Dense network.
            - `batch_size` (int): Minibatch size, the expected input size of the network.
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer. Examples: 'Adam', 'SGD', 'RMSProp', etc.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `validation_data` (list): Portion of validation data. Should be a tuple like [validation split, dataset as in 'trainset' or 'testset']
            - `l2_reg` (float): L2 regularization parameter.
            - `loss_function` (str): Loss function. Examples: 'binary_crossentropy', 'categorical_crossentropy', 'mse', etc.
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.
            - `metrics` (list): list of metrics for Keras compilation.
            - `checkpoint_path` (str): Path to the directory where checkpoints will be saved at every epoch.
            - `early_stopping_monitor` (str): Monitor whose critical value will cause early stopping. Default is 'loss', but 'val_loss' is typically used.
            - `early_stopping_mode` (str): Mode of the parameter whose critical value will be used for early stopping. Deafults to 'min' for any error. 'max' is for accuracy, etc.
            - `early_stopping_value` (float): Value of the monitor at which point training will stop becasue the critical value has been reached.

        ### Returns
        
        Returns a `tf.keras.models.Model` object that can be trained and used accordingly.
        Run `net.summary()` afterwards to see what you have inside the network.
        The returned model is an instance of `KerasSmartModel`, which has built-in functions for training, evaluation, etc.
        """
        super(Recurrent_Network, self).__init__(hparams)
        hparams = hparams if hparams is not None else self.sample_hparams
        self._rnn_type = hparams["rnn_type"]
        self._rnn = getattr(tf.keras.layers, self._rnn_type)
        self._denseactivation = hparams["dense_activation"] if hparams.get("dense_activation") else "relu"
        self._denseactivation_params = hparams.get("dense_activation_params")
        self._outactivation = hparams.get("output_activation")
        self._outactivation_params = hparams.get("output_activation_params")
        self._normlayer_type = hparams.get("norm_layer_type")
        self._normlayer_params = hparams.get("norm_layer_params")
        self._normlayer_position = hparams.get("norm_layer_position")
        self._infeatures = hparams["in_features"]
        self._outfeatures = hparams["out_features"]
        self._rnnhidsizes = hparams["rnn_hidden_sizes"] if hparams.get("rnn_hidden_sizes") else 16
        self._densehidsizes = hparams["dense_width"] if hparams.get("dense_width") else 16
        self._densedepth = hparams["dense_depth"] if hparams.get("dense_depth") else 0
        self._rnndepth = hparams["rnn_depth"] if hparams.get("rnn_depth") else 1
        self._bidirectional = True if hparams.get("rnn_bidirectional") else False
        self._rnndropout = hparams["rnn_dropout"] if hparams.get("rnn_dropout") else 0
        self._densedropout = hparams["dense_dropout"] if hparams.get("dense_dropout") else None
        self._final_rnn_return_sequences = True if hparams.get("final_rnn_return_sequences") else False
        self._apply_dense_for_each_timestep = True if hparams.get("apply_dense_for_each_timestep") else False
        self._permute_output = True if hparams.get("permute_output") else False
        self._N = int(hparams["batch_size"])
        self._L_in = int(hparams["in_seq_len"])
        self._L_out = int(hparams["out_seq_len"]) if hparams.get("out_seq_len") else 1
        self._D = int(2 if self._bidirectional else 1)
        self._rnn_params = hparams.get("rnn_params")
        self._H_in = int(self._infeatures)
        if self._rnnhidsizes == "auto": self._H_cell = int(2**(np.round(math.log2(self._H_in*self._L_in))))
        else: self._H_cell = int(self._rnnhidsizes)
        self._H_out = int(self._H_cell)
        self.batch_input_shape = (self._N, self._L_in, self._H_in)
        if self._final_rnn_return_sequences and self._apply_dense_for_each_timestep:
            if self._permute_output: self.batch_output_shape = (self._N, self._outfeatures, self._L_out)
            else: self.batch_output_shape = (self._N, self._L_out, self._outfeatures)
        else: self.batch_output_shape = (self._N, self._L_out * self._outfeatures)
        self._out_activation_module = actdict_keras[self._outactivation] if self._outactivation else None
        
        # Initializing sequential network
        self.net = tf.keras.models.Sequential()
        
        # Constructing RNN layers
        _kwargs = {'units':self._H_cell, 'dropout':self._rnndropout, 
            'return_sequences':(True if self._rnndepth > 1 else self._final_rnn_return_sequences), 
            'kernel_regularizer':(tf.keras.regularizers.L2(self._l2_reg) if self._l2_reg else None)}
        if self._rnn_params: 
            _kwargs.update(self._rnn_params)
        self.net.add(tf.keras.Input((self._L_in,self._H_in)))
        for i in range(self._rnndepth):
            if i != self._rnndepth-1:
                _kwargs.update({'return_sequences':True})
            else:
                _kwargs.update({'return_sequences':self._final_rnn_return_sequences})
            self.net.add(tf.keras.layers.Bidirectional(self._rnn(**_kwargs)) if self._bidirectional else self._rnn(**_kwargs))
        
        # If the final RNN layer returns sequences, and we are NOT applying the dense on each time step of it, then the returned sequence must be flattened before being decocded.
        if self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep:
            self.net.add(tf.keras.layers.Flatten())
        
        # Calculating Dense layers widths
        cf = self._L_in if (self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep) else 1 
        self._dense_input_size = self._H_out * self._D * cf
        if self._final_rnn_return_sequences and not self._apply_dense_for_each_timestep:
            self._dense_output_size = int(self._L_out*self._outfeatures)
        else:
            self._dense_output_size = int(self._outfeatures)
            
        # Generate arrays containing parameters of each Dense Block (Every block contains a linear, normalization, activation, and dropout layer).
        self._dense_width_vec = self._gen_hparam_vec_for_dense(self._densehidsizes, 'dense_width')
        self._dense_activation_vec = self._gen_hparam_vec_for_dense(self._denseactivation, 'dense_activation')
        self._dense_activation_params_vec = self._gen_hparam_vec_for_dense(self._denseactivation_params, 'dense_activation_params')
        self._dense_norm_layer_type_vec = self._gen_hparam_vec_for_dense(self._normlayer_type, 'norm_layer_type')
        self._dense_norm_layer_params_vec = self._gen_hparam_vec_for_dense(self._normlayer_params, 'norm_layer_params')
        self._dense_norm_layer_position_vec = self._gen_hparam_vec_for_dense(self._normlayer_position, 'norm_layer_position')
        self._dense_dropout_vec = self._gen_hparam_vec_for_dense(self._densedropout, 'dense_dropout')
        
        # Construct the dense layers
        in_size = self._dense_input_size
        for i in range(self._densedepth):
            out_size = self._dense_width_vec[i]
            temp_dropout_rate = self._dense_dropout_vec[i] if (i != self._densedepth-1) else None # The hidden layer just before the output layer rarely has Dropout.
            add_dense_block(self.net, output_size=out_size, input_shape=None, activation=self._dense_activation_vec[i], activation_params=self._dense_activation_params_vec[i], 
                    norm_layer_type=self._dense_norm_layer_type_vec[i], norm_layer_position=self._dense_norm_layer_position_vec[i], 
                    norm_layer_params=self._dense_norm_layer_params_vec[i], dropout=temp_dropout_rate, 
                    kernel_regularizer=(tf.keras.regularizers.L2(self._l2_reg) if self._l2_reg else None))
            in_size = out_size
        
        # Output layer
        self.net.add(tf.keras.layers.Dense(self._dense_output_size))
        if self._outactivation:
            if self._outactivation_params: self.net.add(tf.keras.layers.Activation(self._out_activation_module(**self._outactivation_params)))
            else: self.net.add(tf.keras.layers.Activation(self._out_activation_module))
            
        # Permute if necessary
        if self._final_rnn_return_sequences and self._apply_dense_for_each_timestep and self._permute_output:
            self.net.add(tf.keras.layers.Permute((2,1)))
        
    def _gen_hparam_vec_for_dense(self, hparam, hparam_name, **kwargs):
        return generate_array_for_hparam(hparam, self._densedepth, hparam_name=hparam_name, count_if_not_list_name='dense_depth', **kwargs)
    
    def call(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)



if __name__ == '__main__':
    hparams = Recurrent_Network.sample_hparams
    hparams['rnn_bidirectional'] = True
    model = Recurrent_Network(hparams)
    model.summary()