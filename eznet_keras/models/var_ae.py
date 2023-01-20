
if __package__=="eznet_keras.models":
    from ..utils import *
else:
    import os, sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    from utils import *


# Variational Autoencoder with Keras
class Keras_VarAE(tf.keras.Model):
    def __init__(self, hparams, **kwargs):
        super(Keras_VarAE, self).__init__(**kwargs)

        self.num_features = hparams["num_features"]
        self.hidden_size = hparams["hidden_size"]
        self.latent_size = hparams["latent_size"]
        self.encoder_depth = hparams["encoder_depth"]
        self.encoder_activations = hparams["encoder_activations"]
        self.decoder_depth = hparams["decoder_depth"]
        self.decoder_activations = hparams["decoder_activations"]
        self.learning_rate = hparams["learning_rate"]
        self.learning_rate_decay_gamma = hparams.get("learning_rate_decay_gamma")
        self.optimizer = hparams["optimizer"]
        self.optimizer_params = hparams.get("optimizer_params")
        self.batch_size = hparams["batch_size"]
        self.epochs = hparams["epochs"]
        self.validation_tolerance_epochs = hparams.get("validation_tolerance_epochs")

        #self.inputlayer = Input(shape=(self.num_features,))
        self.encoder = tf.keras.models.Sequential(name="encoder")
        self.encoder.add(tf.keras.layers.Dense(
            self.hidden_size, input_shape=(self.num_features,), activation=self.encoder_activations,
            name="encoder_dense_1"))
        if self.encoder_depth > 1:
            for i in range(self.encoder_depth - 1):
                self.encoder.add(tf.keras.layers.Dense(self.hidden_size, activation=self.encoder_activations, 
                name="encoder_dense_" + str(i+2)))

        #self.encoder2mean = Sequential()
        #self.encoder2logvar = Sequential()
        #self.encoder2mean.add(Dense(self.latent_size, name="z_mean", input_shape=(self.hidden_size,)))
        #self.encoder2logvar.add(Dense(self.latent_size, name="z_logvar", input_shape=(self.hidden_size,)))
        self.encoder2mean = tf.keras.layers.Dense(self.latent_size, name="z_mean", input_shape=(self.hidden_size,))
        self.encoder2logvar = tf.keras.layers.Dense(self.latent_size, name="z_logvar", input_shape=(self.hidden_size,))
        
        self.decoder = tf.keras.models.Sequential(name="decoder")
        self.decoder.add(tf.keras.layers.Dense(
            self.hidden_size, input_shape=(self.latent_size,), activation=self.decoder_activations, 
            name="decoder_dense_1"))
        if self.decoder_depth > 1:
            for i in range(self.decoder_depth - 1):
                self.decoder.add(tf.keras.layers.Dense(
                    self.hidden_size, activation=self.decoder_activations, name="decoder_dense_" + str(i+2)))
        self.decoder.add(tf.keras.layers.Dense(self.num_features, activation="linear", name="decoder_dense_output"))

        #hid = self.encoder(self.inputlayer)
        #self.mu = self.encoder2mean(hid)
        #self.log_var = self.encoder2logvar(hid)
        #self.z = self.reparameterize(self.mu, self.log_var)
        #self.out_model = self.decoder(self.z)
        #self.out = Concatenate()([self.out_model, self.mu, self.log_var])
        #self.net = Model(self.inputlayer, self.out)
        

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            hid = self.encoder(data)
            z_mean = self.encoder2mean(hid)
            z_log_var = self.encoder2logvar(hid)
            z = Sampling()([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(data, reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    
    def call(self, inputs):
        hid = self.encoder(inputs)
        z_mean = self.encoder2mean(hid)
        z_log_var = self.encoder2logvar(hid)
        z = Sampling()([z_mean, z_log_var])
        reconstruction = self.decoder(z)
        #return reconstruction, z_mean, z_log_var
        return reconstruction    

