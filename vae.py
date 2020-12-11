import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class VAE(tf.keras.Model):
    def __init__(self, n_latent=8):
        super(VAE, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=3, activation='relu') 
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
        
        self.N = n_latent

        self.mu_dense = tf.keras.layers.Dense(self.N, activation='relu')
        self.sig_dense = tf.keras.layers.Dense(self.N, activation='relu')

        self.normal = tfd.Normal(loc=0, scale=1)
        
        self.reconst_dense = tf.keras.layers.Dense(8, activation='relu')

        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=4, strides=1, activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=3, strides=2, activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=3, activation='sigmoid')

    def encode(self, x):
        batchsize = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.reshape(x, [-1, 32]) # flatten output of convolutions
        mu = self.mu_dense(x)
        log_var = self.sig_dense(x)
        sig = tf.exp(log_var / 2.0) # from world models implementation, why is mu encoded directly and var assumed to be logit output?
        # sample from normal
        s = self.normal.sample([batchsize,self.N])

        # latent activation
        z = mu + sig * s # TODO: double check that this works as intended, shape checks out though
        return mu, log_var, z

    def decode(self, x):
        x = self.reconst_dense(x)
        x = tf.reshape(x, [-1, 1, 1, 8])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

    def call(self,x):
        mu, logvar, z = self.encode(x)
        r = self.decode(z)
        return mu, logvar, z, r

    def loss(self, x, z, r, logvar, mu, pixel_loss_weight=.5, KL_loss_weight=.5):
      sig = tf.exp(logvar / 2.0)

      # reconstruction loss: sum of squares difference of pixel values
      ssqr = tf.math.square(x - r)
      reconstruction_loss = tf.math.reduce_sum(ssqr)

      # KL divergence with normal distribution
      # kl_div = 0.5*tf.square(z) - logvar - (1/(2*sig))*tf.math.square(z-mu)
      kl_div = 0.5*(tf.exp(logvar) + mu**2 - 1 - logvar)
      kl_loss = tf.math.reduce_sum(kl_div)

      return reconstruction_loss + kl_loss, reconstruction_loss, kl_loss



