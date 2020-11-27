#%%

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
import tensorboard
import matplotlib.pyplot as plt
import datetime

tfd = tfp.distributions

class VAE(tf.keras.Model):
    def __init__(self, n_latent=8):
        super(VAE, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, activation='relu') # output shape: 50,50,3
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation='relu') # output shape: 50,50,3
        # self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu') # output shape: 50,50,3
        # self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, activation='relu') # output shape: 50,50,3
        
        self.N = n_latent

        self.mu_dense = tf.keras.layers.Dense(self.N, activation='relu')
        self.sig_dense = tf.keras.layers.Dense(self.N, activation='relu')

        self.normal = tfd.Normal(loc=0, scale=1)

        self.reconst_dense = tf.keras.layers.Dense(25*16)

        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, activation='relu')
        #self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation='relu')
        #self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2, activation='sigmoid')

    def encode(self, x):
      batchsize = x.shape[0]

      x = self.conv1(x)
      x = self.conv2(x)
      # x = self.conv3(x)
      # x = self.conv4(x)
      x = tf.reshape(x, [-1, 6*6*16]) # flatten output of convolutions
      mu = self.mu_dense(x)
      log_var = self.sig_dense(x)
      sig = tf.exp(log_var / 2.0) # from world models implementation, why is mu encoded directly and var assumed to be logit output?
      # sample from normal
      s = self.normal.sample([batchsize,self.N])

      # latent activation
      z = mu + sig * s # TODO: double check that this works as intended, shape checks out though

      return mu, log_var, z

    def decode(self, z):
      # adopted from world models git
      h = self.reconst_dense(z)
      print(f"H {h.shape}")
      h = tf.reshape(h, [-1, 1, 1, 400])
      h = self.deconv1(h)
      reconstruction = self.deconv2(h)
      # h = self.deconv3(h)
      # reconstruction = self.deconv4(h)

      return reconstruction

    def call(self,x):
      mu, logvar, z = self.encode(x)
      r = self.decode(z)

      return mu, logvar, z, r

    def loss(self, x, z, r, logvar, mu):
      sig = tf.exp(logvar / 2.0)

      # reconstruction loss: sum of squares difference of pixel values
      ssqr = tf.math.square(x - r)
      reconstruction_loss = tf.math.reduce_sum(ssqr)

      # KL divergence with normal distribution
      kl_div = 0.5*tf.square(z) - logvar - (1/(2*sig))*tf.math.square(z-mu)
      kl_loss = tf.math.reduce_sum(kl_div)

      return reconstruction_loss + kl_loss


class TestModel(tf.keras.Model):
    def __init__(self, n_latent=8):
        super(TestModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=3, activation='relu') 
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')
        
        self.N = n_latent

        self.mu_dense = tf.keras.layers.Dense(self.N, activation='relu')
        self.sig_dense = tf.keras.layers.Dense(self.N, activation='relu')

        self.normal = tfd.Normal(loc=0, scale=1)
        
        self.reconst_dense = tf.keras.layers.Dense(32, activation='relu')

        self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=1, activation='relu')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=3, activation='relu')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=3, activation='sigmoid')

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
        x = tf.reshape(x, [-1, 1, 1, 32])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

    def call(self,x):
        mu, logvar, z = self.encode(x)
        r = self.decode(z)
        return mu, logvar, z, r


def train_step(model, optimizer, X):
  with tf.GradientTape() as tape:
    mu, logvar, z, r  = model(X)
    loss = model.loss(X,z,r, logvar, mu)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_loss(loss)
  return loss

n_epochs = 1000

# set up logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

# load data
(X_train, _), _ = mnist.load_data()
X_train = tf.cast(X_train, tf.float32) / 255
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(60000)
dataset = dataset.batch(1000)
dataset = dataset.repeat(n_epochs)

model = TestModel()
test_datum = X_train[0,:,:]
test_datum = test_datum[None,:,:,None]
print("test data shape: {test_datum.shape}")
mu, logvar, z, r = model(test_datum)
##%% 

## model
#model = VAE()

## optimizer
#optimizer = tf.keras.optimizers.SGD()

## training loop
#for epoch in range(n_epochs):
#  epochloss = 0
#  print(f"EPOCH: {epoch}")
#  for X in dataset:
#    X = X[:,:,:,None]
#    epochloss += train_step(model, optimizer, X)
#  with train_summary_writer.as_default():
# tf.summary.scalar('loss', epochloss, step=epoch)
