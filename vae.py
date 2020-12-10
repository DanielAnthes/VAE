import util
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
import tensorboard
import matplotlib.pyplot as plt
import datetime
from math import ceil

tfd = tfp.distributions

class VAE(tf.keras.Model):

    def __init__(self, n_latent=8):
        super(VAE, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=3, activation='relu') 
        self.conv2 = tf.keras.layers.Conv2D(filters=4, kernel_size=3, strides=3, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=1, activation='relu')
        
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
        x = tf.reshape(x, [-1, 8]) # flatten output of convolutions
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
      kl_div = 0.5*tf.square(z) - logvar - (1/(2*sig))*tf.math.square(z-mu)
      kl_loss = tf.math.reduce_sum(kl_div)

      return reconstruction_loss + kl_loss


def train_step(model, optimizer, X, pixel_weight=.5, KL_weight=.5):
    with tf.GradientTape() as tape:
        mu, logvar, z, r  = model(X)
        loss, pix, kl = model.loss(X,z,r, logvar, mu, pixel_loss_weight=pixel_weight, KL_loss_weight=KL_weight)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    KL_loss(kl)
    pixel_loss(pix)
    return loss


# set up logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
KL_loss = tf.keras,metrics.Mean('KL_loss', dtype=tf.float32)
pixel_loss = tf.keras,metrics.Mean('pixel_loss', dtype=tf.float32)

### HYPERPARAMETERS ###

BATCHSIZE = 400 
DATASET_REPS = 100
KL_LOSS_WEIGHT = 0
PIXEL_LOSS_WEIGHT = 1


(X_train, _), _ = mnist.load_data()
X_train = tf.cast(X_train, tf.float32) / 255
n_data = X_train.shape[0]
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(n_data) # TODO redo shuffling in proper location?
dataset = dataset.batch(BATCHSIZE)
dataset = dataset.repeat(DATASET_REPS)
n_batch = ceil((n_data/BATCHSIZE)*DATASET_REPS)
print(f"DATASET SIZE: {n_data}\nBATCHSIZE: {BATCHSIZE}\nDATASET REPS: {DATASET_REPS}")

# model
model = VAE(n_latent=16)

# optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

# training loop
i = 1
for X in dataset:
    X = X[:,:,:,None]
    print(f"BATCH: {i}/{n_batch}, NUM IMGS: {X.shape[0]}", end='\r')
    loss = train_step(model, optimizer, X, PIXEL_LOSS_WEIGHT, KL_LOSS_WEIGHT)
    i += 1
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=i)
        tf.summary.scalar('KL loss', KL_loss, step=i)
        tf.summary.scalar('pixel loss', pixel_loss, step=i)

xplot = X_train[:10,:,:].numpy()
_, _, _, reconst = model(xplot[:,:,:,None]) # add dimension for colour channel
reconst = reconst.numpy().squeeze()

plt.figure()
util.plot_MNIST_images(xplot, reconst)
plt.show()
