from vae import VAE 
import util
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.datasets import mnist
import tensorboard
import matplotlib.pyplot as plt
import datetime
from math import ceil


### LOGGING ###

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()

### HYPERPARAMETERS ###

BATCHSIZE = 400 
DATASET_REPS = 100
KL_LOSS_WEIGHT = .05
PIXEL_LOSS_WEIGHT = .95

### DATA ###

(X_train, _), _ = mnist.load_data()
X_train = tf.cast(X_train, tf.float32) / 255
n_data = X_train.shape[0]
dataset = tf.data.Dataset.from_tensor_slices(X_train)
dataset = dataset.shuffle(n_data) # TODO redo shuffling in proper location?
dataset = dataset.batch(BATCHSIZE)
dataset = dataset.repeat(DATASET_REPS)
n_batch = ceil((n_data/BATCHSIZE)*DATASET_REPS)
print(f"DATASET SIZE: {n_data}\nBATCHSIZE: {BATCHSIZE}\nDATASET REPS: {DATASET_REPS}")

### MODEL ###

model = VAE(n_latent=6)

# optimizer
optimizer = tf.keras.optimizers.Adam()

### TRAINING ###

def train_step(model, optimizer, X):
    with tf.GradientTape() as tape:
        mu, logvar, z, r  = model(X)
        loss = model.loss(X,z,r, logvar, mu, pixel_loss_weight=PIXEL_LOSS_WEIGHT, KL_loss_weight=KL_LOSS_WEIGHT)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# training loop
i = 1
for X in dataset:
    X = X[:,:,:,None]
    print(f"BATCH: {i}/{n_batch}, NUM IMGS: {X.shape[0]}", end='\r')
    loss, pix, kl = train_step(model, optimizer, X)
    i += 1
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=i)
        tf.summary.scalar('KL loss', kl, step=i)
        tf.summary.scalar('pixel loss', pix, step=i)

xplot = X_train[:10,:,:].numpy()
_, _, _, reconst = model(xplot[:,:,:,None]) # add dimension for colour channel
reconst = reconst.numpy().squeeze()

model.save_weights("./mnist_vae_6d" + current_time)

plt.figure()
util.plot_MNIST_images(xplot, reconst)
plt.show()
